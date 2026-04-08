"""
NoLACE-mSBC: PyTorch model matching the C inference pipeline.

The DNN takes blind features as input and predicts parameters for each
DDSP module (AdaComb x2, AdaConv, AdaShape x2, BWE) per frame.
"""

import torch
import torch.nn as nn

# Must match C defines
FEATURE_DIM = 28
FRAME_SIZE = 160
ADACOMB_KERNEL = 5
ADACONV_KERNEL = 16
ADASHAPE_BASES = 4
ADASHAPE_SHAPE_DIM = 8
BWE_ENVELOPE_DIM = 16

# Total output dimension per frame
PARAM_DIM = (
    (ADACOMB_KERNEL + 1) * 2  # comb1 + comb2: kernel + gain each
    + ADACONV_KERNEL + 1       # conv: kernel + gain
    + (ADASHAPE_BASES + ADASHAPE_SHAPE_DIM + 1) * 2  # shape1 + shape2
    + BWE_ENVELOPE_DIM + 1 + 1  # bwe: envelope + excitation_gain + voicing
)


class FeatureEncoder(nn.Module):
    """Processes blind features into a latent representation."""

    def __init__(self, feat_dim=FEATURE_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class FrameGRU(nn.Module):
    """Temporal modeling across frames using a GRU."""

    def __init__(self, input_dim=128, hidden_dim=192, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True)

    def forward(self, x, state=None):
        out, state = self.gru(x, state)
        return out, state


class ParameterHead(nn.Module):
    """Predicts DDSP module parameters from the GRU output."""

    def __init__(self, hidden_dim=192, param_dim=PARAM_DIM):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, param_dim)

    def forward(self, x):
        return self.linear(x)


class NoLACEmSBC(nn.Module):
    """
    Full model: features -> GRU -> module parameters.

    During training, the DDSP modules run differentiably in Python.
    For inference, weights are exported and the C pipeline runs the
    same modules with the predicted parameters.
    """

    def __init__(self, feat_dim=FEATURE_DIM, enc_hidden=128,
                 gru_hidden=192, gru_layers=2):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim, enc_hidden)
        self.gru = FrameGRU(enc_hidden, gru_hidden, gru_layers)
        self.head = ParameterHead(gru_hidden, PARAM_DIM)

    def forward(self, features, state=None):
        """
        Args:
            features: (batch, num_frames, FEATURE_DIM)
            state: optional GRU hidden state
        Returns:
            params: (batch, num_frames, PARAM_DIM)
            state: updated GRU state
        """
        x = self.encoder(features)
        x, state = self.gru(x, state)
        params = self.head(x)
        return params, state

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- Differentiable DDSP modules for training ---

def apply_adacomb_diff(signal, pitch_lag, kernel, gain):
    """
    Differentiable adaptive comb filter.
    signal: (batch, frame_size)
    pitch_lag: (batch,) int
    kernel: (batch, ADACOMB_KERNEL)
    gain: (batch, 1)
    """
    B, T = signal.shape
    half = ADACOMB_KERNEL // 2
    padded = torch.nn.functional.pad(signal, (ADACOMB_KERNEL + 320, 0))

    output = signal.clone()
    # Vectorized: gather delayed samples and convolve with kernel
    for b in range(B):
        lag = pitch_lag[b].item()
        for k in range(-half, half + 1):
            delay = lag - k
            if delay > 0 and delay <= padded.shape[1] - T:
                idx_start = padded.shape[1] - T - delay
                delayed = padded[b, idx_start:idx_start + T]
                output[b] += gain[b] * kernel[b, k + half] * delayed
    return output


def apply_adaconv_diff(signal, kernel, gain):
    """
    Differentiable adaptive (per-sample-kernel) FIR.

    signal: (batch, frame_size)
    kernel: (batch, ADACONV_KERNEL)  -- per-batch FIR taps
    gain:   (batch, 1)

    Implemented as pad + unfold + einsum, which on MPS is dramatically
    faster than F.conv1d with groups=batch when batch is large
    (e.g. B*T = 1600 in vectorized training).
    """
    B, T = signal.shape
    K = ADACONV_KERNEL
    sig_padded = torch.nn.functional.pad(signal, (K - 1, 0))   # (B, T+K-1)
    windows = sig_padded.unfold(-1, K, 1)                       # (B, T, K)
    # Convolution = correlation with flipped kernel.
    kern = kernel.flip(-1)                                      # (B, K)
    out = torch.einsum('btk,bk->bt', windows, kern)             # (B, T)
    return out * gain


def apply_adashape_diff(signal, pitch_lag, select_w, shape_p, mix_gain):
    """
    Differentiable adaptive temporal shaping.
    signal: (batch, frame_size)
    select_w: (batch, NUM_BASES)
    shape_p: (batch, SHAPE_DIM) -- pairs of (a, b) for tanh basis
    mix_gain: (batch, 1)
    """
    B, T = signal.shape
    padded = torch.nn.functional.pad(signal, (320, 0))
    offset = 320

    # Basis signals
    bases = torch.stack([
        signal,
        padded[:, offset - pitch_lag.clamp(min=1).unsqueeze(1).expand(-1, T)
               + torch.arange(T, device=signal.device)].clone()
        if False else signal,  # simplified: use signal as placeholder
        signal,
        signal,
    ], dim=-1)  # (B, T, NUM_BASES)

    selected = (bases * select_w.unsqueeze(1)).sum(-1)  # (B, T)

    # Shaping: sum of tanh activations
    shaped = torch.zeros_like(selected)
    for i in range(0, ADASHAPE_SHAPE_DIM, 2):
        a = shape_p[:, i:i+1]
        b = shape_p[:, i+1:i+2]
        shaped = shaped + a * torch.tanh(b * selected)

    return signal + mix_gain * (shaped - signal)


def unpack_params(params):
    """
    Unpack flat parameter vector into per-module parameters.
    params: (batch, PARAM_DIM)
    Returns dict of parameter tensors.
    """
    idx = 0

    def take(n):
        nonlocal idx
        out = params[:, idx:idx + n]
        idx += n
        return out

    return {
        'comb1_kernel': take(ADACOMB_KERNEL),
        'comb1_gain': torch.sigmoid(take(1)),
        'comb2_kernel': take(ADACOMB_KERNEL),
        'comb2_gain': torch.sigmoid(take(1)),
        'conv_kernel': take(ADACONV_KERNEL),
        'conv_gain': torch.sigmoid(take(1)) * 2,  # allow up to 2x gain
        'shape1_select': torch.softmax(take(ADASHAPE_BASES), dim=-1),
        'shape1_shape': take(ADASHAPE_SHAPE_DIM),
        'shape1_mix': torch.sigmoid(take(1)),
        'shape2_select': torch.softmax(take(ADASHAPE_BASES), dim=-1),
        'shape2_shape': take(ADASHAPE_SHAPE_DIM),
        'shape2_mix': torch.sigmoid(take(1)),
        'bwe_envelope': take(BWE_ENVELOPE_DIM),
        'bwe_excitation_gain': torch.sigmoid(take(1)),
        'bwe_voicing': torch.sigmoid(take(1)),
    }


if __name__ == '__main__':
    model = NoLACEmSBC()
    print(f"Model parameters: {model.count_params():,}")
    print(f"Output dim per frame: {PARAM_DIM}")

    # Test forward pass
    batch = torch.randn(2, 50, FEATURE_DIM)
    params, state = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {params.shape}")
    print(f"GRU state shape: {state.shape}")
