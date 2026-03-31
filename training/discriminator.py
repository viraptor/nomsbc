"""
Multi-resolution STFT discriminator for LS-GAN training.

Uses multiple STFT resolutions with frequency-axis striding to maintain
constant receptive fields, improving detection of inter-harmonic noise
(per NoLACE paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTDiscriminator(nn.Module):
    """Single-resolution STFT discriminator."""

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024,
                 channels=32, n_layers=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer(
            'window', torch.hann_window(win_length)
        )

        # Input: 2 channels (real + imag) of STFT
        n_bins = n_fft // 2 + 1
        layers = [
            nn.Conv2d(2, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(n_layers - 1):
            in_ch = channels * (2 ** i) if i > 0 else channels
            out_ch = channels * (2 ** (i + 1))
            # Stride along frequency axis to keep temporal receptive field constant
            layers += [
                nn.Conv2d(in_ch, out_ch, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ]
        layers += [
            nn.Conv2d(out_ch, 1, (3, 3), padding=(1, 1)),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, samples)
        Returns: (batch, T', F') discriminator logits
        """
        # Compute STFT
        spec = torch.stft(x, self.n_fft, self.hop_length, self.win_length,
                          self.window, return_complex=True)
        # Stack real and imaginary as channels: (B, 2, F, T)
        spec = torch.stack([spec.real, spec.imag], dim=1)
        return self.net(spec)


class MultiResolutionSTFTDiscriminator(nn.Module):
    """
    Ensemble of STFT discriminators at different resolutions.

    Default resolutions follow common practice:
      - (1024, 256) for broad spectral view
      - (512, 128)  for medium resolution
      - (256, 64)   for fine temporal detail
    """

    def __init__(self, resolutions=None, channels=32, n_layers=4):
        super().__init__()
        if resolutions is None:
            resolutions = [
                (1024, 256, 1024),
                (512, 128, 512),
                (256, 64, 256),
            ]
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft, hop, win, channels, n_layers)
            for n_fft, hop, win in resolutions
        ])

    def forward(self, x):
        """Returns list of discriminator outputs, one per resolution."""
        return [d(x) for d in self.discriminators]


if __name__ == '__main__':
    disc = MultiResolutionSTFTDiscriminator()
    n_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {n_params:,}")

    x = torch.randn(2, 16000)  # 1 second at 16 kHz
    outputs = disc(x)
    for i, out in enumerate(outputs):
        print(f"Resolution {i}: output shape {out.shape}")
