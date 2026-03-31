"""
Loss functions for NoLACE-mSBC training.

Primary: LS-GAN losses (generator + discriminator).
Auxiliary: multi-resolution STFT loss for stability.
"""

import torch
import torch.nn.functional as F


def ls_discriminator_loss(real_outputs, fake_outputs):
    """
    Least-squares discriminator loss.
    real_outputs, fake_outputs: lists of discriminator outputs (multi-resolution).
    """
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean((real - 1) ** 2)
        loss += torch.mean(fake ** 2)
    return loss / len(real_outputs)


def ls_generator_loss(fake_outputs):
    """
    Least-squares generator loss: fake should look real.
    """
    loss = 0
    for fake in fake_outputs:
        loss += torch.mean((fake - 1) ** 2)
    return loss / len(fake_outputs)


def feature_matching_loss(real_features, fake_features):
    """
    Feature matching loss across discriminator layers.
    real_features, fake_features: lists of lists of intermediate activations.
    """
    loss = 0
    count = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
            count += 1
    return loss / max(count, 1)


def multi_resolution_stft_loss(predicted, target,
                                fft_sizes=(512, 1024, 2048),
                                hop_sizes=(128, 256, 512),
                                win_sizes=(512, 1024, 2048)):
    """
    Multi-resolution STFT loss: spectral convergence + log magnitude.
    Provides a stable auxiliary loss alongside the GAN loss.
    """
    loss = 0
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win, device=predicted.device)
        pred_stft = torch.stft(predicted, n_fft, hop, win, window,
                               return_complex=True).abs()
        targ_stft = torch.stft(target, n_fft, hop, win, window,
                               return_complex=True).abs()

        # Spectral convergence
        sc = torch.norm(targ_stft - pred_stft, p='fro') / \
             (torch.norm(targ_stft, p='fro') + 1e-8)

        # Log magnitude loss
        log_mag = F.l1_loss(
            torch.log(pred_stft.clamp(min=1e-7)),
            torch.log(targ_stft.clamp(min=1e-7))
        )

        loss += sc + log_mag

    return loss / len(fft_sizes)
