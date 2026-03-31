"""
Training loop for NoLACE-mSBC.

LS-GAN training with multi-resolution STFT discriminator.
Follows a two-phase approach:
  Phase 1: Pre-train generator with STFT loss only (stabilizes early training)
  Phase 2: Full GAN training with discriminator
"""

import argparse
import os
import torch
import torch.optim as optim
from pathlib import Path

from model import NoLACEmSBC, unpack_params, apply_adaconv_diff
from discriminator import MultiResolutionSTFTDiscriminator
from losses import (ls_discriminator_loss, ls_generator_loss,
                    multi_resolution_stft_loss)
from dataset import create_dataloader


def build_enhanced_signal(model, batch, device):
    """
    Run the model and differentiable DDSP to produce enhanced signal.

    In full training, this runs through all DDSP modules differentiably.
    For now, uses a simplified path: AdaConv only (most impactful single module).
    """
    features = batch['features'].to(device)
    degraded = batch['degraded_signal'].to(device)
    degraded_frames = batch['degraded_frames'].to(device)

    params_flat, _ = model(features)  # (B, T, PARAM_DIM)

    # Apply per-frame processing
    B, T, _ = params_flat.shape
    enhanced_frames = []
    for t in range(T):
        p = unpack_params(params_flat[:, t, :])
        frame = degraded_frames[:, t, :]

        # Apply AdaConv (most significant spectral correction)
        frame = apply_adaconv_diff(frame, p['conv_kernel'], p['conv_gain'])
        enhanced_frames.append(frame)

    enhanced = torch.stack(enhanced_frames, dim=1)  # (B, T, FRAME_SIZE)
    enhanced_signal = enhanced.reshape(B, -1)
    return enhanced_signal


def train_epoch_pretrain(model, loader, optimizer, device):
    """Phase 1: generator pre-training with STFT loss only."""
    model.train()
    total_loss = 0
    for batch in loader:
        clean = batch['clean_signal'].to(device)
        enhanced = build_enhanced_signal(model, batch, device)

        loss = multi_resolution_stft_loss(enhanced, clean)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_gan(model, disc, loader, opt_g, opt_d, device,
                    lambda_stft=10.0):
    """Phase 2: full LS-GAN training."""
    model.train()
    disc.train()
    total_g_loss = 0
    total_d_loss = 0

    for batch in loader:
        clean = batch['clean_signal'].to(device)
        enhanced = build_enhanced_signal(model, batch, device)

        # --- Discriminator step ---
        real_out = disc(clean)
        fake_out = disc(enhanced.detach())
        d_loss = ls_discriminator_loss(real_out, fake_out)

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # --- Generator step ---
        fake_out = disc(enhanced)
        g_adv_loss = ls_generator_loss(fake_out)
        g_stft_loss = multi_resolution_stft_loss(enhanced, clean)
        g_loss = g_adv_loss + lambda_stft * g_stft_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_g.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    n = len(loader)
    return total_g_loss / n, total_d_loss / n


def main():
    parser = argparse.ArgumentParser(description='Train NoLACE-mSBC')
    parser.add_argument('--data-dir', required=True, help='Training data directory')
    parser.add_argument('--output-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    parser.add_argument('--gan-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=100, help='Frames per sequence')
    parser.add_argument('--lr-g', type=float, default=1e-4)
    parser.add_argument('--lr-d', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    loader = create_dataloader(args.data_dir, args.batch_size, args.seq_len)
    model = NoLACEmSBC().to(device)
    disc = MultiResolutionSTFTDiscriminator().to(device)

    print(f"Generator params: {model.count_params():,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    opt_g = optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    opt_d = optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99))

    # Phase 1: Pre-train with STFT loss
    print("\n--- Phase 1: Pre-training (STFT loss) ---")
    for epoch in range(args.pretrain_epochs):
        loss = train_epoch_pretrain(model, loader, opt_g, device)
        print(f"Epoch {epoch+1}/{args.pretrain_epochs}  STFT loss: {loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       f"{args.output_dir}/pretrain_ep{epoch+1}.pt")

    # Phase 2: GAN training
    print("\n--- Phase 2: GAN training ---")
    for epoch in range(args.gan_epochs):
        g_loss, d_loss = train_epoch_gan(
            model, disc, loader, opt_g, opt_d, device
        )
        print(f"Epoch {epoch+1}/{args.gan_epochs}  "
              f"G loss: {g_loss:.4f}  D loss: {d_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': model.state_dict(),
                'discriminator': disc.state_dict(),
                'epoch': epoch + 1,
            }, f"{args.output_dir}/gan_ep{epoch+1}.pt")

    # Final save
    torch.save(model.state_dict(), f"{args.output_dir}/final_model.pt")
    print(f"\nTraining complete. Final model saved to {args.output_dir}/final_model.pt")


if __name__ == '__main__':
    main()
