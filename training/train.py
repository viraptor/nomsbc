"""
Training loop for NoLACE-mSBC.

LS-GAN training with multi-resolution STFT discriminator.
Follows a two-phase approach:
  Phase 1: Pre-train generator with STFT loss only (stabilizes early training)
  Phase 2: Full GAN training with discriminator
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', None)
        if desc:
            print(f"[{desc}] starting"
                  + (f" ({total} batches)" if total else ""), flush=True)
        for i, item in enumerate(iterable):
            if i % 10 == 0:
                print(f"[{desc}] batch {i}"
                      + (f"/{total}" if total else ""), flush=True)
            yield item

from model import NoLACEmSBC, unpack_params, apply_adaconv_diff
from discriminator import MultiResolutionSTFTDiscriminator
from losses import (ls_discriminator_loss, ls_generator_loss,
                    multi_resolution_stft_loss)
from dataset import create_dataloader


def build_enhanced_signal(model, batch, device):
    """
    Run the model and differentiable DDSP to produce enhanced signal.

    Vectorized over (B, T): all frames processed in one batched FIR
    instead of a Python loop, drastically reducing kernel-launch overhead
    on MPS/CUDA.

    The GRU/encoder always run in fp32 (MPS autocast does not cover GRU
    weights). Caller may wrap the DDSP path in autocast separately.
    """
    features = batch['features'].to(device, non_blocking=True)
    degraded_frames = batch['degraded_frames'].to(device, non_blocking=True)

    # Force fp32 model forward regardless of outer autocast context.
    with torch.autocast(device_type=device.type, enabled=False):
        params_flat, _ = model(features)  # (B, T, PARAM_DIM)
    B, T, _ = params_flat.shape

    # Flatten (B, T, ...) -> (B*T, ...) so DDSP runs once over all frames.
    params_bt = params_flat.reshape(B * T, -1)
    frames_bt = degraded_frames.reshape(B * T, -1)

    p = unpack_params(params_bt)
    enhanced_bt = apply_adaconv_diff(
        frames_bt, p['conv_kernel'], p['conv_gain']
    )

    enhanced_signal = enhanced_bt.reshape(B, T * enhanced_bt.shape[-1])
    return enhanced_signal


def _autocast_ctx(device, enabled):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type in ('cuda', 'mps', 'cpu'):
        dtype = torch.float16 if device.type != 'cpu' else torch.bfloat16
        return torch.autocast(device_type=device.type, dtype=dtype)
    return torch.autocast(device_type=device.type, enabled=False)


def train_epoch_pretrain(model, loader, optimizer, device, epoch_desc='',
                         use_amp=False):
    """Phase 1: generator pre-training with STFT loss only."""
    model.train()
    total_loss = 0
    n_batches = len(loader)
    pbar = tqdm(loader, total=n_batches, desc=epoch_desc or 'pretrain',
                dynamic_ncols=True)
    for i, batch in enumerate(pbar, start=1):
        clean = batch['clean_signal'].to(device, non_blocking=True)
        with _autocast_ctx(device, use_amp):
            enhanced = build_enhanced_signal(model, batch, device)
            loss = multi_resolution_stft_loss(enhanced, clean)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             avg=f"{total_loss / i:.4f}")
    return total_loss / n_batches


def train_epoch_gan(model, disc, loader, opt_g, opt_d, device,
                    lambda_stft=10.0, epoch_desc='', use_amp=False):
    """Phase 2: full LS-GAN training."""
    model.train()
    disc.train()
    total_g_loss = 0
    total_d_loss = 0
    n_batches = len(loader)
    pbar = tqdm(loader, total=n_batches, desc=epoch_desc or 'gan',
                dynamic_ncols=True)
    i = 0
    for batch in pbar:
        i += 1
        clean = batch['clean_signal'].to(device, non_blocking=True)
        with _autocast_ctx(device, use_amp):
            enhanced = build_enhanced_signal(model, batch, device)

            # --- Discriminator step ---
            real_out = disc(clean)
            fake_out = disc(enhanced.detach())
            d_loss = ls_discriminator_loss(real_out, fake_out)

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # --- Generator step ---
        with _autocast_ctx(device, use_amp):
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
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(g=f"{g_loss.item():.4f}",
                             d=f"{d_loss.item():.4f}",
                             g_avg=f"{total_g_loss / i:.4f}",
                             d_avg=f"{total_d_loss / i:.4f}")

    return total_g_loss / n_batches, total_d_loss / n_batches


def _run_profile(model, loader, optimizer, device, use_amp, n_iters=20):
    """Time the major stages over a few batches and print a summary."""
    import statistics
    model.train()
    it = iter(loader)
    sync = (torch.cuda.synchronize if device.type == 'cuda'
            else (torch.mps.synchronize if device.type == 'mps'
                  else lambda: None))

    stages = {'data': [], 'h2d': [], 'model_fwd': [], 'ddsp': [],
              'stft_loss': [], 'backward': [], 'step': [], 'total': []}

    # warmup
    for _ in range(2):
        batch = next(it)
        clean = batch['clean_signal'].to(device)
        with _autocast_ctx(device, use_amp):
            enhanced = build_enhanced_signal(model, batch, device)
            loss = multi_resolution_stft_loss(enhanced, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sync()

    for _ in range(n_iters):
        t_total = time.time()
        t = time.time()
        batch = next(it)
        clean = batch['clean_signal'].to(device, non_blocking=True)
        sync(); stages['data'].append(time.time() - t)

        t = time.time()
        features = batch['features'].to(device, non_blocking=True)
        degraded_frames = batch['degraded_frames'].to(device, non_blocking=True)
        sync(); stages['h2d'].append(time.time() - t)

        t = time.time()
        with torch.autocast(device_type=device.type, enabled=False):
            params_flat, _ = model(features)
        sync(); stages['model_fwd'].append(time.time() - t)

        t = time.time()
        B, T, _ = params_flat.shape
        with _autocast_ctx(device, use_amp):
            params_bt = params_flat.reshape(B * T, -1)
            frames_bt = degraded_frames.reshape(B * T, -1)
            p = unpack_params(params_bt)
            enhanced_bt = apply_adaconv_diff(
                frames_bt, p['conv_kernel'], p['conv_gain']
            )
            enhanced = enhanced_bt.reshape(B, -1)
        sync(); stages['ddsp'].append(time.time() - t)

        t = time.time()
        with _autocast_ctx(device, use_amp):
            loss = multi_resolution_stft_loss(enhanced, clean)
        sync(); stages['stft_loss'].append(time.time() - t)

        t = time.time()
        optimizer.zero_grad()
        loss.backward()
        sync(); stages['backward'].append(time.time() - t)

        t = time.time()
        optimizer.step()
        sync(); stages['step'].append(time.time() - t)

        stages['total'].append(time.time() - t_total)

    print("\n=== Profile (per-batch ms, mean over "
          f"{n_iters} iters) ===")
    for name, vals in stages.items():
        ms = statistics.mean(vals) * 1000
        print(f"  {name:14s} {ms:8.2f} ms")
    total_ms = statistics.mean(stages['total']) * 1000
    batch_size = loader.batch_size or 1
    it_per_s = 1000.0 / total_ms
    print(f"  ~{it_per_s:.2f} it/s  "
          f"({it_per_s * batch_size:.0f} samples/s "
          f"@ batch={batch_size})")


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
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision (autocast)')
    parser.add_argument('--profile', action='store_true',
                        help='Profile a few batches and exit')
    parser.add_argument('--gan-samples-per-epoch', type=int, default=1000,
                        help='Random sequences sampled per GAN epoch '
                             '(0 = use full dataset)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Loading dataset from {args.data_dir} ...", flush=True)
    loader = create_dataloader(args.data_dir, args.batch_size, args.seq_len,
                               num_workers=args.num_workers)
    ds = loader.dataset
    print(f"  files: {len(getattr(ds, 'files', []))}  "
          f"sequences: {len(ds)}  "
          f"batches/epoch: {len(loader)}  "
          f"batch_size: {args.batch_size}  seq_len: {args.seq_len}",
          flush=True)

    model = NoLACEmSBC().to(device)
    disc = MultiResolutionSTFTDiscriminator().to(device)

    print(f"Generator params: {model.count_params():,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}",
          flush=True)

    opt_g = optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    opt_d = optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99))

    if args.profile:
        _run_profile(model, loader, opt_g, device, args.amp)
        return

    # Phase 1: Pre-train with STFT loss
    print("\n--- Phase 1: Pre-training (STFT loss) ---", flush=True)
    for epoch in range(args.pretrain_epochs):
        t0 = time.time()
        desc = f"pretrain {epoch+1}/{args.pretrain_epochs}"
        loss = train_epoch_pretrain(model, loader, opt_g, device, desc,
                                    use_amp=args.amp)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.pretrain_epochs}  "
              f"STFT loss: {loss:.4f}  ({dt:.1f}s)", flush=True)

    pretrain_path = f"{args.output_dir}/pretrain_final.pt"
    torch.save(model.state_dict(), pretrain_path)
    print(f"Saved pretrained model to {pretrain_path}", flush=True)

    # Phase 2: GAN training
    # Use a separate loader that draws a random subset each epoch — keeps
    # epochs short and varied without iterating the full dataset.
    if args.gan_samples_per_epoch and args.gan_samples_per_epoch > 0:
        gan_loader = create_dataloader(
            args.data_dir, args.batch_size, args.seq_len,
            num_workers=args.num_workers,
            samples_per_epoch=args.gan_samples_per_epoch,
        )
        print(f"GAN loader: {args.gan_samples_per_epoch} random samples/epoch  "
              f"({len(gan_loader)} batches)", flush=True)
    else:
        gan_loader = loader

    print("\n--- Phase 2: GAN training ---", flush=True)
    for epoch in range(args.gan_epochs):
        t0 = time.time()
        desc = f"gan {epoch+1}/{args.gan_epochs}"
        g_loss, d_loss = train_epoch_gan(
            model, disc, gan_loader, opt_g, opt_d, device, epoch_desc=desc,
            use_amp=args.amp,
        )
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.gan_epochs}  "
              f"G loss: {g_loss:.4f}  D loss: {d_loss:.4f}  ({dt:.1f}s)",
              flush=True)
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
