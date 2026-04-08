"""
Dataset loader for NoLACE-mSBC training data.

Reads binary files produced by the prep_data C tool, or alternatively
performs on-the-fly mSBC degradation using a Python SBC wrapper.
"""

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path

FEATURE_DIM = 28
FRAME_SIZE = 160
FLOATS_PER_FRAME = FEATURE_DIM + 2 * FRAME_SIZE  # features + clean + degraded
BYTES_PER_FRAME = FLOATS_PER_FRAME * 4


class PreparedDataset(Dataset):
    """Loads pre-computed binary data from prep_data tool."""

    def __init__(self, data_dir, seq_len=100):
        """
        Args:
            data_dir: directory containing .bin files from prep_data
            seq_len: number of consecutive frames per training example
        """
        self.seq_len = seq_len
        self.files = sorted(Path(data_dir).glob('*.bin'))
        if not self.files:
            raise FileNotFoundError(f"No .bin files in {data_dir}")

        # Index all files: compute frame counts
        self.file_frames = []
        self.total_frames = 0
        for f in self.files:
            n_frames = f.stat().st_size // BYTES_PER_FRAME
            self.file_frames.append(n_frames)
            self.total_frames += max(0, n_frames - seq_len)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # Find which file and offset
        cumulative = 0
        for fi, n_frames in enumerate(self.file_frames):
            usable = max(0, n_frames - self.seq_len)
            if idx < cumulative + usable:
                frame_offset = idx - cumulative
                return self._load_sequence(self.files[fi], frame_offset)
            cumulative += usable
        raise IndexError(f"Index {idx} out of range")

    def _load_sequence(self, filepath, start_frame):
        with open(filepath, 'rb') as f:
            f.seek(start_frame * BYTES_PER_FRAME)
            data = f.read(self.seq_len * BYTES_PER_FRAME)

        floats = np.frombuffer(data, dtype=np.float32)
        floats = floats.reshape(self.seq_len, FLOATS_PER_FRAME)

        features = torch.from_numpy(floats[:, :FEATURE_DIM].copy())
        clean = torch.from_numpy(
            floats[:, FEATURE_DIM:FEATURE_DIM + FRAME_SIZE].copy()
        )
        degraded = torch.from_numpy(
            floats[:, FEATURE_DIM + FRAME_SIZE:].copy()
        )

        return {
            'features': features,           # (seq_len, FEATURE_DIM)
            'clean_frames': clean,           # (seq_len, FRAME_SIZE)
            'degraded_frames': degraded,     # (seq_len, FRAME_SIZE)
            'clean_signal': clean.reshape(-1),     # (seq_len * FRAME_SIZE,)
            'degraded_signal': degraded.reshape(-1),
        }


class OnTheFlyDataset(Dataset):
    """
    Loads clean audio and degrades through mSBC on the fly.

    Requires the `sbc` Python package or a subprocess call to the
    prep_data tool. This is a fallback for when pre-computed data
    is not available.
    """

    def __init__(self, audio_dir, seq_len=100, sample_rate=16000):
        self.seq_len = seq_len
        self.sr = sample_rate
        self.frame_size = FRAME_SIZE

        # Find audio files
        self.files = sorted(
            list(Path(audio_dir).glob('*.wav')) +
            list(Path(audio_dir).glob('*.raw'))
        )
        if not self.files:
            raise FileNotFoundError(f"No audio files in {audio_dir}")

        # Placeholder: actual implementation needs SBC Python bindings
        # or subprocess calls to the prep_data tool
        self._warn_printed = False

    def __len__(self):
        return len(self.files) * 10  # approximate

    def __getitem__(self, idx):
        if not self._warn_printed:
            print("WARNING: OnTheFlyDataset is a stub. "
                  "Use prep_data + PreparedDataset for actual training.")
            self._warn_printed = True
        # Return dummy data for API compatibility
        return {
            'features': torch.randn(self.seq_len, FEATURE_DIM),
            'clean_frames': torch.randn(self.seq_len, FRAME_SIZE),
            'degraded_frames': torch.randn(self.seq_len, FRAME_SIZE),
            'clean_signal': torch.randn(self.seq_len * FRAME_SIZE),
            'degraded_signal': torch.randn(self.seq_len * FRAME_SIZE),
        }


def create_dataloader(data_dir, batch_size=16, seq_len=100,
                      num_workers=4, prepared=True, pin_memory=None,
                      samples_per_epoch=None):
    """
    Build a DataLoader.

    samples_per_epoch: if set, each epoch draws this many random sequences
        (without replacement when possible) from the dataset instead of
        iterating over every sequence. A fresh random subset is drawn each
        epoch.
    """
    if prepared:
        ds = PreparedDataset(data_dir, seq_len)
    else:
        ds = OnTheFlyDataset(data_dir, seq_len)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if samples_per_epoch is not None:
        replacement = samples_per_epoch > len(ds)
        sampler = RandomSampler(ds, replacement=replacement,
                                num_samples=samples_per_epoch)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=num_workers > 0,
                          drop_last=True)

    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0,
                      drop_last=True)
