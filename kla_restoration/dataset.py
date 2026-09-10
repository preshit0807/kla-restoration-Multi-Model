"""
Dataset loader for KLA image restoration hackathon.

Loads paired (NoisyLR, GT) .npy files. Handles the value-range overflow
found in the data scan (NoisyLR ranges roughly [-0.28, 2.16], GT is
strictly [0, 1]) -- inputs are NOT clipped, since that overflow carries
real information about the noise. Only random-crop + flip augmentation
is applied at train time.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, crop_size=None, augment=True, files=None):
        """
        gt_dir: folder of ground-truth .npy files (256x256, range [0,1])
        noisy_dir: folder of degraded .npy files (128x128, range ~[-0.3, 2.2])
        crop_size: if set, randomly crop the GT to this size (noisy is cropped
                   proportionally at half the size, since it's 2x downsampled).
                   Leave as None to use full images (256/128).
        augment: random horizontal/vertical flip + 90-degree rotation
        files: optional explicit list of filenames to use (for building
               independent train/val datasets that don't share augment state
               -- see `split_dataset` below).
        """
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.files = files if files is not None else sorted(
            f for f in os.listdir(gt_dir) if f.endswith(".npy")
        )
        self.crop_size = crop_size
        self.augment = augment

        # sanity check pairing once at init
        missing = [f for f in self.files if not os.path.exists(os.path.join(noisy_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} GT files have no matching NoisyLR file, e.g. {missing[:5]}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        noisy = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)

        if self.crop_size is not None:
            gt, noisy = self._paired_crop(gt, noisy, self.crop_size)

        if self.augment:
            gt, noisy = self._paired_augment(gt, noisy)

        # add channel dim: (H, W) -> (1, H, W)
        gt_t = torch.from_numpy(gt.copy()).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy.copy()).unsqueeze(0)
        return noisy_t, gt_t, fname

    @staticmethod
    def _paired_crop(gt, noisy, crop_size):
        """Crop GT to crop_size, noisy to crop_size // 2 (matches the 2x scale gap)."""
        gh, gw = gt.shape
        nh, nw = noisy.shape
        ncrop = crop_size // 2

        if gh < crop_size or gw < crop_size or nh < ncrop or nw < ncrop:
            return gt, noisy  # too small to crop, skip

        # pick crop location in noisy space, scale up to GT space
        ny = random.randint(0, nh - ncrop)
        nx = random.randint(0, nw - ncrop)
        gy, gx = ny * 2, nx * 2

        gt_c = gt[gy:gy + crop_size, gx:gx + crop_size]
        noisy_c = noisy[ny:ny + ncrop, nx:nx + ncrop]
        return gt_c, noisy_c

    @staticmethod
    def _paired_augment(gt, noisy):
        if random.random() < 0.5:
            gt = np.fliplr(gt)
            noisy = np.fliplr(noisy)
        if random.random() < 0.5:
            gt = np.flipud(gt)
            noisy = np.flipud(noisy)
        k = random.randint(0, 3)
        if k:
            gt = np.rot90(gt, k)
            noisy = np.rot90(noisy, k)
        return gt, noisy


def split_dataset(gt_dir, noisy_dir, crop_size=None, val_split=0.1, seed=42):
    """
    Builds independent train/val RestorationDataset instances (train augmented,
    val not), splitting on filenames so there's no shared mutable state between
    them -- avoids the classic bug where toggling .augment on one split silently
    affects the other (which happens with torch.utils.data.random_split, since
    it just wraps the same underlying Dataset object).

    IMPORTANT: validation always uses full, uncropped images (crop_size=None),
    regardless of what crop_size is passed for training. Using a random crop
    for validation means the crop changes every epoch, so two checkpoints
    aren't compared on the same pixels and "best.pt" selection becomes noisy.
    Validation must be deterministic and full-image to be a reliable signal
    for which checkpoint is actually best.
    """
    all_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n_val = int(len(all_files) * val_split)
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    train_ds = RestorationDataset(gt_dir, noisy_dir, crop_size=crop_size,
                                   augment=True, files=train_files)
    val_ds = RestorationDataset(gt_dir, noisy_dir, crop_size=None,
                                 augment=False, files=val_files)
    return train_ds, val_ds


if __name__ == "__main__":
    # quick smoke test -- edit paths and run directly to sanity check
    GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
    NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"

    ds = RestorationDataset(GT_DIR, NOISY_DIR, crop_size=128)
    print(f"Dataset size: {len(ds)}")
    noisy, gt, fname = ds[0]
    print(f"Sample: {fname}, noisy shape={tuple(noisy.shape)}, gt shape={tuple(gt.shape)}")
    print(f"noisy range=[{noisy.min():.3f}, {noisy.max():.3f}], gt range=[{gt.min():.3f}, {gt.max():.3f}]")
