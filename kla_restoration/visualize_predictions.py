"""
Visualize model predictions against noisy input and ground truth.
Usage: python visualize_predictions.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import RestorationNet
from dataset import split_dataset

GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"
CHECKPOINT = r"checkpoints\best.pt"
NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42   # fixed seed -- same samples every run, reproducible/comparable

BASE_CH = 64
NUM_BLOCKS = 12


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 99.0
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    model = RestorationNet(base_ch=BASE_CH, num_blocks=NUM_BLOCKS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    _, val_ds = split_dataset(GT_DIR, NOISY_DIR, crop_size=None, val_split=0.1)
    print(f"Validation set size: {len(val_ds)}")

    n = min(NUM_SAMPLES, len(val_ds))
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(val_ds), n, replace=False)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))

    for col, idx in enumerate(indices):
        noisy, gt, fname = val_ds[idx]
        noisy_in = noisy.unsqueeze(0).to(DEVICE)
        gt_in = gt.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(noisy_in)

        p = psnr(pred, gt_in)

        noisy_np = noisy.squeeze().numpy()
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().numpy()

        axes[0, col].imshow(np.clip(noisy_np, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Noisy Input\n{fname}\nshape={noisy_np.shape}")
        axes[0, col].axis("off")

        axes[1, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"Model Output\nPSNR={p:.2f}dB")
        axes[1, col].axis("off")

        axes[2, col].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(f"Ground Truth\nshape={gt_np.shape}")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.savefig("predictions_preview.png", dpi=100)
    print("Saved predictions_preview.png")
    plt.show()


if __name__ == "__main__":
    main()
