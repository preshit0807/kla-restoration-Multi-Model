"""
Computes PSNR/SSIM between the raw noisy input (bicubic-upsampled, no
model) and ground truth -- the "do nothing" baseline for context.

Usage: python baseline_psnr.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from losses import SSIMLoss

GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 99.0
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".npy"))
    ssim_fn = SSIMLoss()

    psnr_scores, ssim_scores = [], []

    print(f"Computing baseline (noisy input, bicubic-upsampled, no model) "
          f"over {len(gt_files)} images...")

    for i, fname in enumerate(gt_files):
        gt = np.load(os.path.join(GT_DIR, fname)).astype(np.float32)
        noisy = np.load(os.path.join(NOISY_DIR, fname)).astype(np.float32)

        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)

        upsampled = F.interpolate(noisy_t, size=gt_t.shape[-2:], mode="bicubic",
                                   align_corners=False)
        upsampled = torch.clamp(upsampled, 0, 1)

        p = psnr(upsampled, gt_t)
        s = 1 - ssim_fn(upsampled, gt_t).item()

        psnr_scores.append(p)
        ssim_scores.append(s)

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(gt_files)}")

    print(f"\n--- Baseline (bicubic upsample, no denoising) ---")
    print(f"Mean PSNR: {np.mean(psnr_scores):.3f} dB")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")


if __name__ == "__main__":
    main()
