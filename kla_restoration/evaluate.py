"""
Evaluation script: compares restored outputs against ground truth,
reporting PSNR, SSIM, and LPIPS.

Usage:
    python evaluate.py --pred_dir path/to/restored --gt_dir path/to/GT
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from losses import SSIMLoss

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 99.0
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate restored images against ground truth")
    parser.add_argument("--pred_dir", required=True, help="Folder of restored .npy outputs")
    parser.add_argument("--gt_dir", required=True, help="Folder of ground-truth .npy images")
    parser.add_argument("--no_lpips", action="store_true",
                         help="Skip LPIPS even if the lpips package is installed")
    args = parser.parse_args()

    lpips_fn = None
    if LPIPS_AVAILABLE and not args.no_lpips:
        print("Loading LPIPS (downloads pretrained weights on first run)...")
        lpips_fn = lpips_lib.LPIPS(net="alex")
        lpips_fn.eval()
    elif not LPIPS_AVAILABLE:
        print("Note: 'lpips' package not installed -- skipping LPIPS. "
              "Install with: pip install lpips")

    pred_files = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".npy"))
    if not pred_files:
        print(f"No .npy files found in {args.pred_dir}")
        return

    ssim_fn = SSIMLoss()
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    missing = []

    for fname in pred_files:
        gt_path = os.path.join(args.gt_dir, fname)
        if not os.path.exists(gt_path):
            missing.append(fname)
            continue

        pred = np.load(os.path.join(args.pred_dir, fname)).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

        p = psnr(pred_t, gt_t)
        s = 1 - ssim_fn(pred_t, gt_t).item()

        psnr_scores.append(p)
        ssim_scores.append(s)

        if lpips_fn is not None:
            pred_rgb = pred_t.repeat(1, 3, 1, 1) * 2 - 1
            gt_rgb = gt_t.repeat(1, 3, 1, 1) * 2 - 1
            with torch.no_grad():
                l = lpips_fn(pred_rgb, gt_rgb).item()
            lpips_scores.append(l)

    if missing:
        print(f"Warning: {len(missing)} predictions have no matching GT file "
              f"(e.g. {missing[:5]})")

    if not psnr_scores:
        print("No matched pred/GT pairs found -- nothing to evaluate.")
        return

    print(f"\nEvaluated {len(psnr_scores)} images")
    print(f"Mean PSNR: {np.mean(psnr_scores):.3f} dB  (min={np.min(psnr_scores):.2f}, "
          f"max={np.max(psnr_scores):.2f})")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}  (min={np.min(ssim_scores):.4f}, "
          f"max={np.max(ssim_scores):.4f})")
    if lpips_scores:
        print(f"Mean LPIPS: {np.mean(lpips_scores):.4f}  (min={np.min(lpips_scores):.4f}, "
              f"max={np.max(lpips_scores):.4f})  (lower is better)")


if __name__ == "__main__":
    main()
