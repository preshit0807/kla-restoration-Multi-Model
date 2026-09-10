"""
Tests whether the trained model generalizes to an input resolution it never
saw during training (256x256 input -> 512x512 output).

CAVEAT: uses a bicubic-upsampled 128x128 sample as a proxy 256x256 input
(no real 256-input test data available) -- tests shape/structural
robustness, not accuracy at that scale.

Usage: python test_generalization.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import RestorationNet

NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"
SAMPLE_FILE = "000000.npy"
CHECKPOINT = r"checkpoints\best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model = RestorationNet(base_ch=64, num_blocks=12).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    noisy_128 = np.load(os.path.join(NOISY_DIR, SAMPLE_FILE)).astype(np.float32)
    print(f"Original NoisyLR shape: {noisy_128.shape}")

    x_128 = torch.from_numpy(noisy_128).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_256 = model(x_128)
    print(f"128x128 input -> {tuple(out_256.shape)} output (expected 256x256) "
          f"{'OK' if out_256.shape[-1] == 256 else 'MISMATCH'}")

    x_256 = F.interpolate(x_128, size=(256, 256), mode="bicubic", align_corners=False)
    with torch.no_grad():
        out_512 = model(x_256)
    print(f"256x256 input -> {tuple(out_512.shape)} output (expected 512x512) "
          f"{'OK' if out_512.shape[-1] == 512 else 'MISMATCH'}")
    print(f"Output range: [{out_512.min().item():.3f}, {out_512.max().item():.3f}] "
          f"{'OK (within [0,1])' if 0 <= out_512.min() and out_512.max() <= 1 else 'OUT OF RANGE'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(noisy_128, 0, 1), cmap="gray")
    axes[0].set_title(f"Original input\n{tuple(noisy_128.shape)}")
    axes[0].axis("off")

    axes[1].imshow(out_256.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Trained case: 128->256\n{tuple(out_256.shape[-2:])}")
    axes[1].axis("off")

    axes[2].imshow(out_512.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Unseen case: 256->512 (proxy input)\n{tuple(out_512.shape[-2:])}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("generalization_test.png", dpi=100)
    print("\nSaved generalization_test.png")
    plt.show()


if __name__ == "__main__":
    main()
