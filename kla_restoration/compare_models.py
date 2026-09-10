"""
Loads all trained models and runs them on the same validation samples,
displaying: Noisy | CNN | Hybrid | NAFNet | SemiconRestore-DA | Ground Truth

Usage: python compare_models.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import RestorationNet
from model_hybrid import HybridRestorationNet
from model_nafnet import NAFRestorationNet
from model_semiconrestore import SemiconRestoreDA
from dataset import split_dataset

GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"

CNN_CHECKPOINT = r"checkpoints\best.pt"
HYBRID_CHECKPOINT = r"checkpoints_hybrid\best.pt"
NAFNET_CHECKPOINT = r"checkpoints_nafnet\best.pt"
SEMICONRESTORE_CHECKPOINT = r"checkpoints_semiconrestore\best.pt"

NUM_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42   # fixed seed -- same samples every run


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 99.0
    return (10 * torch.log10(1.0 / mse)).item()


def load_models():
    models = {}

    cnn = RestorationNet(base_ch=64, num_blocks=12).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_CHECKPOINT, map_location=DEVICE))
    cnn.eval()
    models["CNN"] = cnn

    hybrid = HybridRestorationNet(
        base_ch=48, enc_blocks=4, downsample_factor=4,
        transformer_dim=192, transformer_layers=4, transformer_heads=4,
    ).to(DEVICE)
    hybrid.load_state_dict(torch.load(HYBRID_CHECKPOINT, map_location=DEVICE))
    hybrid.eval()
    models["Hybrid"] = hybrid

    nafnet = NAFRestorationNet(
        base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4, dec_blocks=(2, 2, 2),
    ).to(DEVICE)
    nafnet.load_state_dict(torch.load(NAFNET_CHECKPOINT, map_location=DEVICE))
    nafnet.eval()
    models["NAFNet"] = nafnet

    semicon = SemiconRestoreDA(
        base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4, dec_blocks=(2, 2, 2),
        z_dim=64, asinh_scale=0.2,
    ).to(DEVICE)
    semicon.load_state_dict(torch.load(SEMICONRESTORE_CHECKPOINT, map_location=DEVICE))
    semicon.eval()
    models["SemiconRestore-DA"] = semicon

    return models


def main():
    print(f"Device: {DEVICE}")
    models = load_models()
    print(f"Loaded all models: {list(models.keys())}")

    _, val_ds = split_dataset(GT_DIR, NOISY_DIR, crop_size=None, val_split=0.1)
    print(f"Validation set size: {len(val_ds)}")

    n = min(NUM_SAMPLES, len(val_ds))
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(val_ds), n, replace=False)

    n_cols = 2 + len(models)
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_names = ["Noisy Input"] + list(models.keys()) + ["Ground Truth"]

    for row, idx in enumerate(indices):
        noisy, gt, fname = val_ds[idx]
        noisy_in = noisy.unsqueeze(0).to(DEVICE)
        gt_in = gt.unsqueeze(0).to(DEVICE)

        noisy_np = noisy.squeeze().numpy()
        gt_np = gt.squeeze().numpy()

        axes[row, 0].imshow(np.clip(noisy_np, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{col_names[0]}\n{fname}" if row == 0 else fname)
        axes[row, 0].axis("off")

        for col, (name, model) in enumerate(models.items(), start=1):
            with torch.no_grad():
                pred = model(noisy_in)
            p = psnr(pred, gt_in)
            pred_np = pred.squeeze().cpu().numpy()

            axes[row, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            title = f"{name}\nPSNR={p:.2f}dB"
            axes[row, col].set_title(title if row == 0 else f"PSNR={p:.2f}dB")
            axes[row, col].axis("off")

        last = n_cols - 1
        axes[row, last].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row, last].set_title(col_names[-1] if row == 0 else "")
        axes[row, last].axis("off")

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=100)
    print("\nSaved model_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
