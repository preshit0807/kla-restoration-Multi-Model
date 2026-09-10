"""
Training script for the NAFNet-style model (current best single-pass
model). Usage: python train_nafnet.py
"""

import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import split_dataset
from model_nafnet import NAFRestorationNet
from losses import CombinedLoss

GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"

CROP_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 150
LR = 2e-4
VAL_SPLIT = 0.1
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints_nafnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CH = 32
ENC_BLOCKS = (2, 2, 4)
MIDDLE_BLOCKS = 4
DEC_BLOCKS = (2, 2, 2)


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(99.0)
    return 10 * torch.log10(1.0 / mse)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_ds, val_ds = split_dataset(GT_DIR, NOISY_DIR, crop_size=CROP_SIZE,
                                      val_split=VAL_SPLIT)
    train_size, val_size = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    print(f"Train samples: {train_size}, Val samples: {val_size} (full-image validation)")

    model = NAFRestorationNet(
        base_ch=BASE_CH, enc_blocks=ENC_BLOCKS, middle_blocks=MIDDLE_BLOCKS,
        dec_blocks=DEC_BLOCKS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    loss_fn = CombinedLoss().to(DEVICE)

    best_val_psnr = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0

        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(noisy)
            loss, parts = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * noisy.size(0)

        train_loss = train_loss_sum / train_size
        scheduler.step()

        model.eval()
        val_psnr_sum, val_loss_sum = 0.0, 0.0
        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
                pred = model(noisy)
                loss, _ = loss_fn(pred, gt)
                val_loss_sum += loss.item() * noisy.size(0)
                val_psnr_sum += psnr(pred, gt).item() * noisy.size(0)

        val_loss = val_loss_sum / max(val_size, 1)
        val_psnr = val_psnr_sum / max(val_size, 1)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f}dB (full-image) | {dt:.1f}s")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last.pt"))
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> new best model saved (val_psnr={val_psnr:.2f}dB)")

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f}dB")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
