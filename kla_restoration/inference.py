"""
Standalone inference script. No manual edits required -- all paths are CLI
arguments.

Usage:
    python inference.py --input_dir path/to/NoisyLR --output_dir path/to/results \
        --checkpoint checkpoints_nafnet/best.pt --model_type nafnet --device cuda

Add --tta for test-time augmentation (original + h-flip + v-flip averaged),
which improves PSNR/SSIM at ~3x inference cost.
"""

import os
import argparse
import time
import numpy as np
import torch

from model import RestorationNet
from model_hybrid import HybridRestorationNet
from model_nafnet import NAFRestorationNet
from model_semiconrestore import SemiconRestoreDA


def build_model(model_type, device):
    if model_type == "cnn":
        model = RestorationNet(base_ch=64, num_blocks=12)
    elif model_type == "hybrid":
        model = HybridRestorationNet(
            base_ch=48, enc_blocks=4, downsample_factor=4,
            transformer_dim=192, transformer_layers=4, transformer_heads=4,
        )
    elif model_type == "nafnet":
        model = NAFRestorationNet(
            base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4, dec_blocks=(2, 2, 2),
        )
    elif model_type == "semiconrestore":
        model = SemiconRestoreDA(
            base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4, dec_blocks=(2, 2, 2),
            z_dim=64, asinh_scale=0.2,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type} "
                          f"(expected 'cnn', 'hybrid', 'nafnet', or 'semiconrestore')")
    return model.to(device)


def run_tta(model, noisy_t):
    """Average predictions across original + h-flip + v-flip inputs."""
    preds = []

    out = model(noisy_t)
    preds.append(out)

    flipped_h = torch.flip(noisy_t, dims=[-1])
    out_h = model(flipped_h)
    preds.append(torch.flip(out_h, dims=[-1]))

    flipped_v = torch.flip(noisy_t, dims=[-2])
    out_v = model(flipped_v)
    preds.append(torch.flip(out_v, dims=[-2]))

    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description="KLA restoration inference script")
    parser.add_argument("--input_dir", required=True,
                         help="Folder of noisy .npy input images")
    parser.add_argument("--output_dir", required=True,
                         help="Folder to save restored .npy outputs")
    parser.add_argument("--checkpoint", required=True,
                         help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model_type", choices=["cnn", "hybrid", "nafnet", "semiconrestore"],
                         default="semiconrestore",
                         help="Which architecture the checkpoint was trained with")
    parser.add_argument("--device", default=None,
                         help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--tta", action="store_true",
                         help="Test-time augmentation: average predictions across "
                              "original + horizontal-flip + vertical-flip inputs. "
                              "Improves PSNR/SSIM at the cost of ~3x inference time.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.model_type, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {args.checkpoint} ({n_params:,} params)")
    if args.tta:
        print("Test-time augmentation: ENABLED (original + h-flip + v-flip averaged)")

    input_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not input_files:
        print(f"No .npy files found in {args.input_dir}")
        return

    print(f"Running inference on {len(input_files)} images...")

    per_image_times = []

    if len(input_files) > 0:
        warm = np.load(os.path.join(args.input_dir, input_files[0])).astype(np.float32)
        warm_t = torch.from_numpy(warm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(warm_t)
        if device == "cuda":
            torch.cuda.synchronize()

    total_start = time.time()

    with torch.no_grad():
        for fname in input_files:
            noisy = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            pred = model(noisy_t) if not args.tta else run_tta(model, noisy_t)

            if device == "cuda":
                torch.cuda.synchronize()
            per_image_times.append(time.time() - t0)

            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
            np.save(os.path.join(args.output_dir, fname), pred_np)

    total_time = time.time() - total_start
    avg_time = sum(per_image_times) / len(per_image_times)

    print(f"\nDone. Restored {len(input_files)} images -> {args.output_dir}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average per-image time: {avg_time * 1000:.2f}ms")
    print(f"Throughput: {1.0 / avg_time:.2f} images/sec")


if __name__ == "__main__":
    main()
