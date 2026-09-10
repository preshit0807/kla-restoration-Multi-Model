"""
Full dataset scanner for GT vs NoisyLR .npy pairs.
Usage: python scan_dataset.py
"""

import os
import numpy as np

GT_DIR = r"C:\Users\presh\Downloads\train\train\GT"
NOISY_DIR = r"C:\Users\presh\Downloads\train\train\NoisyLR"


def main():
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".npy")])
    noisy_files = set(f for f in os.listdir(NOISY_DIR) if f.endswith(".npy"))

    total = len(gt_files)
    missing_pairs = []
    resolution_counts = {}
    downsample_ratios = {}
    gt_ranges = []
    noisy_ranges = []
    noisy_overflow_count = 0
    noisy_underflow_count = 0
    max_overflow = -np.inf
    min_underflow = np.inf

    print(f"Scanning {total} GT files...")

    for i, fname in enumerate(gt_files):
        if fname not in noisy_files:
            missing_pairs.append(fname)
            continue

        gt = np.load(os.path.join(GT_DIR, fname))
        noisy = np.load(os.path.join(NOISY_DIR, fname))

        key = (gt.shape, noisy.shape)
        resolution_counts[key] = resolution_counts.get(key, 0) + 1

        if noisy.shape[0] > 0:
            ratio = gt.shape[0] / noisy.shape[0]
            downsample_ratios[ratio] = downsample_ratios.get(ratio, 0) + 1

        gt_ranges.append((gt.min(), gt.max()))
        noisy_ranges.append((noisy.min(), noisy.max()))

        if noisy.max() > 1.0:
            noisy_overflow_count += 1
            max_overflow = max(max_overflow, noisy.max())
        if noisy.min() < 0.0:
            noisy_underflow_count += 1
            min_underflow = min(min_underflow, noisy.min())

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{total} scanned")

    lines = []
    lines.append("=" * 60)
    lines.append("DATASET SCAN REPORT")
    lines.append("=" * 60)
    lines.append(f"Total GT files:        {total}")
    lines.append(f"Matched pairs:         {total - len(missing_pairs)}")
    lines.append(f"Missing noisy pairs:   {len(missing_pairs)}")
    if missing_pairs[:10]:
        lines.append(f"  (first few missing: {missing_pairs[:10]})")
    lines.append("")

    lines.append("--- Resolution pairs (GT shape -> Noisy shape) ---")
    for key, count in sorted(resolution_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  GT{key[0]} -> Noisy{key[1]} : {count} samples")
    lines.append("")

    lines.append("--- Downsample ratios ---")
    for ratio, count in sorted(downsample_ratios.items()):
        lines.append(f"  {ratio}x downsample : {count} samples")
    lines.append("")

    if gt_ranges:
        gt_mins = [r[0] for r in gt_ranges]
        gt_maxs = [r[1] for r in gt_ranges]
        noisy_mins = [r[0] for r in noisy_ranges]
        noisy_maxs = [r[1] for r in noisy_ranges]

        lines.append("--- Value ranges ---")
        lines.append(f"  GT global min/max:      {min(gt_mins):.4f} / {max(gt_maxs):.4f}")
        lines.append(f"  NoisyLR global min/max: {min(noisy_mins):.4f} / {max(noisy_maxs):.4f}")
        lines.append(f"  NoisyLR samples exceeding max=1.0: {noisy_overflow_count} "
                      f"({100 * noisy_overflow_count / len(noisy_ranges):.1f}%), "
                      f"max overflow value seen: {max_overflow:.4f}")
        lines.append(f"  NoisyLR samples below min=0.0: {noisy_underflow_count} "
                      f"({100 * noisy_underflow_count / len(noisy_ranges):.1f}%), "
                      f"min underflow value seen: {min_underflow:.4f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    with open("dataset_report.txt", "w") as f:
        f.write(report)
    print("\nSaved full report to dataset_report.txt")


if __name__ == "__main__":
    main()
