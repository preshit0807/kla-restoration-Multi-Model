# KLA Image Restoration — SEMICON India Hackathon 2026

Joint denoising (speckle + Gaussian noise) and 2x super-resolution for
grayscale semiconductor inspection imagery, built for the KLA track of the
SEMICON India Hackathon 2026.

## Problem

Input images suffer three simultaneous degradations: speckle (multiplicative)
noise, Gaussian (additive) noise, and 2x spatial downsampling. The task is to
recover a clean, full-resolution image from the degraded low-resolution input.

## Results (full 3,200-image validation set)

| Model | Params | PSNR | SSIM | Speed | Notes |
|---|---:|---:|---:|---:|---|
| CNN | 1.07M | 28.035 dB | 0.7641 | 5.16 ms/img (193.7 img/s) | Fastest |
| CNN + Transformer Hybrid | 2.76M | 27.837 dB | 0.7627 | 8.59 ms/img (116.5 img/s) | Ablation — see notes below |
| NAFNet-style | 3.16M | 28.082 dB | 0.7676 | 12.86 ms/img (77.8 img/s) | Best single-pass accuracy |
| NAFNet + TTA | 3.16M | 28.229 dB | 0.7745 | 41.22 ms/img (24.3 img/s) | Best accuracy, high latency cost |
| SemiconRestore-DA | 3.76M | *fill in after training* | | | Degradation-aware, newest |

All numbers computed on the same held-out validation split with
`evaluate.py`, per-image PSNR/SSIM averaged.

## Approach

**1. Data-driven design.** `scan_dataset.py` audits all 3,200 pairs before
any modeling: confirms consistent 2x downsampling, and finds 97.5% of noisy
samples exceed the [0,1] ground-truth range (up to 2.16x) — evidence of
multiplicative speckle noise rather than simple additive noise. This
motivated *not clipping* the model input and using a bounded (clamped)
residual output instead of assuming a fixed input range.

**2. Four architectures, evaluated empirically, not by reputation.**

- **CNN** (`model.py`) — fully convolutional residual-block network with
  pixel-shuffle upsampling. Lightweight, fast baseline.
- **CNN + Transformer Hybrid** (`model_hybrid.py`) — CNN encoder/decoder
  with a self-attention bottleneck for global context. *Ablation finding*:
  improved perceptual texture quality on fine detail (foliage, fur) but
  scored *lower* on PSNR/SSIM at full-dataset scale and was 1.66x slower —
  a case of the perception-distortion tradeoff. Kept as a documented
  negative result, not the final submission.
- **NAFNet-style** (`model_nafnet.py`) — based on "Simple Baselines for
  Image Restoration" (Chen et al., ECCV 2022): activation-free blocks
  (SimpleGate), Simplified Channel Attention, U-Net encoder-decoder. Adds
  **residual learning**: predicts a correction over a naive bicubic upsample
  rather than the full image, with a zero-initialized final layer so
  training starts from a stable, known baseline. Best single-pass model.
- **SemiconRestore-DA** (`model_semiconrestore.py`) — extends the NAFNet
  backbone with (a) a 4-channel informational input stem (raw intensity,
  asinh-compressed intensity, morphological gradient, local variance — all
  deterministic, non-destructive, near-zero cost) and (b) a lightweight
  degradation encoder + FiLM conditioning per block, so the network adapts
  its processing based on a learned, continuous "how is this image
  degraded" signal. Inspired by all-in-one restoration literature (AirNet,
  PromptIR) but much lighter.

**3. Loss design**, tuned iteratively based on visually diagnosing failure
modes rather than guessing: weighted L1 + SSIM + Sobel edge loss + VGG16
perceptual loss. Early experiments showed pixel/structural losses alone
over-smooth fine repeating texture (foliage, clouds, fabric); this was
traced to the loss function, not model capacity, and fixed by adding
perceptual loss and reweighting edge loss upward.

**4. Validation methodology fix.** Validation always uses full, uncropped,
deterministic images (`dataset.py`'s `split_dataset`), not a random crop
that changes every epoch — ensuring checkpoint selection ("best.pt") is
based on a reliable, consistent signal rather than noisy per-epoch crops.

## Repo structure

```
dataset.py                        # paired GT/NoisyLR loader + augmentation
losses.py                         # combined L1 + SSIM + edge + perceptual loss

model.py                 + train.py                    + checkpoints/
model_hybrid.py           + train_hybrid.py              + checkpoints_hybrid/
model_nafnet.py            + train_nafnet.py               + checkpoints_nafnet/
model_semiconrestore.py     + train_semiconrestore.py        + checkpoints_semiconrestore/

inference.py               # standalone inference (no manual edits; --model_type cnn|hybrid|nafnet|semiconrestore; optional --tta)
evaluate.py                 # PSNR / SSIM / LPIPS against ground truth
visualize_predictions.py     # visual check for one model
compare_models.py             # visual check across all 4 models at once
scan_dataset.py                 # dataset statistics / sanity-check scan
test_generalization.py           # verifies model handles unseen input resolution (256->512)
baseline_psnr.py                  # naive bicubic-upsample baseline, for context
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running inference

```bash
python inference.py \
    --input_dir path/to/NoisyLR \
    --output_dir path/to/results \
    --checkpoint checkpoints_nafnet/best.pt \
    --model_type nafnet \
    --device cuda
```

`--model_type` accepts `cnn`, `hybrid`, `nafnet`, or `semiconrestore`.
Add `--tta` for test-time augmentation (higher accuracy, ~3x slower).
Prints total and per-image inference time.

## Evaluating results

```bash
python evaluate.py --pred_dir path/to/results --gt_dir path/to/GT
```

Reports mean/min/max PSNR, SSIM, and LPIPS (if `lpips` is installed).

## Training from scratch

Edit `GT_DIR` / `NOISY_DIR` at the top of the relevant `train_*.py` script,
then run it directly, e.g.:

```bash
python train_nafnet.py
```

## Key design decisions

- **No input clipping**: speckle-noise overflow in the raw input (values up
  to 2.16, some below 0) is preserved, since it carries real information
  the model needs to learn to correct.
- **Residual learning with zero-init** (NAFNet, SemiconRestore-DA): stable,
  known starting point for training rather than learning from scratch.
- **Fully convolutional**: all four models generalize to input resolutions
  beyond the 128x128 they were trained on, verified in `test_generalization.py`
  for the 256→512 case mentioned in the problem statement.
- **Joint, single-pass denoise + SR**: avoids compounding errors from a
  sequential pipeline, and doesn't assume a fixed degradation order.

## References

- Chen, L. et al. "Simple Baselines for Image Restoration." ECCV 2022.
  (NAFNet backbone used in `model_nafnet.py` and `model_semiconrestore.py`)
- Perez, E. et al. "FiLM: Visual Reasoning with a General Conditioning
  Layer." AAAI 2018. (Conditioning mechanism used in `model_semiconrestore.py`)
- Li, B. et al. "All-in-One Image Restoration for Unknown Corruption."
  CVPR 2022. (AirNet — degradation-encoder concept)
- Potlapalli, V. et al. "PromptIR: Prompting for All-in-One Image
  Restoration." NeurIPS 2023. (Degradation-conditioned restoration concept)
- Blau, Y. & Michaeli, T. "The Perception-Distortion Tradeoff." CVPR 2018.
  (Explains the Hybrid model's PSNR/SSIM vs. perceptual-quality tradeoff)
- Zhang, R. et al. "The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric." CVPR 2018. (LPIPS, used in `evaluate.py`)
