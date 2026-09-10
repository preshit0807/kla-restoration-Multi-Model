# KLA Image Restoration — SEMICON India Hackathon 2026

Joint denoising (speckle + Gaussian) and 2x super-resolution for grayscale
images, built for the KLA track of the SEMICON India Hackathon 2026.

## Results (full 3,200-image validation set)

| Model | Params | PSNR | SSIM | Speed (ms/img) | Throughput |
|---|---|---|---|---|---|
| CNN | 1.07M | 28.035 dB | 0.7641 | 5.16 | 193.7 img/s |
| CNN + Transformer Hybrid | 2.76M | 27.837 dB | 0.7627 | 8.59 | 116.5 img/s |
| **NAFNet-style** | 3.16M | **28.082 dB** | **0.7676** | 12.86 | 77.8 img/s |
| NAFNet + TTA | 3.16M | 28.229 dB | 0.7745 | 41.22 | 24.3 img/s |
| SemiconRestore-DA | 3.76M | (see your training log) | | | |

## Approach summary

1. **Data analysis** (`scan_dataset.py`) confirmed 3,200 pairs, consistent
   2x downsampling, and that 97.5% of noisy samples exceed the [0,1]
   ground-truth range (up to 2.16x) — evidence of multiplicative speckle
   noise, not simple additive noise.
2. **Four architectures were built and compared**, in order of development:
   - `model.py` — CNN baseline (residual blocks + pixel-shuffle SR)
   - `model_hybrid.py` — CNN + Transformer bottleneck (ablation: underperformed
     on PSNR/SSIM despite better perceptual texture quality — a case of the
     perception-distortion tradeoff)
   - `model_nafnet.py` — NAFNet-style (Chen et al., ECCV 2022) with residual
     learning (predicts a correction over a bicubic upsample, zero-initialized
     tail for a stable training start) — best single-pass model
   - `model_semiconrestore.py` — SemiconRestore-DA: NAFNet backbone extended
     with (a) a 4-channel informational input stem (raw, asinh-compressed,
     morphological gradient, local variance) and (b) a lightweight
     degradation encoder + FiLM conditioning per block, inspired by
     all-in-one restoration literature (AirNet, PromptIR)
3. **Loss**: weighted L1 + SSIM + Sobel edge loss + VGG16 perceptual loss,
   tuned iteratively based on visual diagnosis of failure modes (fine
   texture over-smoothing was traced to the loss function, not model
   capacity, and fixed by adding perceptual loss and reweighting edge loss).

## Repo structure

```
dataset.py                    # paired GT/NoisyLR loader + augmentation
                               # (validation always uses full images, not
                               # random crops, for reliable checkpoint selection)
model.py                      # CNN baseline
model_hybrid.py                # CNN+Transformer hybrid (ablation)
model_nafnet.py                 # NAFNet-style (best single-pass model)
model_semiconrestore.py          # SemiconRestore-DA (degradation-aware)
losses.py                        # combined L1 + SSIM + edge + perceptual loss
train.py / train_hybrid.py / train_nafnet.py / train_semiconrestore.py
inference.py                      # standalone inference script (no manual edits, all 4 models, optional --tta)
evaluate.py                        # PSNR/SSIM/LPIPS evaluation against ground truth
visualize_predictions.py            # visual comparison for one model
compare_models.py                    # visual comparison across all 4 models at once
scan_dataset.py                       # dataset statistics / sanity-check scan
test_generalization.py                 # verifies model handles unseen input resolution
baseline_psnr.py                        # naive bicubic-upsample baseline for context
requirements.txt
checkpoints/, checkpoints_hybrid/, checkpoints_nafnet/, checkpoints_semiconrestore/
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
    --model_type nafnet
```

`--model_type` accepts `cnn`, `hybrid`, `nafnet`, or `semiconrestore`.
Add `--tta` for test-time augmentation (higher accuracy, ~3x slower).
Reports total and per-image inference time.

## Evaluating results

```bash
python evaluate.py --pred_dir path/to/results --gt_dir path/to/GT
```

Reports mean/min/max PSNR, SSIM, and LPIPS (if the `lpips` package is installed).

## Training from scratch

Edit the `GT_DIR` / `NOISY_DIR` paths at the top of the relevant `train_*.py`
script, then run it directly, e.g.:

```bash
python train_nafnet.py
```

## Design notes

- **No input clipping**: the noisy input is fed to every model as-is, since
  its range overflow (caused by speckle noise) carries real information the
  model needs to learn to correct.
- **Residual learning** (NAFNet, SemiconRestore-DA): the network predicts a
  correction over a naive bicubic upsample rather than reconstructing the
  entire image from scratch, with a zero-initialized final layer so training
  starts from a known, stable baseline.
- **Fully convolutional architectures**: no fixed-size layers, so models
  generalize to input resolutions beyond the 128x128 they were trained on
  (verified in `test_generalization.py`).
- **Joint (not sequential) denoise + SR**: a single pass handles both, since
  KLA's own material notes degradation order may vary, and sequential
  pipelines risk compounding early-stage errors.
