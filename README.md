# CORE — Canopy-Occluded Road Extraction

> **Deep learning pipeline for extracting roads from satellite imagery, with specific focus on roads hidden under tree canopy.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)](notebooks/CORE_Road_Extraction.ipynb)

---

## The Problem

Standard segmentation models fail on tree-covered roads because canopy pixels visually dominate the road surface. This causes two specific failure modes:

| Failure Mode | Description | Fix Applied |
|---|---|---|
| **Missed roads** | Model predicts nothing in heavily canopied tiles | Focal-Tversky loss (β=0.7) — punishes missed roads harder |
| **Broken roads** | Road detected but fragmented into disconnected pieces | Connectivity loss + skeleton-based gap bridging |

---

## Architecture — D-LinkNet

```
Input (512×512 RGB Satellite Image)
        │
   ┌────▼──────────────────────────────────────┐
   │   ResNet50 Encoder (ImageNet pretrained)   │
   │   Layer1→256ch  Layer2→512ch               │
   │   Layer3→1024ch Layer4→2048ch              │
   └────┬──────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │   Dilated Center Block  ← KEY INNOVATION   │
   │   Dilation rates: 1, 2, 4, 8              │
   │   Receptive field: 50+ pixels             │
   │   "Sees around" tree canopy gaps          │
   └────┬──────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │   LinkNet Decoder + Skip Connections       │
   │   Restores fine road edges                │
   └────┬──────────────────────────────────────┘
        │
   Output (512×512 Binary Road Mask)
```

Based on [D-LinkNet (Zhou et al., CVPR Workshops 2018)](https://arxiv.org/abs/1807.02736) — winner of the DeepGlobe Road Extraction Challenge.

---

## Results

| Metric | Value |
|---|---|
| Best Validation IoU | **49.70%** |
| Best Validation Recall | **80.23%** |
| Final Validation Loss | 0.2187 (↓68% from Epoch 1) |
| Best Checkpoint Epoch | 59 / 60 |
| Training Platform | Google Colab T4 GPU (~6 sec/epoch) |

**Benchmark context:** The published D-LinkNet paper achieved 64.66% IoU on the large-scale DeepGlobe dataset. This project trains on a custom dataset (Delhi satellite + Kaggle) in a resource-constrained setting.

---

## Quick Start

### Option 1 — Google Colab (Recommended)

Open the notebook and run top to bottom — no setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/CORE_Road_Extraction.ipynb)

The notebook covers everything: dataset upload → preprocessing → training → evaluation → prediction.

### Option 2 — Local Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/CORE-RoadExtraction.git
cd CORE-RoadExtraction

# 2. Install (CPU-only — no GPU needed for inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Prepare your dataset
python scripts/prepare_data.py \
    --raw_dir /path/to/your/dataset \
    --out_dir data/processed \
    --format flat   # or "deepglobe" for DeepGlobe format

# 4. Train
python scripts/train.py --config configs/default.yaml

# 5. Predict on new images
python scripts/predict.py \
    --checkpoint outputs/best_model.pth \
    --input /path/to/satellite_image.png \
    --output outputs/predictions/road_mask.png \
    --tta --tile --clean
```

---

## Dataset Format

```
your_dataset/
├── images/
│   ├── tile_001.png    ← RGB satellite image
│   ├── tile_002.png
│   └── ...
└── masks/
    ├── tile_001.png    ← Binary mask: 255=road, 0=background
    ├── tile_002.png    ← Filename must match the image exactly
    └── ...
```

Supports: `.png`, `.jpg`, `.tif`, `.tiff`

---

## Key Contributions

### 1. Loss Function
```
Total Loss = 0.25×BCE + 0.55×FocalTversky + 0.20×Focal + 0.15×Connectivity
```

- **Focal-Tversky** (β=0.7 > α=0.3): punishes missed roads more than over-predicted roads — directly raises recall on occluded segments
- **Connectivity Loss**: compares Sobel edge maps of predicted vs. ground-truth masks — penalises broken/gapped predictions topologically

### 2. Custom Augmentations (`utils/dataset.py`)

| Augmentation | What it does | Why |
|---|---|---|
| `ShadowSimulation` | Random dark polygon overlays on images | Teaches: dark ≠ no road |
| `VegetationColorShift` | Hue shifted toward green | Teaches: green tint ≠ no road |
| `RoadGapOcclusion` ⭐ | Punches canopy-like patches *only along road pixels*, leaves mask intact | Explicitly trains: road continues under canopy |

### 3. Inference Post-Processing (`scripts/predict.py`)

- **Test-Time Augmentation (TTA)** — 6 orientations averaged for robust predictions
- **Tiled inference** — handles images of any size with Hanning blend
- **Skeleton-based gap bridging** ⭐ — skeletonizes broken road segments, finds dead-end endpoints, reconnects co-linear pairs within configurable distance

---

## Project Structure

```
CORE-RoadExtraction/
├── models/
│   ├── d_linknet.py          # D-LinkNet architecture
│   └── losses.py             # BCE + Focal-Tversky + Focal + Connectivity
├── utils/
│   ├── dataset.py            # RoadDataset + custom augmentations
│   └── metrics.py            # IoU, F1, Precision, Recall, Connectivity Score
├── scripts/
│   ├── train.py              # Training loop (AMP, two-group LR, checkpointing)
│   ├── predict.py            # Inference (TTA, tiling, gap bridging)
│   └── prepare_data.py       # Dataset chipping and splitting
├── configs/
│   └── default.yaml          # All hyperparameters
├── notebooks/
│   └── CORE_Road_Extraction.ipynb   # Full Colab notebook
├── data/
│   ├── raw/                  # Put your original dataset here
│   └── processed/            # Auto-generated train/val splits
├── outputs/
│   └── predictions/          # Inference outputs
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `configs/default.yaml` to tune training. Key parameters for tree-occluded datasets:

```yaml
loss:
  tversky_beta:   0.7    # raise to 0.8 if roads are still being missed
  connectivity_w: 0.15   # raise to 0.25 if roads are still breaking up

training:
  epochs:      60        # raise if val IoU still climbing at end
  batch_size:  8         # reduce to 4 if GPU OOM
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Roads missed entirely | Class imbalance overwhelming the loss | Raise `tversky_beta` → 0.8 |
| Roads still broken after prediction | Gap too large for default bridge distance | Pass `--bridge_dist 60` to `predict.py` |
| GPU out of memory | Batch too large | Reduce `batch_size` to 4, `image_size` to 384 |
| `No matching mask found` | Image/mask filenames don't share the same stem | Rename files to match exactly |

---





