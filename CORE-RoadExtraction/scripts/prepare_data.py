"""
Data preparation utility.

Supported datasets:
  1. DeepGlobe Road Extraction (1024×1024 tiles, 6226 train / 1243 val)
     → Best public benchmark for road extraction including forested areas
     → Download: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

  2. SpaceNet 3 (Roads in 5 cities)
     → More annotation detail, includes canopy-covered areas
     → Download: s3://spacenet-dataset/spacenet/SN3_roads/

  3. Massachusetts Roads Dataset (1500×1500 aerial, rural + forested)
     → https://www.cs.toronto.edu/~vmnih/data/

This script:
  - Reads raw tiles and their corresponding binary masks
  - Chips large tiles into training-size patches (default 512×512)
  - Applies a road-density filter to discard blank chips
  - Creates train/val/test splits with stratification by road density
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil


# ─────────────────────────────────────────────
# Tiling
# ─────────────────────────────────────────────

def chip_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    chip_size: int = 512,
    stride: int = 256,           # 50% overlap → more training samples
    min_road_ratio: float = 0.01, # discard chips with <1% road pixels
):
    """
    Chips a large satellite tile into overlapping sub-images.
    Returns list of (image_chip, mask_chip) tuples passing road density filter.
    """
    H, W = image.shape[:2]
    chips = []

    for y in range(0, H - chip_size + 1, stride):
        for x in range(0, W - chip_size + 1, stride):
            img_chip  = image[y:y+chip_size, x:x+chip_size]
            mask_chip = mask[y:y+chip_size, x:x+chip_size]

            road_ratio = mask_chip.sum() / (chip_size * chip_size * 255.0)
            if road_ratio >= min_road_ratio:
                chips.append((img_chip, mask_chip))

    return chips


# ─────────────────────────────────────────────
# DeepGlobe layout parser
# ─────────────────────────────────────────────

def parse_deepglobe(raw_dir: Path):
    """
    DeepGlobe naming: {id}_sat.jpg  ↔  {id}_mask.png
    Returns list of (image_path, mask_path) pairs.
    """
    pairs = []
    for img_path in raw_dir.glob("*_sat.jpg"):
        mask_path = img_path.parent / img_path.name.replace("_sat.jpg", "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return sorted(pairs)


# ─────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────

def split_chips(chip_list, val_ratio=0.15, test_ratio=0.10, seed=42):
    random.seed(seed)
    random.shuffle(chip_list)
    n = len(chip_list)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (
        chip_list[n_test + n_val:],   # train
        chip_list[n_test:n_test + n_val],   # val
        chip_list[:n_test],            # test
    )


# ─────────────────────────────────────────────
# Save chips
# ─────────────────────────────────────────────

def save_chips(chips, out_img_dir: Path, out_mask_dir: Path, prefix: str):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in enumerate(chips):
        name = f"{prefix}_{i:05d}.png"
        cv2.imwrite(str(out_img_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_mask_dir / name), mask)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare road segmentation dataset")
    parser.add_argument("--raw_dir",    required=True,  help="Raw dataset root")
    parser.add_argument("--out_dir",    default="data/processed")
    parser.add_argument("--format",     default="deepglobe",
                        choices=["deepglobe", "spacenet", "flat"])
    parser.add_argument("--chip_size",  type=int, default=512)
    parser.add_argument("--stride",     type=int, default=256)
    parser.add_argument("--min_road",   type=float, default=0.01)
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Get image-mask pairs
    if args.format == "deepglobe":
        pairs = parse_deepglobe(raw_dir)
    else:
        # Generic flat layout: images/ and masks/ directories
        img_dir  = raw_dir / "images"
        mask_dir = raw_dir / "masks"
        pairs = sorted([
            (p, mask_dir / p.name)
            for p in img_dir.iterdir()
            if (mask_dir / p.name).exists()
        ])

    print(f"Found {len(pairs)} image-mask pairs")

    all_chips = []
    for img_path, mask_path in tqdm(pairs, desc="Chipping"):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"WARNING: Could not read {img_path.name}")
            continue

        chips = chip_image_and_mask(
            image, mask,
            chip_size=args.chip_size,
            stride=args.stride,
            min_road_ratio=args.min_road,
        )
        all_chips.extend(chips)

    print(f"Total chips (after road-density filter): {len(all_chips)}")

    train_chips, val_chips, test_chips = split_chips(
        all_chips, args.val_ratio, args.test_ratio
    )
    print(f"Train: {len(train_chips)} | Val: {len(val_chips)} | Test: {len(test_chips)}")

    save_chips(train_chips, out_dir/"train"/"images", out_dir/"train"/"masks", "train")
    save_chips(val_chips,   out_dir/"val"/"images",   out_dir/"val"/"masks",   "val")
    save_chips(test_chips,  out_dir/"test"/"images",  out_dir/"test"/"masks",  "test")

    print(f"\nDataset prepared at: {out_dir}")
    print("Next: update configs/default.yaml with data paths, then run train.py")


if __name__ == "__main__":
    main()
