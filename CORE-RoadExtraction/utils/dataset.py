"""
Dataset & augmentation pipeline for satellite road extraction.

Augmentation strategy for tree-occluded roads:
  1. Spectral jitter — satellite sensors vary; HSV shifts simulate sensor noise
  2. Shadow simulation — artificial shadows mimic tree canopy on roads
  3. CutOut / GridMask — randomly blanks image regions → forces model to
     infer roads from context (like it must under dense canopy)
  4. Geometric transforms — scale, flip, rotate handle varied image orientations
  5. Blur & noise — simulates atmospheric haze / lower-res sensors
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


# ─────────────────────────────────────────────
# Custom augmentations for canopy occlusion
# ─────────────────────────────────────────────

class ShadowSimulation(A.ImageOnlyTransform):
    """
    Randomly draws polygonal shadow patches on the image.
    Simulates tree canopy shadows falling on roads.
    p controls how often this is applied per image.
    """
    def __init__(self, shadow_intensity=(0.3, 0.7), num_shadows=(1, 4), p=0.5):
        super().__init__(p=p)
        self.shadow_intensity = shadow_intensity
        self.num_shadows = num_shadows

    def apply(self, img, **params):
        h, w = img.shape[:2]
        n = np.random.randint(*self.num_shadows)
        out = img.copy().astype(np.float32)
        for _ in range(n):
            intensity = np.random.uniform(*self.shadow_intensity)
            pts = np.random.randint([0, 0], [w, h], size=(np.random.randint(3, 7), 2))
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(mask, [pts], 1.0)
            out[..., :3] *= (1 - intensity * mask[..., None])
        return out.clip(0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("shadow_intensity", "num_shadows")


class VegetationColorShift(A.ImageOnlyTransform):
    """
    Shifts hue toward green to simulate dense canopy colour spill
    onto adjacent road pixels — a common cause of false negatives.
    """
    def __init__(self, hue_shift=(-15, 15), sat_shift=(-20, 20), p=0.4):
        super().__init__(p=p)
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift

    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[..., 0] = np.clip(hsv[..., 0] + np.random.randint(*self.hue_shift), 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(*self.sat_shift), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def get_transform_init_args_names(self):
        return ("hue_shift", "sat_shift")


class RoadGapOcclusion(A.DualTransform):
    """
    THE KEY FIX FOR "DISCONTINUOUS / BROKEN ROADS".

    Unlike generic CoarseDropout (which blanks random rectangles anywhere
    in the image), this transform finds actual road pixels in the mask
    and punches gaps *only along the road itself* — mimicking exactly
    what a tree canopy does: it interrupts a continuous road with patches
    of canopy, leaving visible road on both sides of the gap.

    Critically, this transform corrupts the IMAGE at the gap location
    (overlays a canopy-like green/dark patch) but leaves the MASK
    untouched — the ground truth still says "this is a road". This is
    what forces the model to learn "bridge the gap, the road continues
    underneath", rather than learning "blank patch = no road" (which is
    what plain CoarseDropout combined with an unmodified mask near a
    real canopy gap would otherwise teach implicitly).

    Args:
        num_gaps      : how many gaps to cut per image
        gap_length    : gap length in pixels along the road, sampled in this range
        gap_width     : how wide (perpendicular to road) the occlusion patch is
        p             : probability of applying this transform per image
    """
    def __init__(self, num_gaps=(1, 4), gap_length=(15, 45), gap_width=(10, 25), p=0.5):
        super().__init__(p=p)
        self.num_gaps = num_gaps
        self.gap_length = gap_length
        self.gap_width = gap_width

    def apply(self, img, gap_boxes=(), **params):
        out = img.copy()
        for (x, y, w, h, angle) in gap_boxes:
            out = self._draw_canopy_patch(out, x, y, w, h, angle)
        return out

    def apply_to_mask(self, mask, **params):
        # Mask is intentionally left untouched — the road is still
        # "there" under the canopy gap, ground truth doesn't change.
        return mask

    def _draw_canopy_patch(self, img, x, y, w, h, angle):
        """Overlays a rotated dark/green ellipse to mimic canopy shadow."""
        overlay = img.copy().astype(np.float32)
        patch_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(patch_mask, (x, y), (w // 2, h // 2), angle, 0, 360, 255, -1)

        # Canopy colour: dark desaturated green, with soft edge blending
        canopy_color = np.array([
            np.random.randint(20, 60),
            np.random.randint(45, 90),
            np.random.randint(20, 55),
        ], dtype=np.float32)

        blurred_mask = cv2.GaussianBlur(patch_mask, (9, 9), 0).astype(np.float32) / 255.0
        for c in range(3):
            overlay[..., c] = (
                overlay[..., c] * (1 - blurred_mask) + canopy_color[c] * blurred_mask
            )
        return overlay.clip(0, 255).astype(np.uint8)

    def get_params_dependent_on_data(self, params, data):
        mask = data.get("mask")
        h, w = mask.shape[:2] if mask is not None else (512, 512)
        gap_boxes = []

        if mask is not None and mask.sum() > 0:
            ys, xs = np.where(mask > 0)
            n_gaps = np.random.randint(*self.num_gaps)
            for _ in range(n_gaps):
                idx = np.random.randint(len(xs))
                cx, cy = int(xs[idx]), int(ys[idx])
                length = np.random.randint(*self.gap_length)
                width  = np.random.randint(*self.gap_width)
                angle  = np.random.randint(0, 180)
                gap_boxes.append((cx, cy, length, width, angle))

        return {"gap_boxes": gap_boxes}

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_transform_init_args_names(self):
        return ("num_gaps", "gap_length", "gap_width")


# ─────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = 512):
    return A.Compose([
        # Geometric
        A.RandomResizedCrop(height=image_size, width=image_size,
                            scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=45, p=0.5,
                           border_mode=cv2.BORDER_REFLECT),

        # Canopy-specific augmentations
        ShadowSimulation(shadow_intensity=(0.3, 0.7), num_shadows=(1, 5), p=0.6),
        VegetationColorShift(p=0.4),
        RoadGapOcclusion(num_gaps=(1, 4), gap_length=(15, 45), gap_width=(10, 25), p=0.6),

        # Spectral / sensor variance
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.4),

        # Blur & noise (simulate haze, compression, lower res sensors)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ISONoise(p=0.2),

        # Generic CutOut — light touch now; RoadGapOcclusion above
        # handles the road-targeted case, this just adds scene variety
        A.CoarseDropout(max_holes=4, max_height=image_size // 10,
                        max_width=image_size // 10,
                        min_holes=1, fill_value=0, p=0.25),

        # Elastic / grid distortion — handles warped satellite projections
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.2),

        # Normalise to ImageNet stats (ResNet50 pretrained)
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RoadDataset(Dataset):
    """
    Expects a flat directory layout:
        data/
          images/   *.tif / *.png / *.jpg   (RGB satellite chips)
          masks/    *.png                    (binary road mask, 0/255)

    Image and mask filenames must share the same stem (e.g. tile_001.tif
    paired with tile_001.png).

    Args:
        image_dir  : path to satellite image directory
        mask_dir   : path to mask directory
        transform  : albumentations Compose pipeline
        image_size : resize target (used only if no transform provided)
    """

    EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        image_size: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform or get_val_transforms(image_size)

        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_mask(self, img_path: Path) -> Path:
        """Find matching mask by stem, trying multiple extensions."""
        for ext in [".png", ".tif", ".tiff", ".jpg"]:
            candidate = self.mask_dir / (img_path.stem + ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No mask found for {img_path.name} in {self.mask_dir}"
        )

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self._find_mask(img_path)

        # Load image (force 3-channel RGB)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (binary: road=1, background=0)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)   # normalise to 0/1

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]              # float tensor [3, H, W]
        mask  = augmented["mask"].unsqueeze(0)  # float tensor [1, H, W]

        return image, mask.float(), str(img_path.name)


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
):
    train_ds = RoadDataset(train_img_dir, train_mask_dir,
                           transform=get_train_transforms(image_size))
    val_ds   = RoadDataset(val_img_dir,   val_mask_dir,
                           transform=get_val_transforms(image_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    return train_loader, val_loader
