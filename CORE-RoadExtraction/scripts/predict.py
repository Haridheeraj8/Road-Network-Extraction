"""
Inference script — generates road masks from satellite images.

Supports:
  - Single image prediction
  - Batch folder prediction
  - Test-Time Augmentation (TTA) — averages flipped/rotated predictions
    to recover roads that are only visible from certain orientations
    (helps with tree-occluded roads where angle matters)
  - Large-image tiling with overlap-blend for images >512×512

Usage:
    python scripts/predict.py \
        --checkpoint outputs/best_model.pth \
        --input data/test_images/ \
        --output outputs/predictions/ \
        --tta \
        --tile
"""

import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.d_linknet import DLinkNet

# ImageNet normalisation constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────
# Pre/post processing
# ─────────────────────────────────────────────

def preprocess(image: np.ndarray, size: int = 512) -> torch.Tensor:
    """Resize → normalise → CHW tensor."""
    img = cv2.resize(image, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def postprocess(logit: torch.Tensor,
                orig_h: int,
                orig_w: int,
                threshold: float = 0.5) -> np.ndarray:
    """Sigmoid → threshold → resize to original dims → uint8 mask."""
    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────
# Test-Time Augmentation
# ─────────────────────────────────────────────

TTA_OPS = [
    (lambda x: x,                        lambda x: x),                # original
    (lambda x: torch.flip(x, dims=[3]),  lambda x: torch.flip(x, dims=[3])),  # H-flip
    (lambda x: torch.flip(x, dims=[2]),  lambda x: torch.flip(x, dims=[2])),  # V-flip
    (lambda x: torch.rot90(x, 1, [2,3]), lambda x: torch.rot90(x, -1, [2,3])), # +90°
    (lambda x: torch.rot90(x, 2, [2,3]), lambda x: torch.rot90(x,  2, [2,3])), # 180°
    (lambda x: torch.rot90(x, 3, [2,3]), lambda x: torch.rot90(x, -3, [2,3])), # +270°
]


@torch.no_grad()
def predict_tta(model: DLinkNet,
                tensor: torch.Tensor,
                device: torch.device) -> torch.Tensor:
    """
    Run 6 TTA variants and average their sigmoid probabilities.
    Averaging across orientations helps recover roads whose
    tree-occluded signature is orientation-dependent.
    """
    probs = []
    tensor = tensor.to(device)
    for aug, deaug in TTA_OPS:
        inp    = aug(tensor)
        logit  = model(inp)
        prob   = torch.sigmoid(deaug(logit))
        probs.append(prob)
    return torch.stack(probs).mean(dim=0)


# ─────────────────────────────────────────────
# Tiled prediction for large images
# ─────────────────────────────────────────────

def predict_tiled(model: DLinkNet,
                   image: np.ndarray,
                   device: torch.device,
                   tile_size: int = 512,
                   overlap: int = 64,
                   threshold: float = 0.5,
                   use_tta: bool = False) -> np.ndarray:
    """
    Splits large satellite images into overlapping tiles, predicts each,
    and blends results with a cosine weight window to avoid seam artifacts.
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Cosine blend window
    win_1d = np.hanning(tile_size).astype(np.float32)
    window = np.outer(win_1d, win_1d) + 1e-6

    ys = list(range(0, H - tile_size + 1, stride)) + [max(0, H - tile_size)]
    xs = list(range(0, W - tile_size + 1, stride)) + [max(0, W - tile_size)]

    for y in ys:
        for x in xs:
            tile = image[y:y+tile_size, x:x+tile_size]
            tensor = preprocess(tile, tile_size)

            if use_tta:
                prob = predict_tta(model, tensor, device).squeeze().cpu().numpy()
            else:
                logit = model(tensor.to(device))
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

            prob_map[y:y+tile_size, x:x+tile_size]    += prob * window
            weight_map[y:y+tile_size, x:x+tile_size]  += window

    blended = prob_map / weight_map
    return (blended > threshold).astype(np.uint8) * 255


# ─────────────────────────────────────────────
# Post-processing: clean up the mask
# ─────────────────────────────────────────────

def postprocess_mask(mask: np.ndarray,
                      min_area: int = 200,
                      close_kernel: int = 5,
                      bridge_gaps: bool = True,
                      max_bridge_dist: int = 40) -> np.ndarray:
    """
    Clean-up pipeline targeting broken/discontinuous road predictions:
      1. Light morphological closing — fills tiny 1-3px gaps
      2. Endpoint bridging (KEY FIX) — finds road segment endpoints that
         are near each other and roughly co-linear, and draws a connecting
         line between them. This directly repairs the "road disappears
         under a tree, reappears a few pixels later" failure pattern
         without blurring the road width everywhere (which plain closing
         with a large kernel does).
      3. Remove small disconnected blobs — eliminates leftover noise
         islands that aren't part of any real road.

    Args:
        min_area        : minimum pixel area to keep a component (removes noise)
        close_kernel     : small kernel for light gap-filling (3-7 typical)
        bridge_gaps      : enable endpoint-bridging step
        max_bridge_dist  : max pixel distance between two endpoints to
                            consider them part of the same broken road
    """
    # Step 1: light closing — only for very small gaps (don't oversmooth)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: bridge gaps between separate road components
    if bridge_gaps:
        closed = _bridge_road_gaps(closed, max_dist=max_bridge_dist)

    # Step 3: remove small disconnected blobs (noise, not bridged roads)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cleaned = np.zeros_like(closed)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def _get_skeleton_endpoints(component_mask: np.ndarray) -> list:
    """
    Skeletonizes a single road component and returns its endpoint
    pixels (skeleton points with exactly 1 neighbour = a dead end,
    i.e. where the road is cut off by occlusion).
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return []

    skeleton = skeletonize(component_mask > 0)
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return []

    coords = set(zip(xs.tolist(), ys.tolist()))
    endpoints = []
    for (x, y) in coords:
        neighbors = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in coords:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((x, y))
    return endpoints


def _bridge_road_gaps(mask: np.ndarray, max_dist: int = 40,
                      angle_tolerance_deg: float = 35.0) -> np.ndarray:
    """
    Finds endpoints of separate road components and connects pairs that
    are close together AND roughly co-linear with the component's local
    direction — i.e. likely the same road, broken by a tree, rather than
    two unrelated roads that happen to be near each other.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 2:   # background + at most 1 component, nothing to bridge
        return mask

    all_endpoints = []   # (x, y, component_id, direction_vector)
    for comp_id in range(1, n_labels):
        comp_mask = (labels == comp_id).astype(np.uint8) * 255
        if stats[comp_id, cv2.CC_STAT_AREA] < 10:
            continue
        endpoints = _get_skeleton_endpoints(comp_mask)
        for (x, y) in endpoints:
            # Estimate local direction via PCA on nearby skeleton pixels
            ys_c, xs_c = np.where(comp_mask > 0)
            dists = (xs_c - x) ** 2 + (ys_c - y) ** 2
            nearby = dists < (15 ** 2)
            if nearby.sum() >= 2:
                pts = np.stack([xs_c[nearby], ys_c[nearby]], axis=1).astype(np.float32)
                pts -= pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts, full_matrices=False)
                direction = vt[0]
            else:
                direction = np.array([1.0, 0.0])
            all_endpoints.append((x, y, comp_id, direction))

    bridged = mask.copy()
    used = set()
    for i, (x1, y1, c1, d1) in enumerate(all_endpoints):
        if i in used:
            continue
        best_j, best_dist = None, max_dist + 1
        for j, (x2, y2, c2, d2) in enumerate(all_endpoints):
            if j == i or j in used or c2 == c1:
                continue
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist > max_dist:
                continue
            # Check co-linearity: angle between connecting vector and
            # each endpoint's local road direction should be small
            connect_vec = np.array([x2 - x1, y2 - y1])
            connect_vec = connect_vec / (np.linalg.norm(connect_vec) + 1e-6)
            angle1 = np.degrees(np.arccos(np.clip(abs(np.dot(connect_vec, d1)), -1, 1)))
            angle2 = np.degrees(np.arccos(np.clip(abs(np.dot(connect_vec, d2)), -1, 1)))
            if angle1 > angle_tolerance_deg or angle2 > angle_tolerance_deg:
                continue
            if dist < best_dist:
                best_dist, best_j = dist, j

        if best_j is not None:
            x2, y2 = all_endpoints[best_j][0], all_endpoints[best_j][1]
            # Draw the bridge with the same approximate width as the road
            thickness = max(3, int(np.sqrt(stats[c1, cv2.CC_STAT_AREA]) / 8))
            cv2.line(bridged, (x1, y1), (x2, y2), 255, thickness=thickness)
            used.add(i)
            used.add(best_j)

    return bridged


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> DLinkNet:
    state = torch.load(checkpoint_path, map_location=device)
    model = DLinkNet(num_classes=1, pretrained=False)
    # Handle both raw state_dict and wrapped checkpoints
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_single(args, model, device):
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if args.tile:
        mask = predict_tiled(model, image, device,
                             tile_size=args.size, overlap=args.overlap,
                             threshold=args.threshold, use_tta=args.tta)
    else:
        tensor = preprocess(image, args.size)
        if args.tta:
            prob = predict_tta(model, tensor, device).squeeze().cpu().numpy()
            mask = (prob > args.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            logit = model(tensor.to(device))
            mask = postprocess(logit, h, w, args.threshold)

    if args.clean:
        mask = postprocess_mask(mask, bridge_gaps=not args.no_bridge,
                                max_bridge_dist=args.bridge_dist)

    out_path = args.output
    cv2.imwrite(str(out_path), mask)
    print(f"Saved: {out_path}")


def predict_folder(args, model, device):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = [p for p in Path(args.input).iterdir() if p.suffix.lower() in exts]
    print(f"Found {len(paths)} images")

    for img_path in tqdm(paths):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if args.tile:
            mask = predict_tiled(model, image, device,
                                 tile_size=args.size, overlap=args.overlap,
                                 threshold=args.threshold, use_tta=args.tta)
        else:
            tensor = preprocess(image, args.size)
            if args.tta:
                prob = predict_tta(model, tensor, device).squeeze().cpu().numpy()
                mask = (prob > args.threshold).astype(np.uint8) * 255
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                logit = model(tensor.to(device))
                mask = postprocess(logit, h, w, args.threshold)

        if args.clean:
            mask = postprocess_mask(mask, bridge_gaps=not args.no_bridge,
                                    max_bridge_dist=args.bridge_dist)

        cv2.imwrite(str(out_dir / (img_path.stem + "_mask.png")), mask)


def main():
    parser = argparse.ArgumentParser(description="D-LinkNet Road Mask Prediction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      required=True,  help="Image file or folder")
    parser.add_argument("--output",     required=True,  help="Output file or folder")
    parser.add_argument("--size",       type=int, default=512)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--tta",        action="store_true",
                        help="Enable Test-Time Augmentation")
    parser.add_argument("--tile",       action="store_true",
                        help="Use tiled inference for large images")
    parser.add_argument("--overlap",    type=int, default=64,
                        help="Overlap in pixels between tiles")
    parser.add_argument("--clean",      action="store_true",
                        help="Apply morphological post-processing")
    parser.add_argument("--bridge_dist", type=int, default=40,
                        help="Max pixel gap to bridge between broken road segments (used with --clean)")
    parser.add_argument("--no_bridge",  action="store_true",
                        help="Disable gap-bridging, keep only closing+denoise (used with --clean)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if os.path.isdir(args.input):
        predict_folder(args, model, device)
    else:
        predict_single(args, model, device)


if __name__ == "__main__":
    main()
