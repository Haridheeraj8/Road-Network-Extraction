"""
Metrics for road segmentation under tree occlusion.

Standard metrics:
  - IoU (Intersection over Union)  — primary segmentation metric
  - F1 / Dice Score                — balanced precision-recall
  - Precision & Recall             — recall especially important for hidden roads

Road-specific metrics:
  - Connectivity Score (APLS-inspired) — penalises broken / disconnected roads
    which are the main failure mode when tree canopy causes prediction gaps.
"""

import torch
import numpy as np
import cv2
from scipy import ndimage


# ─────────────────────────────────────────────
# Pixel-level metrics (fast, GPU-compatible)
# ─────────────────────────────────────────────

class RoadMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-8)

    @property
    def recall(self) -> float:
        """High recall = fewer missed roads under trees."""
        return self.tp / (self.tp + self.fn + 1e-8)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + 1e-8)

    @property
    def iou(self) -> float:
        return self.tp / (self.tp + self.fp + self.fn + 1e-8)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / (total + 1e-8)

    def compute(self) -> dict:
        return {
            "iou":       round(self.iou,       4),
            "f1":        round(self.f1,        4),
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "accuracy":  round(self.accuracy,  4),
        }


# ─────────────────────────────────────────────
# Connectivity / topology metrics (CPU, per-batch)
# ─────────────────────────────────────────────

def compute_connectivity_score(pred_mask: np.ndarray,
                                gt_mask: np.ndarray,
                                min_component: int = 50) -> float:
    """
    Connectivity Score: fraction of ground-truth connected road components
    that are also connected in the prediction.

    A score of 1.0 means every road segment is fully continuous.
    Score drops when tree occlusion causes 'holes' in predicted roads.

    Args:
        pred_mask      : binary numpy array, shape [H, W], values 0/1
        gt_mask        : binary numpy array, shape [H, W], values 0/1
        min_component  : ignore tiny road fragments (noise)
    """
    labeled_gt, n_components = ndimage.label(gt_mask)
    if n_components == 0:
        return 1.0   # no roads → nothing to evaluate

    connected = 0
    total = 0

    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled_gt == comp_id)
        if comp_mask.sum() < min_component:
            continue   # skip tiny fragments

        # Find the pred pixels that overlap this GT component
        pred_in_comp = pred_mask * comp_mask
        if pred_in_comp.sum() == 0:
            total += 1
            continue

        # Check if the pred pixels form a single connected component
        labeled_pred, n_pred_comp = ndimage.label(pred_in_comp)
        if n_pred_comp == 1:
            connected += 1
        total += 1

    return connected / (total + 1e-8)


def road_apls_iou(pred_mask: np.ndarray,
                   gt_mask: np.ndarray,
                   buffer_px: int = 5) -> dict:
    """
    Buffered IoU: dilates ground-truth roads by buffer_px before IoU.
    Accounts for annotation imprecision and slight spatial offsets
    caused by projection corrections in satellite imagery.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1)
    )
    gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel)

    intersection = np.logical_and(pred_mask, gt_dilated).sum()
    union        = np.logical_or(pred_mask,  gt_dilated).sum()
    buffered_iou = intersection / (union + 1e-8)

    connectivity = compute_connectivity_score(pred_mask, gt_mask)

    return {
        "buffered_iou":    round(float(buffered_iou),    4),
        "connectivity":    round(float(connectivity),    4),
        "road_quality":    round(float((buffered_iou + connectivity) / 2), 4),
    }


# ─────────────────────────────────────────────
# Threshold search utility
# ─────────────────────────────────────────────

def find_best_threshold(logits_list: list, masks_list: list,
                         thresholds=None) -> dict:
    """
    Sweeps probability thresholds and returns the one that maximises F1.
    Run once on validation set after training to tune the decision boundary.
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.75, 0.05)

    best_t, best_f1 = 0.5, 0.0
    results = []

    for t in thresholds:
        m = RoadMetrics(threshold=t)
        for logits, targets in zip(logits_list, masks_list):
            m.update(logits, targets)
        stats = m.compute()
        results.append({"threshold": round(t, 2), **stats})
        if stats["f1"] > best_f1:
            best_f1, best_t = stats["f1"], t

    return {"best_threshold": best_t, "best_f1": best_f1, "sweep": results}
