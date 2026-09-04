"""
Loss functions for occluded road segmentation.

Why multiple losses?
  - BCE alone: treats all pixels equally → misses thin/hidden roads
  - Dice Loss: handles class imbalance (roads are a tiny % of pixels)
  - Focal Loss: down-weights easy background, forces model to focus on
                hard examples (roads under tree canopy)
  - Combined: best of all three for the tree-occlusion problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss — a generalisation of Dice that lets you independently
    weight False Negatives vs False Positives.

    THIS IS THE KEY FIX FOR "MISSED ROADS":
    Standard Dice penalises FN and FP equally. But a missed road pixel
    (FN) is a much worse outcome than a slightly-too-thick road
    prediction (FP). Setting beta > alpha makes the loss punish FN
    (missed roads) harder than FP, directly improving recall.

    Args:
        alpha : weight on False Positives (lower = more tolerant of FP)
        beta  : weight on False Negatives (higher = punishes missed roads more)

    Recommended starting point for "roads not being detected":
        alpha=0.3, beta=0.7   (recall-biased)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Tversky Loss + focal-style exponent — focuses training on the hardest
    (most ambiguous) road pixels, which is exactly where tree-occluded
    or low-contrast roads live. This combination is the single most
    effective fix for "model doesn't recognise the road at all".

    Args:
        gamma : >1 sharpens focus on hard examples (1.33 is a common default)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return tversky_loss ** self.gamma


class DiceLoss(nn.Module):
    """
    Soft Dice Loss — robust to class imbalance.
    Roads occupy ~5-15% of satellite imagery pixels.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss — penalises easy negatives less, hard positives more.
    Particularly useful for roads hidden under vegetation where the
    model is uncertain.

    Args:
        alpha : weight for positive class (roads)
        gamma : focusing parameter (2.0 is standard)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class ConnectivityAwareLoss(nn.Module):
    """
    Adds a gradient-based connectivity penalty.
    Penalises broken/disconnected road predictions — directly targets
    the "roads are discontinuous and broken" failure mode by comparing
    the *edge structure* of predicted vs. ground-truth masks. A road
    with gaps has extra edge segments (the start/end of each gap) that
    a continuous road doesn't have, so this loss pushes the predicted
    edge map to match the continuous ground-truth edge map.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _edge_map(self, mask):
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pred_edges   = self._edge_map(probs)
        target_edges = self._edge_map(targets.float())
        return F.mse_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """
    Weighted combination: BCE + Focal-Tversky + Connectivity

    THIS IS THE UPDATED RECIPE FOR BROKEN / MISSED ROADS.
    Compared to the original BCE+Dice+Focal recipe, this:
      1. Replaces plain Dice with Focal-Tversky (beta > alpha) → biases
         the model toward catching every road pixel (higher recall),
         which directly fixes "roads not being detected".
      2. Adds a connectivity penalty (on by default, weight 0.15) →
         directly penalises gaps/breaks in predicted road topology,
         which fixes "roads being discontinuous/broken".

    Tuning guide:
      - Roads still missed entirely      → raise tversky_beta toward 0.8,
                                            raise focal_alpha toward 0.85
      - Roads detected but broken/patchy → raise connectivity_w to 0.25–0.3
      - Too many false-positive roads    → raise tversky_alpha toward 0.4

    Args:
        use_connectivity : if True, adds the topology-aware penalty.
                            Costs a bit of extra compute per step but is
                            the most direct fix for broken/disconnected roads.
    """
    def __init__(
        self,
        bce_w:          float = 0.25,
        tversky_w:       float = 0.55,
        focal_w:        float = 0.20,
        connectivity_w: float = 0.15,
        tversky_alpha:  float = 0.3,    # FP weight (lower = tolerate thicker roads)
        tversky_beta:   float = 0.7,    # FN weight (higher = punish missed roads)
        tversky_gamma:  float = 1.33,
        focal_alpha:    float = 0.8,
        focal_gamma:    float = 2.0,
        use_connectivity: bool = True,
    ):
        super().__init__()
        self.bce_w          = bce_w
        self.tversky_w       = tversky_w
        self.focal_w        = focal_w
        self.connectivity_w = connectivity_w if use_connectivity else 0.0

        self.bce         = nn.BCEWithLogitsLoss()
        self.tversky      = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)
        self.focal        = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.connectivity = ConnectivityAwareLoss() if use_connectivity else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        l_bce     = self.bce(logits, targets)
        l_tversky = self.tversky(logits, targets)
        l_focal   = self.focal(logits, targets)

        total = (
            self.bce_w     * l_bce     +
            self.tversky_w * l_tversky +
            self.focal_w   * l_focal
        )

        parts = {"bce": l_bce.item(), "tversky": l_tversky.item(), "focal": l_focal.item()}

        if self.connectivity is not None:
            l_conn = self.connectivity(logits, targets)
            total = total + self.connectivity_w * l_conn
            parts["connectivity"] = l_conn.item()

        return total, parts
