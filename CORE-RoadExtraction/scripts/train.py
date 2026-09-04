"""
Training script for D-LinkNet road extraction.

Usage:
    python scripts/train.py --config configs/default.yaml

Key training features:
  - Mixed-precision (fp16) for faster GPU training
  - CosineAnnealingWarmRestarts LR schedule
  - Early stopping on val IoU
  - Best model checkpointing
  - Gradient clipping (prevents loss spikes on hard occlusion batches)
  - Wandb logging (optional)
"""

import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.d_linknet import DLinkNet
from models.losses import CombinedLoss
from utils.dataset import get_dataloaders
from utils.metrics import RoadMetrics


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found — training will be slow")
    return device


# ─────────────────────────────────────────────
# Training / validation step
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    metrics = RoadMetrics()
    total_loss = 0.0

    for i, (images, masks, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss, loss_parts = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics.update(logits.detach(), masks.detach())

        if i % 20 == 0:
            stats = metrics.compute()
            print(f"  [Ep {epoch} | {i}/{len(loader)}] "
                  f"loss={loss.item():.4f}  "
                  f"bce={loss_parts['bce']:.3f}  "
                  f"tversky={loss_parts['tversky']:.3f}  "
                  f"focal={loss_parts['focal']:.3f}  "
                  f"conn={loss_parts.get('connectivity', 0):.3f}  "
                  f"IoU={stats['iou']:.4f}  "
                  f"Recall={stats['recall']:.4f}")

    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metrics = RoadMetrics()
    total_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast():
            logits = model(images)
            loss, _ = criterion(logits, masks)

        total_loss += loss.item()
        metrics.update(logits, masks)

    return total_loss / len(loader), metrics.compute()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(cfg: dict):
    device = setup_device()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        train_img_dir=cfg["data"]["train_images"],
        train_mask_dir=cfg["data"]["train_masks"],
        val_img_dir=cfg["data"]["val_images"],
        val_mask_dir=cfg["data"]["val_masks"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    # ── Model ───────────────────────────────────────────────────────
    model = DLinkNet(num_classes=1, pretrained=cfg["model"]["pretrained"])
    model = model.to(device)

    # Optional: resume from checkpoint
    start_epoch = 0
    best_iou = 0.0
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg["training"].get("resume") and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        print(f"Resumed from epoch {start_epoch} (best IoU={best_iou:.4f})")

    # ── Optimiser & schedule ─────────────────────────────────────────
    # Two param groups: lower LR for pretrained encoder
    encoder_params = list(model.firstconv.parameters()) + \
                     list(model.encoder1.parameters()) + \
                     list(model.encoder2.parameters()) + \
                     list(model.encoder3.parameters()) + \
                     list(model.encoder4.parameters())
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["training"]["encoder_lr"]},
        {"params": decoder_params, "lr": cfg["training"]["decoder_lr"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["training"]["epochs"] // 3,
        T_mult=1,
        eta_min=1e-6,
    )

    criterion = CombinedLoss(
        bce_w=cfg["loss"]["bce_w"],
        tversky_w=cfg["loss"]["tversky_w"],
        focal_w=cfg["loss"]["focal_w"],
        connectivity_w=cfg["loss"]["connectivity_w"],
        tversky_alpha=cfg["loss"]["tversky_alpha"],
        tversky_beta=cfg["loss"]["tversky_beta"],
        tversky_gamma=cfg["loss"]["tversky_gamma"],
        focal_alpha=cfg["loss"]["focal_alpha"],
        focal_gamma=cfg["loss"]["focal_gamma"],
        use_connectivity=cfg["loss"].get("use_connectivity", True),
    )

    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────────
    patience = cfg["training"].get("early_stopping_patience", 10)
    no_improve = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {cfg['training']['epochs']}")
        t0 = time.time()

        train_loss, train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch + 1
        )
        val_loss, val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1} Summary ({elapsed:.0f}s) | LR={current_lr:.2e}")
        print(f"  Train → loss={train_loss:.4f} | IoU={train_stats['iou']:.4f} | "
              f"F1={train_stats['f1']:.4f} | Recall={train_stats['recall']:.4f}")
        print(f"  Val   → loss={val_loss:.4f} | IoU={val_stats['iou']:.4f} | "
              f"F1={val_stats['f1']:.4f} | Recall={val_stats['recall']:.4f}")

        # Save best model
        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "val_stats": val_stats,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ New best IoU={best_iou:.4f} — checkpoint saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = ckpt_path.parent / f"epoch_{epoch+1}.pth"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_stats": val_stats}, periodic_path)

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    print(f"Best model saved to: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
