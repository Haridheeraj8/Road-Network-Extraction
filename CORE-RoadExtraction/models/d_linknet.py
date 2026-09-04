"""
D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for Road Extraction
Paper: https://arxiv.org/abs/1807.02736

Key innovations for tree-occluded road detection:
  - ResNet50 pretrained encoder (ImageNet features capture texture/context)
  - Dilated convolution center block (large receptive field without losing resolution)
  - LinkNet-style decoder (lightweight, skip connections preserve spatial detail)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding if dilation == 1 else dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DilatedCenterBlock(nn.Module):
    """
    Center block with cascaded dilated convolutions.
    Dilation rates [1, 2, 4, 8] give receptive fields that
    'see through' tree canopy gaps to infer hidden road structure.
    """
    def __init__(self, channels=512):
        super().__init__()
        self.d1 = ConvBnRelu(channels, channels, dilation=1)
        self.d2 = ConvBnRelu(channels, channels, dilation=2)
        self.d4 = ConvBnRelu(channels, channels, dilation=4)
        self.d8 = ConvBnRelu(channels, channels, dilation=8)
        self.fuse = ConvBnRelu(channels * 4, channels, kernel=1, padding=0)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        out = torch.cat([d1, d2, d4, d8], dim=1)
        return self.fuse(out)


class DecoderBlock(nn.Module):
    """
    LinkNet decoder block:
    1×1 conv (reduce channels) → transposed conv (upsample 2×) → 1×1 conv (restore channels)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = in_ch // 4
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, kernel=1, padding=0),
            nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            ConvBnRelu(mid_ch, out_ch, kernel=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Main D-LinkNet model
# ─────────────────────────────────────────────

class DLinkNet(nn.Module):
    """
    D-LinkNet with ResNet50 backbone.

    Encoder channels (ResNet50 layer outputs):
        layer1 → 256, layer2 → 512, layer3 → 1024, layer4 → 2048

    The dilated center block operates at the bottleneck (2048 ch),
    giving the network a large effective receptive field to handle
    roads hidden under vegetation.

    Args:
        num_classes  : 1 for binary road/no-road mask
        pretrained   : use ImageNet weights for ResNet50 encoder
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        # ── Encoder (ResNet50 backbone) ──────────────────────────────
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.firstconv  = backbone.conv1       # 3 → 64, stride 2
        self.firstbn    = backbone.bn1
        self.firstrelu  = backbone.relu
        self.firstpool  = backbone.maxpool     # stride 2  →  ¼ size

        self.encoder1   = backbone.layer1      # 64  → 256,  same size
        self.encoder2   = backbone.layer2      # 256 → 512,  ½ size
        self.encoder3   = backbone.layer3      # 512 → 1024, ½ size
        self.encoder4   = backbone.layer4      # 1024→ 2048, ½ size

        # ── Dilated center block ─────────────────────────────────────
        self.center = DilatedCenterBlock(channels=2048)

        # ── Decoder (LinkNet style) ──────────────────────────────────
        self.decoder4   = DecoderBlock(2048, 1024)
        self.decoder3   = DecoderBlock(1024, 512)
        self.decoder2   = DecoderBlock(512,  256)
        self.decoder1   = DecoderBlock(256,  64)

        # ── Final up-sampling to input resolution ────────────────────
        self.finaldeconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.finalconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────────────
        x0 = self.firstrelu(self.firstbn(self.firstconv(x)))  # /2
        x0p = self.firstpool(x0)                              # /4

        e1 = self.encoder1(x0p)   # /4,  256 ch
        e2 = self.encoder2(e1)    # /8,  512 ch
        e3 = self.encoder3(e2)    # /16, 1024 ch
        e4 = self.encoder4(e3)    # /32, 2048 ch

        # ── Dilated center ───────────────────────────────────────────
        c = self.center(e4)       # /32, 2048 ch (large receptive field)

        # ── Decoder with skip connections ────────────────────────────
        d4 = self.decoder4(c)   + e3   # /16, 1024 ch
        d3 = self.decoder3(d4)  + e2   # /8,  512  ch
        d2 = self.decoder2(d3)  + e1   # /4,  256  ch
        d1 = self.decoder1(d2)  + x0   # /2,  64   ch

        # ── Final upsampling → original resolution ───────────────────
        out = self.finaldeconv(d1)     # /1,  32   ch
        out = self.finalconv(out)      # /1,  num_classes
        return out                     # raw logits (apply sigmoid externally)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = DLinkNet(num_classes=1, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # expect [2, 1, 512, 512]
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.1f} M")
