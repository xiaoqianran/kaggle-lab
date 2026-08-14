"""Minimal depth networks for supervised learning experiments."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Small U-Net for monocular depth.

    Output: max_depth * sigmoid(raw)  ∈ (0, max_depth) for stable metric learning.
    """

    def __init__(self, base: int = 32, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        b = base
        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)
        # bias toward mid-range depths
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.max_depth * torch.sigmoid(self.head(d1)) + 1e-3


def silog_torch(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, lamb: float = 0.85) -> torch.Tensor:
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    mask = mask.squeeze(1).bool()
    if mask.sum() < 1:
        return pred.sum() * 0.0
    p = pred[mask]
    g = gt[mask]
    d = torch.log(p) - torch.log(g)
    return torch.mean(d ** 2) - lamb * torch.mean(d) ** 2
