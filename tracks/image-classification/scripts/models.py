"""Minimal models for research learning track (P2–P5). CPU-friendly defaults."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed_dim = 128
        self.head = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """Thin residual net (not full ResNet-18) — residual learning without CPU timeout."""

    def __init__(self, num_classes: int = 10, base: int = 32, layers=(2, 2, 2)):
        super().__init__()
        self.in_planes = base
        self.conv1 = nn.Conv2d(3, base, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(base, layers[0], stride=1)
        self.layer2 = self._make_layer(base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base * 4, layers[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 4, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, expand * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expand * dim, dim, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        return shortcut + x


class MiniConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 10, dims=(32, 64, 128), depths=(2, 2, 2)):
        super().__init__()
        layers = [nn.Conv2d(3, dims[0], 3, padding=1), nn.GroupNorm(1, dims[0]), nn.GELU()]
        in_dim = dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            if i > 0:
                layers += [nn.Conv2d(in_dim, dim, 2, stride=2), nn.GroupNorm(1, dim)]
                in_dim = dim
            for _ in range(depth):
                layers.append(ConvNeXtBlock(dim))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MiniViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 96,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch(x)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class EncoderForSSL(nn.Module):
    def __init__(self, backbone: Optional[nn.Module] = None, proj_dim: int = 64):
        super().__init__()
        self.backbone = backbone or MiniCNN(num_classes=10)
        in_dim = getattr(self.backbone, "embed_dim", 128)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.forward_features(x)), dim=-1)
