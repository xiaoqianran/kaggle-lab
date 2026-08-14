"""Smoke tests: model forwards."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.models import MiniCNN, MiniConvNeXt, MiniViT, ResNetCIFAR  # noqa: E402


def test_minicnn_shape():
    assert MiniCNN(10)(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_resnet_shape():
    assert ResNetCIFAR(10)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_convnext_vit_shapes():
    x = torch.randn(2, 3, 32, 32)
    assert MiniConvNeXt(10)(x).shape == (2, 10)
    assert MiniViT(10)(x).shape == (2, 10)


def test_resnet_param_count_reasonable():
    n = sum(p.numel() for p in ResNetCIFAR(10).parameters())
    assert 5_000_000 > n > 10_000
