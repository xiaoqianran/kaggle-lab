"""Synthetic RGB-D for supervised / self-supervised experiments (no external download)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def make_scene(h: int = 128, w: int = 160, seed: int = 0, easy: bool = False) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if easy:
        # Color encodes depth: learnable mapping (for trainability under CPU budget)
        yy, xx = np.mgrid[0:h, 0:w]
        depth = 2.0 + 6.0 * (xx / max(w - 1, 1))  # left near-ish gradient + structure
        depth = depth.astype(np.float32)
        # RGB channels correlated with depth + noise
        d01 = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        rgb = np.stack([d01, 1.0 - d01, 0.5 * np.ones_like(d01)], axis=-1).astype(np.float32)
        rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape).astype(np.float32), 0, 1)
        # add a rectangular "object" with constant depth and unique color
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 3, 2 * w // 3
        depth[y0:y1, x0:x1] = 3.0
        rgb[y0:y1, x0:x1] = np.array([0.9, 0.2, 0.2], dtype=np.float32)
        return {"rgb": rgb, "depth": depth}

    depth = np.full((h, w), 8.0, dtype=np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    bg = rng.uniform(0.2, 0.8, size=3).astype(np.float32)
    rgb[:] = bg
    for z, color_scale in [(6.0, 0.9), (4.0, 0.7), (2.5, 0.5)]:
        y0 = int(rng.integers(0, h // 2))
        x0 = int(rng.integers(0, w // 2))
        y1 = int(rng.integers(y0 + 20, h))
        x1 = int(rng.integers(x0 + 20, w))
        depth[y0:y1, x0:x1] = z
        col = rng.uniform(0.1, 1.0, size=3).astype(np.float32) * color_scale
        rgb[y0:y1, x0:x1] = col
    rgb = np.clip(rgb + rng.normal(0, 0.01, rgb.shape).astype(np.float32), 0, 1)
    return {"rgb": rgb, "depth": depth}


def make_batch(
    n: int = 8,
    h: int = 128,
    w: int = 160,
    seed: int = 0,
    easy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgbs, depths, masks = [], [], []
    for i in range(n):
        s = make_scene(h, w, seed=seed + i, easy=easy)
        rgbs.append(s["rgb"].transpose(2, 0, 1))
        depths.append(s["depth"][None, ...])
        masks.append((s["depth"] > 0.1).astype(np.float32)[None, ...])
    return (
        torch.from_numpy(np.stack(rgbs)),
        torch.from_numpy(np.stack(depths)),
        torch.from_numpy(np.stack(masks)),
    )
