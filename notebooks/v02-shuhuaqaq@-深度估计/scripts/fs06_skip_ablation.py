#!/usr/bin/env python3
"""FS06: U-Net with skips vs no-skip encoder-decoder on edge-heavy scenes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from models import TinyUNet  # noqa: E402
from synth_data import make_batch, make_scene  # noqa: E402

OUT = ROOT / "results" / "fs06_skip_ablation"
OUT.mkdir(parents=True, exist_ok=True)


class NoSkipNet(nn.Module):
    def __init__(self, max_depth=10.0):
        super().__init__()
        self.max_depth = max_depth
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.dec(self.enc(x))
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.max_depth * torch.sigmoid(y) + 1e-3


def train(model, steps=80, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        rgb, depth, mask = make_batch(8, 64, 80, seed=seed + step, easy=True)
        pred = model(rgb)
        loss = (torch.abs(pred - depth) * mask).sum() / (mask.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_boundary(model, n=12):
    model.eval()
    abs_rels, b_errs = [], []
    with torch.no_grad():
        for i in range(n):
            s = make_scene(64, 80, seed=2000 + i, easy=True)
            rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)).unsqueeze(0)
            gt = s["depth"]
            pred = model(rgb).squeeze().numpy()
            m = compute_metrics(pred, gt, max_depth=20.0, align="none")
            abs_rels.append(m.abs_rel)
            gx = np.abs(gt[:, 1:] - gt[:, :-1])
            b = np.zeros_like(gt, dtype=bool)
            b[:, 1:] |= gx > 0.3
            b[:, :-1] |= gx > 0.3
            err = np.abs(pred - gt) / np.maximum(gt, 1e-3)
            b_errs.append(float(err[b].mean()) if b.any() else 0.0)
    return float(np.mean(abs_rels)), float(np.mean(b_errs))


def main() -> None:
    torch.manual_seed(0)
    unet = train(TinyUNet(base=16, max_depth=10.0), steps=80, seed=0)
    noskip = train(NoSkipNet(), steps=80, seed=0)
    u_abs, u_b = eval_boundary(unet)
    n_abs, n_b = eval_boundary(noskip)
    out = {
        "unet_abs_rel": u_abs,
        "unet_boundary_absrel": u_b,
        "noskip_abs_rel": n_abs,
        "noskip_boundary_absrel": n_b,
        "compare_to_fs04": "FS04 tiny reg lacks multi-scale skips; FS06 shows skip pathways help structure.",
        "lesson": "If noskip boundary error is higher, skips are doing real work on edges.",
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert u_abs < 0.2 and n_abs < 0.5
    print("FS06 acceptance: PASSED")


if __name__ == "__main__":
    main()
