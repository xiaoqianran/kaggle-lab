#!/usr/bin/env python3
"""FS05: multi-scale head + SILog vs L1 (Eigen-style lesson)."""
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
from models import silog_torch  # noqa: E402
from synth_data import make_batch, make_scene  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs05_multiscale_silog"
OUT.mkdir(parents=True, exist_ok=True)


class MultiScaleDepth(nn.Module):
    """Tiny encoder with 2 prediction scales (full + 1/2)."""

    def __init__(self, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.head_half = nn.Conv2d(64, 1, 1)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.head_full = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.enc(x)
        d_half = self.max_depth * torch.sigmoid(self.head_half(f)) + 1e-3
        f2 = F.relu(self.up(f))
        if f2.shape[-2:] != (h // 2, w // 2) and f2.shape[-2] != h:
            f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        else:
            f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        d_full = self.max_depth * torch.sigmoid(self.head_full(f2)) + 1e-3
        d_half_up = F.interpolate(d_half, size=(h, w), mode="bilinear", align_corners=False)
        return d_full, d_half_up


def train(loss_name: str, multiscale: bool, steps: int = 90, seed: int = 0):
    torch.manual_seed(seed)
    model = MultiScaleDepth()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        rgb, depth, mask = make_batch(8, 64, 80, seed=seed + step, easy=True)
        d_full, d_half = model(rgb)
        if loss_name == "silog":
            loss = silog_torch(d_full, depth, mask)
            if multiscale:
                loss = loss + 0.5 * silog_torch(d_half, depth, mask)
        else:
            loss = (torch.abs(d_full - depth) * mask).sum() / (mask.sum() + 1e-8)
            if multiscale:
                loss = loss + 0.5 * (torch.abs(d_half - depth) * mask).sum() / (mask.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_model(model):
    model.eval()
    abs_rels = []
    with torch.no_grad():
        for i in range(10):
            s = make_scene(64, 80, seed=3000 + i, easy=True)
            rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)).float().unsqueeze(0)
            pred, _ = model(rgb)
            m = compute_metrics(pred.squeeze().numpy(), s["depth"], max_depth=20.0, align="none")
            abs_rels.append(m.abs_rel)
    return float(np.mean(abs_rels))


def main() -> None:
    configs = [
        ("l1_single", "l1", False),
        ("l1_ms", "l1", True),
        ("silog_single", "silog", False),
        ("silog_ms", "silog", True),
    ]
    results = {}
    models = {}
    for name, loss, ms in configs:
        print("train", name, flush=True)
        models[name] = train(loss, ms)
        results[name] = {"abs_rel": eval_model(models[name])}
        print(" ", results[name], flush=True)

    # visual: silog_ms vs l1_single
    s = make_scene(64, 80, seed=7, easy=True)
    rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        p_l1, _ = models["l1_single"](rgb)
        p_si, _ = models["silog_ms"](rgb)
    save_panels(
        OUT / "compare.png",
        [
            ("RGB", s["rgb"]),
            ("GT", s["depth"]),
            ("L1 single", p_l1.squeeze().numpy()),
            ("SILog multi-scale", p_si.squeeze().numpy()),
        ],
        ncols=4,
    )
    out = {
        "results": results,
        "new_capability": "Multi-scale + SILog stabilize scale-sensitive training (Eigen DNA).",
        "compare_to_fs04": "FS04 single-scale L1 tiny net; FS05 adds Eigen multi-scale/SILog recipe.",
        "artifacts": ["compare.png"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    assert results["silog_ms"]["abs_rel"] < 0.15
    print("FS05 acceptance: PASSED")


if __name__ == "__main__":
    main()
