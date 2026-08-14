#!/usr/bin/env python3
"""FS04: minimal from-scratch monocular regressor (tiny CNN, no U-Net)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from synth_data import make_batch  # noqa: E402

OUT = ROOT / "results" / "fs04_tiny_regressor"
OUT.mkdir(parents=True, exist_ok=True)


class TinyReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x) * 10.0 + 1e-3


def main() -> None:
    torch.manual_seed(0)
    model = TinyReg()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(100):
        rgb, depth, mask = make_batch(8, 48, 64, seed=step, easy=True)
        pred = model(rgb)
        loss = (torch.abs(pred - depth) * mask).sum() / (mask.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    rgb, depth, mask = make_batch(4, 48, 64, seed=999, easy=True)
    with torch.no_grad():
        pred = model(rgb)
    m = compute_metrics(pred[0, 0].numpy(), depth[0, 0].numpy(), max_depth=20.0, align="none")

    # domain shift: hard multi-plane random colors (easy=False)
    rgb_h, depth_h, _ = make_batch(4, 48, 64, seed=123, easy=False)
    with torch.no_grad():
        pred_h = model(rgb_h)
    m_h = compute_metrics(pred_h[0, 0].numpy(), depth_h[0, 0].numpy(), max_depth=20.0, align="none")

    out = {
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
        "metrics_easy_holdout": m.as_dict(),
        "metrics_hard_domain_shift": m_h.as_dict(),
        "compare_to_fs02": "Stereo searched correspondences; FS04 memorizes monocular cues — collapses under domain shift.",
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert losses[-1] < losses[0] * 0.5
    assert m.abs_rel < 0.2
    assert m_h.abs_rel > m.abs_rel  # domain shift hurts
    print("FS04 acceptance: PASSED")


if __name__ == "__main__":
    main()
