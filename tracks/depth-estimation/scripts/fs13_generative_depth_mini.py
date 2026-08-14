#!/usr/bin/env python3
"""FS13: generative-style depth — iterative denoising of depth maps (toy diffusion).

Not Marigold weights (multi-GB); teaches the *capability difference*:
iterative refinement vs single forward pass, with uncertainty via multi-sample.
"""
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
from synth_data import make_batch, make_scene  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs13_generative_mini"
OUT.mkdir(parents=True, exist_ok=True)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # RGB + noisy depth
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, rgb, depth_noisy):
        x = torch.cat([rgb, depth_noisy], dim=1)
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    model = Denoiser()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(120):
        rgb, depth, mask = make_batch(8, 48, 64, seed=step, easy=True)
        # normalize depth 0-1
        d = depth / 10.0
        t = torch.rand(rgb.size(0), 1, 1, 1)
        noise = torch.randn_like(d)
        noisy = (1 - t) * d + t * noise
        pred = model(rgb, noisy)
        loss = (F.mse_loss(pred, d, reduction="none") * mask).sum() / (mask.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # single-step regression baseline: predict depth from rgb only
    class OneShot(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, rgb):
            return self.net(rgb)

    one = OneShot()
    opt2 = torch.optim.Adam(one.parameters(), lr=2e-3)
    for step in range(120):
        rgb, depth, mask = make_batch(8, 48, 64, seed=step, easy=True)
        d = depth / 10.0
        pred = one(rgb)
        loss = (F.mse_loss(pred, d, reduction="none") * mask).sum() / (mask.sum() + 1e-8)
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    def multi_sample_denoise(rgb, steps=5, samples=4):
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(samples):
                x = torch.randn(1, 1, rgb.shape[-2], rgb.shape[-1])
                for k in range(steps):
                    # simple fixed schedule toward clean
                    eps = model(rgb, x)
                    x = 0.5 * x + 0.5 * eps  # pull to prediction
                preds.append(x)
        stack = torch.cat(preds, dim=0)
        return stack.mean(0, keepdim=True), stack.std(0, keepdim=True)

    s = make_scene(48, 64, seed=42, easy=True)
    rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)[None].astype(np.float32))
    mean_d, std_d = multi_sample_denoise(rgb)
    with torch.no_grad():
        one_pred = one(rgb)

    gt = s["depth"] / 10.0
    m_gen = compute_metrics(mean_d.squeeze().numpy() * 10, s["depth"], max_depth=20.0, align="none")
    m_one = compute_metrics(one_pred.squeeze().numpy() * 10, s["depth"], max_depth=20.0, align="none")

    save_panels(
        OUT / "generative_vs_oneshot.png",
        [
            ("RGB", s["rgb"]),
            ("GT", s["depth"]),
            ("one-shot", one_pred.squeeze().numpy() * 10),
            ("gen mean", mean_d.squeeze().numpy() * 10),
            ("gen std (uncert.)", std_d.squeeze().numpy()),
        ],
        ncols=5,
    )

    out = {
        "denoise_train_loss_start": losses[0],
        "denoise_train_loss_end": losses[-1],
        "abs_rel_generative_mean": m_gen.abs_rel,
        "abs_rel_oneshot": m_one.abs_rel,
        "new_capability": "Iterative generative refinement + multi-sample uncertainty map.",
        "compare_to_fs11_12": "Discriminative foundation is one forward; generative trades compute for sampling/uncertainty.",
        "note": "Toy diffusion on synthetic data — Marigold-scale models need GPU+weights (optional upgrade).",
        "artifacts": ["generative_vs_oneshot.png"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert losses[-1] < losses[0]
    assert np.isfinite(m_gen.abs_rel) and np.isfinite(m_one.abs_rel)
    print("FS13 acceptance: PASSED")


if __name__ == "__main__":
    main()
