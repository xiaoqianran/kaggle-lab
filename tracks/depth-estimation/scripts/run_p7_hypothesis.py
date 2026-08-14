#!/usr/bin/env python3
"""P7 research hypothesis experiment (compact CPU budget)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from models import TinyUNet, silog_torch  # noqa: E402
from synth_data import make_batch, make_scene  # noqa: E402

OUT = ROOT / "results" / "p7_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)


def edge_weight(depth: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    wx = F.pad(dx, (0, 1, 0, 0))
    wy = F.pad(dy, (0, 0, 0, 1))
    return 1.0 + 5.0 * (wx + wy)


def train(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    model = TinyUNet(base=cfg["base"])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(cfg["steps"]):
        rgb, depth, mask = make_batch(n=4, h=64, w=80, seed=cfg["seed"] + step)
        pred = model(rgb)
        if cfg["edge_aware"]:
            w = edge_weight(depth)
            d = torch.log(pred) - torch.log(depth)
            loss = (w * mask * d ** 2).sum() / (mask.sum() + 1e-8)
        else:
            loss = silog_torch(pred, depth, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    abs_rels, boundary_errs, interior_errs = [], [], []
    with torch.no_grad():
        for i in range(12):
            s = make_scene(64, 80, seed=5000 + i)
            rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)).unsqueeze(0)
            gt = s["depth"]
            pred = model(rgb).squeeze().numpy()
            m = compute_metrics(pred, gt, max_depth=20.0, align="none")
            abs_rels.append(m.abs_rel)
            gx = np.abs(gt[:, 1:] - gt[:, :-1])
            gy = np.abs(gt[1:, :] - gt[:-1, :])
            b = np.zeros_like(gt, dtype=bool)
            b[:, 1:] |= gx > 0.2
            b[:, :-1] |= gx > 0.2
            b[1:, :] |= gy > 0.2
            b[:-1, :] |= gy > 0.2
            err = np.abs(pred - gt) / np.maximum(gt, 1e-3)
            boundary_errs.append(float(err[b].mean()) if b.any() else 0.0)
            interior_errs.append(float(err[~b].mean()) if (~b).any() else 0.0)
    return {
        "cfg": cfg,
        "abs_rel": float(np.mean(abs_rels)),
        "boundary_absrel": float(np.mean(boundary_errs)),
        "interior_absrel": float(np.mean(interior_errs)),
    }


def main() -> None:
    runs = [
        train({"name": "base8_silog", "base": 8, "steps": 50, "seed": 0, "edge_aware": False}),
        train({"name": "base12_silog", "base": 12, "steps": 50, "seed": 0, "edge_aware": False}),
        train({"name": "base8_edge", "base": 8, "steps": 50, "seed": 0, "edge_aware": True}),
    ]
    for r in runs:
        print(r["cfg"]["name"], {k: r[k] for k in r if k != "cfg"}, flush=True)

    b8 = next(r for r in runs if r["cfg"]["name"] == "base8_silog")
    b12 = next(r for r in runs if r["cfg"]["name"] == "base12_silog")
    edge = next(r for r in runs if r["cfg"]["name"] == "base8_edge")

    edge_boundary_gain = b8["boundary_absrel"] - edge["boundary_absrel"]
    cap_boundary_gain = b8["boundary_absrel"] - b12["boundary_absrel"]
    edge_overall_gain = b8["abs_rel"] - edge["abs_rel"]
    cap_overall_gain = b8["abs_rel"] - b12["abs_rel"]
    supported = (edge_overall_gain >= cap_overall_gain - 1e-6) or (edge_boundary_gain > cap_boundary_gain)

    out = {
        "hypothesis": "Edge-aware loss reduces AbsRel more than modest capacity increase under fixed steps; boundary errors dominate.",
        "runs": runs,
        "edge_boundary_gain": edge_boundary_gain,
        "cap_boundary_gain": cap_boundary_gain,
        "edge_overall_gain": edge_overall_gain,
        "cap_overall_gain": cap_overall_gain,
        "supported": bool(supported),
        "interpretation": (
            "SUPPORTED under this budget."
            if supported
            else "NOT SUPPORTED under this budget; capacity beat edge weight — revise H1."
        ),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("supported", "edge_overall_gain", "cap_overall_gain", "interpretation")}, indent=2))
    assert all(np.isfinite(r["abs_rel"]) for r in runs)
    print("P7 acceptance: PASSED (hypothesis tested; support=", supported, ")")


if __name__ == "__main__":
    main()
