#!/usr/bin/env python3
"""P2: train TinyUNet on synthetic RGB-D; ablate loss and capacity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from models import TinyUNet, silog_torch  # noqa: E402
from synth_data import make_batch  # noqa: E402

OUT = ROOT / "results" / "p2_supervised"
OUT.mkdir(parents=True, exist_ok=True)


def eval_model(model: torch.nn.Module, n: int = 8, seed: int = 999) -> dict:
    model.eval()
    rgb, depth, mask = make_batch(n=n, h=64, w=80, seed=seed, easy=True)
    with torch.no_grad():
        pred = model(rgb)
    p = pred.squeeze(1).cpu().numpy()
    g = depth.squeeze(1).cpu().numpy()
    acc = [compute_metrics(p[i], g[i], max_depth=20.0, align="none").as_dict() for i in range(n)]
    keys = acc[0].keys()
    return {k: float(np.mean([a[k] for a in acc])) for k in keys}


def train_one(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    model = TinyUNet(base=cfg["base"], max_depth=10.0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    for step in range(cfg["steps"]):
        rgb, depth, mask = make_batch(n=cfg["bs"], h=64, w=80, seed=cfg["seed"] + step, easy=True)
        pred = model(rgb)
        if cfg["loss"] == "silog":
            loss = silog_torch(pred, depth, mask)
        elif cfg["loss"] == "l1":
            loss = (torch.abs(pred - depth) * mask).sum() / (mask.sum() + 1e-8)
        else:
            raise ValueError(cfg["loss"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == cfg["steps"] - 1:
            history.append({"step": step, "loss": float(loss.item())})
    metrics = eval_model(model)
    return {"cfg": cfg, "history": history, "metrics": metrics}


def main() -> None:
    ablations = [
        {"name": "silog_b16", "loss": "silog", "base": 16, "lr": 3e-3, "bs": 8, "steps": 80, "seed": 0},
        {"name": "l1_b16", "loss": "l1", "base": 16, "lr": 3e-3, "bs": 8, "steps": 80, "seed": 0},
        {"name": "silog_b8", "loss": "silog", "base": 8, "lr": 3e-3, "bs": 8, "steps": 80, "seed": 0},
    ]
    results = []
    for ab in ablations:
        print("training", ab["name"], "...", flush=True)
        r = train_one(ab)
        print("  metrics", r["metrics"], flush=True)
        results.append(r)

    m0 = eval_model(TinyUNet(base=16), seed=999)
    analysis = {
        "untrained": m0,
        "runs": results,
        "interpretation": {
            "data": "Easy synthetic: depth correlated with color + rectangle object.",
            "expect_trained_better": "Trained AbsRel << untrained; delta1 rises.",
            "capacity": "base=8 vs 16 under fixed steps.",
        },
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    silog = next(r for r in results if r["cfg"]["name"] == "silog_b16")["metrics"]
    print("untrained", m0)
    print("wrote", OUT / "results.json")
    assert silog["abs_rel"] < m0["abs_rel"] * 0.5, (silog, m0)
    assert silog["abs_rel"] < 0.25, silog
    assert silog["delta1"] > 0.5, silog
    print("P2 acceptance: PASSED")


if __name__ == "__main__":
    main()
