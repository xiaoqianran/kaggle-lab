#!/usr/bin/env python3
"""P0 experiment: demonstrate alignment effects on synthetic errors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402

OUT = ROOT / "results" / "p0_alignment"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(42)
    gt = rng.uniform(1.0, 40.0, size=(120, 160))

    scenarios = {
        "perfect": gt.copy(),
        "scale_only_x2.5": gt * 2.5,
        "scale_shift": 0.4 * gt + 1.5,
        "additive_noise": gt + rng.normal(0, 0.3, gt.shape),
        "relative_like_inverse": 1.0 / gt,  # inverse-depth style wrong units
    }

    rows = []
    for name, pred in scenarios.items():
        for align in ("none", "median", "least_squares"):
            m = compute_metrics(pred, gt, max_depth=80.0, align=align)
            row = {"scenario": name, "align": align, **m.as_dict()}
            rows.append(row)
            print(f"{name:22s} align={align:14s} abs_rel={m.abs_rel:.4f} rmse={m.rmse:.4f} si={m.si_rmse:.4f} d1={m.delta1:.3f}")

    # Ablation interpretation payload
    analysis = {
        "finding_1": "Global scale error is nearly eliminated by median alignment; SI-RMSE already near 0 without align.",
        "finding_2": "Affine (scale+shift) errors need least_squares align; median alone leaves residual.",
        "finding_3": "Inverse-depth as if metric fails all protocols unless correctly inverted before eval.",
        "finding_4": "Additive noise cannot be fixed by scale alignment — true geometric/sensor error.",
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
