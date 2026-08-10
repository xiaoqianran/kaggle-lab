#!/usr/bin/env python3
"""FS00: metrics + visual error maps for classic failure modes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import align_median, align_scale_shift, compute_metrics  # noqa: E402
from synth_data import make_scene  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs00_metrics"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    gt = make_scene(96, 128, seed=0, easy=True)["depth"]
    preds = {
        "scale_x2": gt * 2.0,
        "affine": 0.4 * gt + 1.5,
        "inverse": 1.0 / np.maximum(gt, 1e-3),
        "noise": gt + np.random.default_rng(0).normal(0, 0.4, gt.shape),
    }
    rows = []
    panels = [("GT depth", gt)]
    for name, pred in preds.items():
        for align in ("none", "median", "least_squares"):
            m = compute_metrics(pred, gt, max_depth=20.0, align=align)
            rows.append({"pred": name, "align": align, **m.as_dict()})
        # visualize abs rel error after best-effort LS (except show raw too)
        err = np.abs(pred - gt) / np.maximum(gt, 1e-3)
        panels.append((f"{name} raw err", err))
        aligned = align_scale_shift(pred, gt)
        err_a = np.abs(aligned - gt) / np.maximum(gt, 1e-3)
        panels.append((f"{name} LS err", err_a))

    save_panels(OUT / "error_maps.png", panels, ncols=3)
    # show inverse cannot be fixed
    inv = preds["inverse"]
    inv_med = align_median(inv, gt)
    inv_ls = align_scale_shift(inv, gt)
    save_panels(
        OUT / "inverse_depth_trap.png",
        [
            ("GT", gt),
            ("pred=1/Z", inv),
            ("median align", inv_med),
            ("LS align", inv_ls),
        ],
        ncols=4,
    )
    summary = {
        "rows": rows,
        "artifacts": ["error_maps.png", "inverse_depth_trap.png"],
        "new_capability": "Read metrics tables AND see which errors alignment can/cannot fix.",
        "compare_to_prev": "Start of chain — no prior method.",
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # acceptance: scale_x2 median abs_rel ~0
    r = next(x for x in rows if x["pred"] == "scale_x2" and x["align"] == "median")
    assert r["abs_rel"] < 1e-6
    print("FS00 acceptance: PASSED")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
