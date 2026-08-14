#!/usr/bin/env python3
"""P5: metric monocular protocol — focal length, absolute scale, model family comparison (simulated + analytic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry import CameraIntrinsics, backproject  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from synth_data import make_scene  # noqa: E402

OUT = ROOT / "results" / "p5_metric"
OUT.mkdir(parents=True, exist_ok=True)


def simulate_metric_model(gt: np.ndarray, scale_bias: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pred = gt * scale_bias
    if noise > 0:
        pred = pred + rng.normal(0, noise, pred.shape)
    return np.clip(pred, 1e-3, None)


def simulate_relative_then_wrong_metric(gt: np.ndarray) -> np.ndarray:
    """Relative inverse depth wrongly treated as meters."""
    return 1.0 / np.maximum(gt, 1e-3)


def focal_depth_coupling(z: float, fx_true: float, fx_assumed: float, baseline_disp: float) -> float:
    """If disparity fixed, wrong fx scales depth: Z = fx * B / d."""
    d = fx_true * 0.5 / z  # fictitious
    return fx_assumed * 0.5 / d


def main() -> None:
    scenes = [make_scene(96, 128, seed=10 + i) for i in range(6)]
    families = {
        "metric_oracle": lambda g, i: simulate_metric_model(g, 1.0, 0.05, i),
        "metric_scale_bias_1.15": lambda g, i: simulate_metric_model(g, 1.15, 0.05, i),
        "relative_misused_as_metric": lambda g, i: simulate_relative_then_wrong_metric(g),
        "depth_pro_like": lambda g, i: simulate_metric_model(g, 1.02, 0.08, i),  # sharp metric-ish
        "metric3d_like": lambda g, i: simulate_metric_model(g, 0.98, 0.1, i),
        "unidepth_like": lambda g, i: simulate_metric_model(g, 1.05, 0.09, i),
    }
    table = []
    for name, fn in families.items():
        ms = []
        for i, s in enumerate(scenes):
            pred = fn(s["depth"], i)
            # metric protocol: align none
            m = compute_metrics(pred, s["depth"], max_depth=20.0, align="none")
            m_med = compute_metrics(pred, s["depth"], max_depth=20.0, align="median")
            ms.append({"none": m.as_dict(), "median": m_med.as_dict()})
        avg_none = {k: float(np.mean([x["none"][k] for x in ms])) for k in ms[0]["none"]}
        avg_med = {k: float(np.mean([x["median"][k] for x in ms])) for k in ms[0]["median"]}
        table.append({"family": name, "align_none": avg_none, "align_median": avg_med})
        print(name, "none abs_rel", avg_none["abs_rel"], "median", avg_med["abs_rel"])

    # Focal length ablation analytic
    z_true = 5.0
    fx_true = 500.0
    focal_rows = []
    for fx in [400, 450, 500, 550, 600]:
        z_hat = focal_depth_coupling(z_true, fx_true, fx, 0.0)
        focal_rows.append({"fx_assumed": fx, "z_hat": z_hat, "abs_rel": abs(z_hat - z_true) / z_true})

    # 3D point error from depth scale
    cam = CameraIntrinsics(500, 500, 64, 48)
    K = cam.K()
    depth = scenes[0]["depth"]
    pts = backproject(depth, K)
    pts_bad = backproject(depth * 1.2, K)
    point_rmse = float(np.sqrt(np.mean((pts - pts_bad) ** 2)))

    analysis = {
        "table": table,
        "focal_rows": focal_rows,
        "point_rmse_scale_1.2": point_rmse,
        "interpretation": {
            "metric_protocol": "True metric models: report align=none. Median alignment hides absolute scale bugs.",
            "relative_misuse": "Treating inverse depth as meters explodes AbsRel under align=none.",
            "focal": "Wrong fx linearly scales stereo/metric depth; Depth Pro-style fx estimation targets this failure.",
            "families": "Simulated DepthPro/Metric3D/UniDepth ranks by noise+bias — replace with real weights when GPU/weights available.",
        },
        "real_weights_status": "Environment has no multi-GB foundation weights cached; protocol + analytic focal/3D tests are the executable core.",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    # acceptance
    oracle = next(t for t in table if t["family"] == "metric_oracle")
    misuse = next(t for t in table if t["family"] == "relative_misused_as_metric")
    assert oracle["align_none"]["abs_rel"] < 0.1
    assert misuse["align_none"]["abs_rel"] > oracle["align_none"]["abs_rel"] * 3
    print("P5 acceptance: PASSED")


if __name__ == "__main__":
    main()
