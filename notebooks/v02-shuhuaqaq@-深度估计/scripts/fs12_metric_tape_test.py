#!/usr/bin/env python3
"""FS12: metric vs relative — synthetic tape-measure object of known height."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry import CameraIntrinsics  # noqa: E402
from metrics import align_scale_shift, compute_metrics  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs12_metric_tape"
OUT.mkdir(parents=True, exist_ok=True)


def render_box_scene(h=160, w=200, fx=400.0, box_depth=3.0, box_height_m=1.0):
    cam = CameraIntrinsics(fx, fx, w / 2, h / 2)
    yy, _ = np.mgrid[0:h, 0:w].astype(np.float64)
    ground_z = 2.0 + 5.0 * (yy / (h - 1))
    depth = ground_z.copy()
    pix_h = int(round(fx * box_height_m / box_depth))
    bottom = h - 20
    top = bottom - pix_h
    x0, x1 = w // 2 - 25, w // 2 + 25
    depth[top:bottom, x0:x1] = box_depth
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 1] = 0.35
    rgb[top:bottom, x0:x1] = np.array([0.85, 0.2, 0.15], np.float32)
    return rgb, depth, cam, (top, bottom, x0, x1), box_height_m


def estimate_height_m(depth: np.ndarray, cam: CameraIntrinsics, box) -> float:
    top, bottom, x0, x1 = box
    uc = (x0 + x1) // 2
    z = float(depth[top, uc])
    return abs((bottom - 1 - top) * z / cam.fy)


def main() -> None:
    rgb, depth_gt, cam, box, H_true = render_box_scene()
    h_gt = estimate_height_m(depth_gt, cam, box)

    rng = np.random.default_rng(0)
    depth_metric = np.clip(depth_gt * 1.02 + rng.normal(0, 0.02, depth_gt.shape), 0.1, None)
    h_metric = estimate_height_m(depth_metric, cam, box)

    # Relative foundation output: disparity-like 1/Z (scale free)
    disp = 1.0 / np.maximum(depth_gt, 1e-3)
    # Product ships a fixed calibration constant from *another* dataset (wrong)
    # Z = c / disp  with wrong c
    c_wrong = 0.8  # true c would be 1.0 for this synthetic
    depth_rel_product = c_wrong / disp
    h_rel_product = estimate_height_m(depth_rel_product, cam, box)

    # Research eval: LS align relative to GT (oracle)
    depth_rel_oracle = align_scale_shift(disp, depth_gt)
    depth_rel_oracle = np.clip(depth_rel_oracle, 0.1, None)
    h_rel_oracle = estimate_height_m(depth_rel_oracle, cam, box)

    cam_bad = CameraIntrinsics(cam.fx * 1.25, cam.fy * 1.25, cam.cx, cam.cy)
    h_bad_fx = estimate_height_m(depth_gt, cam_bad, box)

    m_metric = compute_metrics(depth_metric, depth_gt, max_depth=20.0, align="none")
    m_rel_prod = compute_metrics(depth_rel_product, depth_gt, max_depth=20.0, align="none")

    save_panels(
        OUT / "tape_scene.png",
        [
            ("RGB", rgb),
            ("GT metric", depth_gt),
            ("metric model", depth_metric),
            ("relative product scale", depth_rel_product),
        ],
        ncols=4,
    )

    out = {
        "true_box_height_m": H_true,
        "height_from_gt_depth_m": h_gt,
        "height_metric_model_m": h_metric,
        "height_relative_product_m": h_rel_product,
        "height_relative_oracle_aligned_m": h_rel_oracle,
        "height_wrong_fx_m": h_bad_fx,
        "metric_abs_rel": m_metric.abs_rel,
        "relative_product_abs_rel": m_rel_prod.abs_rel,
        "new_capability": "Tape-measure test: metric OK, wrong relative scale fails meters.",
        "compare_to_fs11": "FS11 learns relative; FS12 shows why metric calibration matters.",
        "lesson": "Oracle LS alignment is research-only; products need correct absolute scale.",
        "artifacts": ["tape_scene.png"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert abs(h_gt - H_true) / H_true < 0.02, (h_gt, H_true)
    assert abs(h_metric - H_true) / H_true < 0.12, (h_metric, H_true)
    assert abs(h_rel_product - H_true) / H_true > 0.15, (h_rel_product, H_true)
    assert abs(h_bad_fx - H_true) / H_true > 0.15
    print("FS12 acceptance: PASSED")


if __name__ == "__main__":
    main()
