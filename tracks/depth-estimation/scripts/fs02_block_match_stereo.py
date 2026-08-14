#!/usr/bin/env python3
"""FS02: from-scratch block-matching stereo (NumPy SAD)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402

OUT = ROOT / "results" / "fs02_block_match"
OUT.mkdir(parents=True, exist_ok=True)


def make_stereo(h=80, w=120, z_near=3.0, z_far=8.0, fx=200.0, baseline=0.2, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    tex = 0.5 + 0.5 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    tex = (tex + 0.15 * rng.normal(size=(h, w))).clip(0, 1)
    depth = np.full((h, w), z_far, np.float64)
    depth[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = z_near
    tex[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = (
        0.2 + 0.6 * ((xx[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] % 7) / 7.0)
    )
    left = np.stack([tex, tex * 0.9, tex * 0.8], axis=-1)
    disp = fx * baseline / depth
    right = np.zeros_like(left)
    for v in range(h):
        for u in range(w):
            u_src = u + disp[v, u]
            u0 = int(np.floor(u_src))
            u1 = u0 + 1
            a = u_src - u0
            if 0 <= u0 < w and 0 <= u1 < w:
                right[v, u] = (1 - a) * left[v, u0] + a * left[v, u1]
            elif 0 <= u0 < w:
                right[v, u] = left[v, u0]
    return left, right, depth, disp, fx, baseline


def sad_block_match(left_g, right_g, max_disp=40, win=3):
    h, w = left_g.shape
    r = win // 2
    disp = np.zeros((h, w), np.float64)
    Lp = np.pad(left_g, r, mode="edge")
    Rp = np.pad(right_g, r, mode="edge")
    for v in range(h):
        for u in range(w):
            best_d, best_c = 0, 1e18
            patch_l = Lp[v : v + win, u : u + win]
            dmax = min(max_disp, u)
            for d in range(0, dmax + 1):
                uu = u - d
                if uu < 0:
                    break
                patch_r = Rp[v : v + win, uu : uu + win]
                c = np.abs(patch_l - patch_r).sum()
                if c < best_c:
                    best_c, best_d = c, d
            disp[v, u] = best_d
    return disp


def main() -> None:
    left, right, depth_gt, disp_gt, fx, B = make_stereo()
    left_g = left.mean(-1)
    right_g = right.mean(-1)

    disp_est = sad_block_match(left_g, right_g, max_disp=int(disp_gt.max()) + 5, win=5)
    depth_est = np.zeros_like(disp_est)
    m = disp_est > 0.5
    depth_est[m] = fx * B / disp_est[m]
    metrics = compute_metrics(depth_est, depth_gt, max_depth=20.0, align="none")

    # weak texture: constant images → matching is random / mostly zero
    left_w = np.full_like(left_g, 0.5)
    right_w = np.full_like(right_g, 0.5)
    disp_weak = sad_block_match(left_w, right_w, max_disp=30, win=5)
    depth_weak = np.zeros_like(disp_weak)
    mw = disp_weak > 0.5
    depth_weak[mw] = fx * B / np.maximum(disp_weak[mw], 1e-3)
    # if almost no disparity found, treat as total failure with huge dummy error
    if mw.sum() < 10:
        weak_fail = True
        metrics_weak = None
        weak_abs = float("inf")
    else:
        weak_fail = False
        metrics_weak = compute_metrics(depth_weak, depth_gt, max_depth=20.0, align="none")
        weak_abs = metrics_weak.abs_rel if np.isfinite(metrics_weak.abs_rel) else float("inf")

    out = {
        "fx": fx,
        "baseline": B,
        "disp_gt_mean": float(disp_gt.mean()),
        "disp_est_mean": float(disp_est.mean()),
        "metrics_textured": metrics.as_dict(),
        "weak_texture_total_failure": weak_fail,
        "metrics_weak_texture": None if metrics_weak is None else metrics_weak.as_dict(),
        "weak_abs_rel_or_inf": weak_abs if np.isfinite(weak_abs) else "inf",
        "compare_to_fs01": "FS01 used given depth; FS02 recovers depth from two images.",
        "observable_failure": "No texture ⇒ matching degenerates (zeros / random) — stereo's fundamental disease.",
    }
    np.savez_compressed(
        OUT / "arrays.npz",
        left=left,
        right=right,
        depth_gt=depth_gt,
        disp_est=disp_est,
        depth_est=depth_est,
    )
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert metrics.abs_rel < 0.25, metrics
    assert weak_fail or weak_abs > metrics.abs_rel * 1.5
    print("FS02 acceptance: PASSED")


if __name__ == "__main__":
    main()
