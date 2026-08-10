#!/usr/bin/env python3
"""FS03: explicit disparity cost volume + WTA vs simple smoothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "fs03_cost_volume"
OUT.mkdir(parents=True, exist_ok=True)


def make_pair(h=48, w=64, fx=150.0, B=0.15, z0=4.0, z1=7.0, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    tex = 0.5 + 0.5 * np.sin(xx / 2.5) * np.cos(yy / 3.0) + 0.05 * rng.normal(size=(h, w))
    tex = tex.clip(0, 1)
    depth = np.full((h, w), z1)
    depth[:, w // 2 :] = z0
    disp = fx * B / depth
    left = tex
    right = np.zeros_like(left)
    for v in range(h):
        for u in range(w):
            us = u + disp[v, u]
            u0 = int(np.floor(us))
            a = us - u0
            if 0 <= u0 < w - 1:
                right[v, u] = (1 - a) * left[v, u0] + a * left[v, u0 + 1]
    return left, right, depth, disp, int(np.ceil(disp.max()) + 2)


def build_cost_volume(left, right, max_disp, win=3):
    h, w = left.shape
    r = win // 2
    Lp = np.pad(left, r, mode="edge")
    Rp = np.pad(right, r, mode="edge")
    cost = np.zeros((max_disp + 1, h, w), np.float64)
    for d in range(max_disp + 1):
        # shift right image to the right by d to compare with left
        shifted = np.zeros_like(Rp)
        if d == 0:
            shifted = Rp
        else:
            shifted[:, d:] = Rp[:, :-d]
            shifted[:, :d] = Rp[:, :1]
        # windowed SAD via integral-less loop (small res)
        for v in range(h):
            for u in range(w):
                pl = Lp[v : v + win, u : u + win]
                pr = shifted[v : v + win, u : u + win]
                cost[d, v, u] = np.abs(pl - pr).sum()
    return cost


def wta(cost):
    return cost.argmin(axis=0).astype(np.float64)


def smooth_disp(disp, iters=5):
    """Very simple 4-neighbor average for demo (not SGM)."""
    out = disp.copy()
    for _ in range(iters):
        p = np.pad(out, 1, mode="edge")
        out = 0.2 * (
            p[1:-1, 1:-1] * 1
            + p[:-2, 1:-1]
            + p[2:, 1:-1]
            + p[1:-1, :-2]
            + p[1:-1, 2:]
        )
    return out


def main() -> None:
    left, right, depth, disp_gt, max_d = make_pair()
    cost = build_cost_volume(left, right, max_d, win=3)
    d_wta = wta(cost)
    d_smooth = smooth_disp(d_wta, iters=8)

    # pick a pixel on the step edge and far from edge
    h, w = left.shape
    edge_uv = (h // 2, w // 2)
    flat_uv = (h // 2, w // 4)
    curve_edge = cost[:, edge_uv[0], edge_uv[1]].tolist()
    curve_flat = cost[:, flat_uv[0], flat_uv[1]].tolist()

    err_wta = float(np.mean(np.abs(d_wta - disp_gt)))
    err_smooth = float(np.mean(np.abs(d_smooth - disp_gt)))

    out = {
        "max_disp": max_d,
        "err_wta_px": err_wta,
        "err_smooth_px": err_smooth,
        "cost_curve_edge_pixel": curve_edge,
        "cost_curve_flat_pixel": curve_flat,
        "argmin_edge": int(np.argmin(curve_edge)),
        "argmin_flat": int(np.argmin(curve_flat)),
        "compare_to_fs02": "FS02 hid the cost volume inside nested loops; FS03 makes D×H×W explicit.",
        "lesson": "Ambiguous cost curves at edges/occlusion; smoothing trades edge sharpness for stability.",
    }
    np.savez_compressed(OUT / "cost.npz", cost=cost, disp_wta=d_wta, disp_gt=disp_gt)
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if "curve" not in k}, indent=2))
    assert err_wta < 3.0, err_wta
    print("FS03 acceptance: PASSED")


if __name__ == "__main__":
    main()
