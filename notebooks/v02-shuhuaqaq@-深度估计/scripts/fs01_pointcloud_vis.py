#!/usr/bin/env python3
"""FS01: pinhole back-project → save depth + pointcloud projections (no GUI)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry import CameraIntrinsics, backproject, synthesize_fronto_plane_depth  # noqa: E402
from synth_data import make_scene  # noqa: E402

OUT = ROOT / "results" / "fs01_pointcloud"
OUT.mkdir(parents=True, exist_ok=True)


def project_xy(pts: np.ndarray, mode: str) -> np.ndarray:
    """Orthographic scatter of 3D points for 2D PNG-less numeric summary."""
    if mode == "xz":
        return pts[..., [0, 2]]
    if mode == "xy":
        return pts[..., [0, 1]]
    if mode == "yz":
        return pts[..., [1, 2]]
    raise ValueError(mode)


def stats_for_fx(fx: float, scene_depth: np.ndarray) -> dict:
    h, w = scene_depth.shape
    cam = CameraIntrinsics(fx, fx, w / 2, h / 2)
    pts = backproject(scene_depth, cam.K())
    valid = np.isfinite(scene_depth) & (scene_depth > 0)
    X = pts[valid]
    return {
        "fx": fx,
        "mean_X": float(X[:, 0].mean()),
        "std_X": float(X[:, 0].std()),
        "mean_Z": float(X[:, 2].mean()),
        "span_X": float(X[:, 0].max() - X[:, 0].min()),
        "n": int(valid.sum()),
    }


def save_ascii_cloud(path: Path, pts: np.ndarray, max_n: int = 2000) -> None:
    flat = pts.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(1)]
    if len(flat) > max_n:
        idx = np.linspace(0, len(flat) - 1, max_n).astype(int)
        flat = flat[idx]
    np.savetxt(path, flat, fmt="%.5f", header="X Y Z", comments="")


def main() -> None:
    scene = make_scene(96, 128, seed=0, easy=True)
    depth = scene["depth"]
    # also fronto plane for clean geometry lesson
    fronto = synthesize_fronto_plane_depth(96, 128, 5.0)

    rows = []
    for fx in (300.0, 500.0, 800.0):
        rows.append(stats_for_fx(fx, depth))

    cam = CameraIntrinsics(500, 500, 64, 48)
    pts = backproject(depth, cam.K())
    save_ascii_cloud(OUT / "cloud_fx500.xyz", pts)

    # scale depth by 1.2 → points should scale
    pts_s = backproject(depth * 1.2, cam.K())
    ratio = float(np.nanmean(pts_s[..., 2] / np.maximum(pts[..., 2], 1e-6)))

    # fronto: X span should shrink when fx increases for same pixel FOV content
    fronto_rows = [stats_for_fx(fx, fronto) for fx in (300.0, 500.0, 800.0)]

    out = {
        "lesson": "Larger fx (same depth map pixels) pulls X,Y toward optical axis — FOV narrower interpretation.",
        "depth_scale_1.2_Z_ratio": ratio,
        "easy_scene_by_fx": rows,
        "fronto_Z5_by_fx": fronto_rows,
        "artifacts": str(OUT / "cloud_fx500.xyz"),
        "compare_to_fs00": "FS00 showed metric tables; FS01 shows the same depth error becomes 3D scale error.",
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert abs(ratio - 1.2) < 1e-5
    # span_X decreases as fx increases on fronto plane
    spans = [r["span_X"] for r in fronto_rows]
    assert spans[0] > spans[1] > spans[2], spans
    print("FS01 acceptance: PASSED")


if __name__ == "__main__":
    main()
