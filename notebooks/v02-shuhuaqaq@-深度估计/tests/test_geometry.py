"""P1 acceptance tests for camera geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry import (  # noqa: E402
    CameraIntrinsics,
    backproject,
    depth_to_disparity,
    disparity_to_depth,
    project_points,
    relative_pose_warp_points,
    synthesize_fronto_plane_depth,
)


def test_project_backproject_roundtrip():
    cam = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    K = cam.K()
    X = np.array([[0.0, 0.0, 5.0], [1.0, -0.5, 4.0], [-0.2, 0.3, 8.0]])
    uv = project_points(X, K)
    Kinv = np.linalg.inv(K)
    for i in range(len(X)):
        u, v = uv[i]
        ray = Kinv @ np.array([u, v, 1.0])
        # recover X = (Z / ray_z) * ray
        scale = X[i, 2] / ray[2]
        got = scale * ray
        assert np.allclose(got, X[i], atol=1e-8), (got, X[i])

    # dense map backproject: center pixel of fronto plane
    depth = synthesize_fronto_plane_depth(480, 640, 5.0)
    pts = backproject(depth, K)
    # optical center pixel (cx,cy) should be (0,0,Z)
    cx, cy = int(cam.cx), int(cam.cy)
    assert np.allclose(pts[cy, cx], [0.0, 0.0, 5.0], atol=1e-6)


def test_stereo_disparity_depth_inverse():
    fx, B = 700.0, 0.54
    z = np.array([5.0, 10.0, 20.0, 40.0])
    d = depth_to_disparity(z, B, fx)
    z2 = disparity_to_depth(d, B, fx)
    assert np.allclose(z, z2, rtol=1e-6)
    assert d[0] > d[-1]


def test_fronto_plane_warp_identity():
    cam = CameraIntrinsics(500, 500, 160, 120)
    K = cam.K()
    h, w = 240, 320
    depth = synthesize_fronto_plane_depth(h, w, 5.0)
    R = np.eye(3)
    t = np.zeros(3)
    coords, valid = relative_pose_warp_points(depth, K, R, t)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    assert np.allclose(coords[..., 0][valid], us[valid], atol=1e-4)
    assert np.allclose(coords[..., 1][valid], vs[valid], atol=1e-4)


def test_baseline_translation_shifts_u():
    fx = 500.0
    cam = CameraIntrinsics(fx, fx, 160, 120)
    K = cam.K()
    h, w = 120, 160
    z = 5.0
    depth = synthesize_fronto_plane_depth(h, w, z)
    B = 0.5
    R = np.eye(3)
    t = np.array([-B, 0.0, 0.0])
    coords, valid = relative_pose_warp_points(depth, K, R, t)
    us, _ = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    expected_shift = fx * B / z
    du = coords[..., 0] - us
    mid = valid & (us > 10) & (us < w - 10)
    mean_du = float(np.mean(du[mid]))
    assert abs(abs(mean_du) - expected_shift) < 0.5, (mean_du, expected_shift)


if __name__ == "__main__":
    test_project_backproject_roundtrip()
    test_stereo_disparity_depth_inverse()
    test_fronto_plane_warp_identity()
    test_baseline_translation_shifts_u()
    print("P1 test_geometry: ALL PASSED")
