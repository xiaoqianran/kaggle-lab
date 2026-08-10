"""Minimal camera geometry for monocular/stereo depth.

Pinhole projection, back-projection, stereo disparity↔depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def project_points(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points (N,3) in camera frame to pixels (N,2)."""
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == 3
    x = (K @ X.T).T  # (N,3)
    return x[:, :2] / (x[:, 2:3] + 1e-12)


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth map (H,W) to camera-frame points (H,W,3)."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def disparity_to_depth(disp: np.ndarray, baseline: float, fx: float) -> np.ndarray:
    """Z = f * B / d  for rectified stereo (disp in pixels)."""
    disp = np.asarray(disp, dtype=np.float64)
    out = np.zeros_like(disp)
    valid = disp > 1e-6
    out[valid] = (fx * baseline) / disp[valid]
    return out


def depth_to_disparity(depth: np.ndarray, baseline: float, fx: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    out = np.zeros_like(depth)
    valid = depth > 1e-6
    out[valid] = (fx * baseline) / depth[valid]
    return out


def relative_pose_warp_points(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp pixels of frame i to frame j given depth_i and pose i→j (R,t).

    Returns:
      coords_j: (H,W,2) sampling coordinates in j
      valid: (H,W) boolean
    """
    pts = backproject(depth, K)  # H,W,3
    h, w = depth.shape
    X = pts.reshape(-1, 3).T  # 3,N
    Xj = R @ X + t.reshape(3, 1)
    zj = Xj[2]
    uj = (K[0, 0] * Xj[0] / (zj + 1e-12) + K[0, 2]).reshape(h, w)
    vj = (K[1, 1] * Xj[1] / (zj + 1e-12) + K[1, 2]).reshape(h, w)
    coords = np.stack([uj, vj], axis=-1)
    valid = (
        (depth > 1e-3)
        & (zj.reshape(h, w) > 1e-3)
        & (uj >= 0)
        & (uj <= w - 1)
        & (vj >= 0)
        & (vj <= h - 1)
    )
    return coords, valid


def synthesize_fronto_plane_depth(h: int, w: int, z: float) -> np.ndarray:
    return np.full((h, w), float(z), dtype=np.float64)


def synthesize_slanted_plane_depth(
    h: int, w: int, K: np.ndarray, plane_n: np.ndarray, plane_d: float
) -> np.ndarray:
    """Depth for plane n·X + d = 0 with ||n||=1-ish, camera at origin looking +Z.

    For pixel (u,v), ray r = K^{-1}[u,v,1], X = lambda r intersects plane:
    n·(lambda r) + d = 0 => lambda = -d / (n·r), Z = lambda.
    """
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(us, dtype=np.float64)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3).T  # 3,N
    Kinv = np.linalg.inv(K)
    rays = Kinv @ pix  # 3,N
    n = plane_n.reshape(3, 1)
    denom = (n.T @ rays).ravel()
    lam = -plane_d / (denom + 1e-12)
    z = lam  # if ray is not unit, this is the scale along ray; for pinhole K^{-1}u with z-comp
    # Better: X = lambda * ray, depth = X_z
    depth = (lam * rays[2]).reshape(h, w)
    depth[depth <= 0] = np.nan
    return depth
