"""Self-supervised photometric losses (Monodepth2-style mini implementation)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ssim(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
    """Structural similarity (1-SSIM)/2 style components; returns SSIM map in [0,1]."""
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def photometric_error(src: torch.Tensor, tgt: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """Per-pixel photometric error: alpha*SSIM + (1-alpha)*L1, mean over channels."""
    s = ssim(src, tgt).mean(1, keepdim=True)
    l1 = torch.abs(src - tgt).mean(1, keepdim=True)
    return alpha * s + (1 - alpha) * l1


def grid_sample_xy(img: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
    """Sample img (B,C,H,W) at absolute pixel coords (B,H,W,2) with x=u, y=v."""
    b, _, h, w = img.shape
    x = coords_xy[..., 0]
    y = coords_xy[..., 1]
    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def backproject_torch(depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
    """depth (B,1,H,W), inv_K (B,3,3) or (3,3) -> cam points (B,3,H,W)."""
    b, _, h, w = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).view(1, 3, -1).expand(b, -1, -1)  # B,3,N
    if inv_K.dim() == 2:
        inv_K = inv_K.unsqueeze(0).expand(b, -1, -1)
    cam = inv_K.bmm(pix)  # B,3,N
    cam = cam * depth.view(b, 1, -1)
    return cam.view(b, 3, h, w)


def project_torch(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """points (B,3,H,W), K (B,3,3)|(3,3) -> pixel coords (B,H,W,2)."""
    b, _, h, w = points.shape
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(b, -1, -1)
    flat = points.view(b, 3, -1)
    pix = K.bmm(flat)
    xy = pix[:, :2] / (pix[:, 2:3] + 1e-8)
    return xy.view(b, 2, h, w).permute(0, 2, 3, 1)


def warp_with_pose(
    img_src: torch.Tensor,
    depth_tgt: torch.Tensor,
    T_src_tgt: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp src image into tgt frame using tgt depth and pose tgt->src (4x4).

    Returns warped_src, valid_mask (B,1,H,W).
    """
    b, _, h, w = depth_tgt.shape
    if K.dim() == 2:
        K_b = K.unsqueeze(0).expand(b, -1, -1)
    else:
        K_b = K
    inv_K = torch.inverse(K_b)
    pts_tgt = backproject_torch(depth_tgt, inv_K)  # B,3,H,W
    ones = torch.ones(b, 1, h, w, device=depth_tgt.device, dtype=depth_tgt.dtype)
    pts_h = torch.cat([pts_tgt, ones], dim=1).view(b, 4, -1)  # B,4,N
    pts_src = T_src_tgt.bmm(pts_h)[:, :3].view(b, 3, h, w)
    coords = project_torch(pts_src, K_b)
    warped = grid_sample_xy(img_src, coords)
    valid = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] <= w - 1)
        & (coords[..., 1] >= 0)
        & (coords[..., 1] <= h - 1)
        & (depth_tgt[:, 0] > 1e-3)
        & (pts_src[:, 2] > 1e-3)
    ).unsqueeze(1).float()
    return warped, valid


def min_reprojection_loss(
    img_tgt: torch.Tensor,
    imgs_src: Tuple[torch.Tensor, ...],
    depth_tgt: torch.Tensor,
    poses_src_tgt: Tuple[torch.Tensor, ...],
    K: torch.Tensor,
) -> torch.Tensor:
    """Monodepth2-style minimum reprojection over source frames."""
    errors = []
    for img_s, T in zip(imgs_src, poses_src_tgt):
        warped, valid = warp_with_pose(img_s, depth_tgt, T, K)
        err = photometric_error(warped, img_tgt) * valid
        # invalidate non-visible with large error so min ignores them poorly;
        # instead set invalid to +inf-like
        err = err + (1.0 - valid) * 1e3
        errors.append(err)
    stacked = torch.cat(errors, dim=1)
    min_err, _ = torch.min(stacked, dim=1, keepdim=True)
    # only average finite/valid-ish
    mask = (min_err < 1e2).float()
    return (min_err * mask).sum() / (mask.sum() + 1e-8)
