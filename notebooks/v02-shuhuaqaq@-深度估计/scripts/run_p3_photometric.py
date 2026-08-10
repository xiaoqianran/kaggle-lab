#!/usr/bin/env python3
"""P3: mini photometric / warp tests (Monodepth2 ideas)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from photometric import min_reprojection_loss, photometric_error, warp_with_pose  # noqa: E402

OUT = ROOT / "results" / "p3_photometric"
OUT.mkdir(parents=True, exist_ok=True)


def make_textured_plane(h=64, w=96, z=4.0, fx=200.0, baseline=0.2, seed=0):
    """Left image + geometrically correct right image for fronto-parallel plane."""
    rng = torch.Generator().manual_seed(seed)
    # high-frequency texture
    yy = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    tex = 0.5 + 0.5 * torch.sin(40 * xx + 10 * yy) * torch.cos(25 * yy)
    tex = tex + 0.1 * torch.randn(h, w, generator=rng)
    tex = tex.clamp(0, 1)
    rgb_l = tex.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # 1,3,H,W
    depth = torch.full((1, 1, h, w), z)
    disp = fx * baseline / z  # pixels
    # right image: sample left at u + disp (right camera)
    # If point at depth z, translation of camera by +baseline (x), u_r = u_l - f*B/z
    us = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(1, 1, h, w)
    vs = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1).expand(1, 1, h, w)
    u_src = us + disp  # sample from left to form right? 
    # Construct right by sampling left: for each right pixel u_r, corresponding left u_l = u_r + disp
    grid_x = 2.0 * (us + disp) / max(w - 1, 1) - 1.0
    grid_y = 2.0 * vs / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x.squeeze(1), grid_y.squeeze(1)], dim=-1)
    rgb_r = F.grid_sample(rgb_l, grid, mode="bilinear", padding_mode="border", align_corners=True)
    K = torch.tensor([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=torch.float32)
    # T_src_tgt: transform points from tgt(left) to src(right)
    # right camera is at +baseline along x in left frame; X_r = X_l - (B,0,0) if R=I... 
    # Actually if world=left cam, right cam center is +B on x, pose of right: t = (+B,0,0), 
    # point in right = R^T (X - t) = X - t with R=I => T_r_l maps left points to right: X_r = X_l - B e_x
    T = torch.eye(4).unsqueeze(0)
    T[0, 0, 3] = -baseline
    return rgb_l, rgb_r, depth, K, T, disp


def main() -> None:
    torch.manual_seed(0)
    img_l, img_r, depth, K, T, disp = make_textured_plane()
    warped, valid = warp_with_pose(img_r, depth, T, K)
    err_good = photometric_error(warped, img_l)
    err_good = (err_good * valid).sum() / (valid.sum() + 1e-8)

    warped_bad, valid_b = warp_with_pose(img_r, depth * 2.0, T, K)
    err_bad = photometric_error(warped_bad, img_l)
    err_bad = (err_bad * valid_b).sum() / (valid_b.sum() + 1e-8)

    # identity image as second source for min reproj
    T_id = torch.eye(4).unsqueeze(0)
    loss_min = min_reprojection_loss(img_l, (img_r, img_l), depth, (T, T_id), K)

    # optimize depth with photo loss
    depth_param = torch.nn.Parameter(torch.full_like(depth, 6.0))  # wrong init
    opt = torch.optim.Adam([depth_param], lr=0.1)
    losses = []
    for i in range(50):
        opt.zero_grad()
        d = F.softplus(depth_param) + 0.5
        loss = min_reprojection_loss(img_l, (img_r,), d, (T,), K)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    results = {
        "disp_px": float(disp),
        "err_good_depth": float(err_good.item()),
        "err_bad_depth": float(err_bad.item()),
        "loss_min": float(loss_min.item()),
        "photo_train_losses_head": losses[:5],
        "photo_train_losses_tail": losses[-5:],
        "photo_train_improved": losses[-1] < losses[0],
        "ratio_bad_over_good": float(err_bad.item() / (err_good.item() + 1e-8)),
        "interpretation": {
            "warp": "Correct geometric depth should photometric-match better than 2x depth.",
            "min_reproj": "Min over sources is implemented as Monodepth2-style.",
            "optim": "Photometric loss decreases when refining depth from bad init.",
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in results if "losses" not in k}, indent=2))
    assert results["err_bad_depth"] > results["err_good_depth"], results
    assert results["photo_train_improved"], results
    print("P3 acceptance: PASSED")


if __name__ == "__main__":
    main()
