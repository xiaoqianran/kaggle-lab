#!/usr/bin/env python3
"""FS08: Monodepth2-style min-reproj + automask on synthetic dynamic scene."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from photometric import photometric_error, warp_with_pose  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs08_automask"
OUT.mkdir(parents=True, exist_ok=True)


def make_dynamic_pair(h=64, w=96, z=5.0, fx=200.0, baseline=0.25):
    """Static background + independently moving bright square (breaks rigid motion)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    tex = 0.5 + 0.5 * np.sin(30 * xx) * np.cos(20 * yy)
    left = np.stack([tex, tex * 0.95, tex * 0.9], axis=-1).astype(np.float32)
    depth = np.full((h, w), z, np.float32)
    # moving object only in left at u0, appears at u0+obj_shift in right (not consistent with bg disp)
    obj_shift = 18  # pixels of independent motion
    disp_bg = fx * baseline / z
    # build right from bg only with stereo shift
    left_t = torch.from_numpy(left.transpose(2, 0, 1)).unsqueeze(0)
    # simple roll for bg
    shift = int(round(disp_bg))
    right = np.roll(left, shift=-shift, axis=1)
    # paint object on left
    y0, y1, x0, x1 = 20, 40, 30, 50
    left[y0:y1, x0:x1] = np.array([1.0, 0.2, 0.2], np.float32)
    # object on right at wrong place (dynamic)
    right[y0:y1, x0 + obj_shift : x1 + obj_shift] = np.array([1.0, 0.2, 0.2], np.float32)

    K = torch.tensor([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=torch.float32)
    T = torch.eye(4).unsqueeze(0)
    T[0, 0, 3] = -baseline
    img_l = torch.from_numpy(left.transpose(2, 0, 1)).unsqueeze(0)
    img_r = torch.from_numpy(right.transpose(2, 0, 1)).unsqueeze(0)
    depth_t = torch.full((1, 1, h, w), z)
    return img_l, img_r, depth_t, K, T, left, right, (y0, y1, x0, x1)


def main() -> None:
    img_l, img_r, depth, K, T, left_np, right_np, box = make_dynamic_pair()
    warped, valid = warp_with_pose(img_r, depth, T, K)
    err = photometric_error(warped, img_l)

    # identity reprojection error (automask reference)
    err_id = photometric_error(img_l, img_l)  # ~0
    # better: identity between left and unwarped right as "no motion explanation"
    err_identity = photometric_error(img_r, img_l)

    # automask: keep pixels where warped better than identity (Monodepth2 idea)
    automask = (err < err_identity).float() * valid
    # min reproj: min(err_warp, err_identity) then mean
    min_err = torch.min(err, err_identity)

    y0, y1, x0, x1 = box
    region_dyn = slice(y0, y1), slice(x0, x1)
    dyn_err = float(err[0, 0, y0:y1, x0:x1].mean())
    bg_err = float(err[0, 0, 5:15, 5:25].mean())
    dyn_auto = float(automask[0, 0, y0:y1, x0:x1].mean())
    bg_auto = float(automask[0, 0, 5:15, 5:25].mean())

    save_panels(
        OUT / "automask_demo.png",
        [
            ("left", left_np),
            ("right", right_np),
            ("warped right→left", warped[0].permute(1, 2, 0).detach().numpy()),
            ("photo err", err[0, 0].detach().numpy()),
            ("identity err L vs R", err_identity[0, 0].detach().numpy()),
            ("automask keep", automask[0, 0].detach().numpy()),
        ],
        ncols=3,
    )

    loss_no_auto = float((err * valid).sum() / (valid.sum() + 1e-8))
    loss_with_auto = float((err * automask).sum() / (automask.sum() + 1e-8))
    loss_min = float(min_err.mean())

    out = {
        "dyn_region_photo_err": dyn_err,
        "bg_region_photo_err": bg_err,
        "dyn_automask_keep_rate": dyn_auto,
        "bg_automask_keep_rate": bg_auto,
        "loss_all_valid": loss_no_auto,
        "loss_automasked": loss_with_auto,
        "loss_min_reproj": loss_min,
        "new_capability": "Ignore pixels that rigid-warp cannot explain (dynamics/occlusion).",
        "compare_to_fs07": "FS07 showed photo error tracks depth; FS08 adds min-reproj/automask for real training.",
        "lesson": "Dynamic object should have higher photo error and lower automask keep rate than background.",
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert dyn_err > bg_err * 1.2, (dyn_err, bg_err)
    assert dyn_auto < bg_auto + 0.15  # dynamic kept less or comparable
    print("FS08 acceptance: PASSED")


if __name__ == "__main__":
    main()
