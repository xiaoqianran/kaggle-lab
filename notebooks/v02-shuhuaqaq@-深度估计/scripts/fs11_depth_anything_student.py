#!/usr/bin/env python3
"""FS11: foundation-model *recipe* mini — teacher pseudo-labels train a student.

If DA-V2 weights unavailable, use MiDaS_small as teacher (same industrial idea:
large/strong teacher → compact student on unlabeled images).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from models import TinyUNet  # noqa: E402
from synth_data import make_batch, make_scene  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs11_pseudo_student"
OUT.mkdir(parents=True, exist_ok=True)


def load_teacher():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, transforms.small_transform


def teacher_pred(model, transform, rgb01: np.ndarray) -> np.ndarray:
    img = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    sample = transform(img)
    inp = sample["image"] if isinstance(sample, dict) else sample
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=rgb01.shape[:2], mode="bilinear", align_corners=False)
    return pred[0, 0].cpu().numpy()


def main() -> None:
    teacher, transform = load_teacher()
    # unlabeled pool: easy synthetic RGB only
    student = TinyUNet(base=16, max_depth=1.0)  # predict normalized relative then scale
    # Actually student predicts positive map; we'll match teacher up to affine via SILog on normalized
    opt = torch.optim.Adam(student.parameters(), lr=2e-3)

    losses = []
    for step in range(60):
        # diverse seeds as "unlabeled internet images" stand-in
        rgbs = []
        targets = []
        for j in range(4):
            s = make_scene(64, 80, seed=step * 4 + j, easy=(j % 2 == 0))
            t = teacher_pred(teacher, transform, s["rgb"])
            # normalize teacher to 0-1 for stable regression
            t = (t - t.min()) / (t.max() - t.min() + 1e-6)
            rgbs.append(s["rgb"].transpose(2, 0, 1))
            targets.append(t[None, ...])
        rgb = torch.from_numpy(np.stack(rgbs).astype(np.float32))
        tgt = torch.from_numpy(np.stack(targets).astype(np.float32))
        pred = student(rgb)
        # normalize pred per-image
        b = pred.shape[0]
        pred_n = []
        for i in range(b):
            p = pred[i]
            p = (p - p.amin()) / (p.amax() - p.amin() + 1e-6)
            pred_n.append(p)
        pred_n = torch.stack(pred_n)
        loss = F.l1_loss(pred_n, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # evaluate student vs teacher agreement on holdout + vs GT with LS
    student.eval()
    agree, vs_gt = [], []
    panels = []
    with torch.no_grad():
        for i in range(6):
            s = make_scene(64, 80, seed=9000 + i, easy=True)
            t = teacher_pred(teacher, transform, s["rgb"])
            t_n = (t - t.min()) / (t.max() - t.min() + 1e-6)
            rgb = torch.from_numpy(s["rgb"].transpose(2, 0, 1)[None].astype(np.float32))
            p = student(rgb).squeeze().numpy()
            p_n = (p - p.min()) / (p.max() - p.min() + 1e-6)
            agree.append(float(np.mean(np.abs(p_n - t_n))))
            m = compute_metrics(p_n, s["depth"], max_depth=20.0, align="least_squares")
            vs_gt.append(m.abs_rel)
            if i < 2:
                panels += [("rgb", s["rgb"]), ("teacher", t_n), ("student", p_n)]

    save_panels(OUT / "teacher_student.png", panels, ncols=3)
    # baseline: student before would be worse — compare to constant pred
    const = np.full_like(s["depth"], 0.5)
    m_const = compute_metrics(const, s["depth"], max_depth=20.0, align="least_squares")

    out = {
        "teacher": "MiDaS_small (stand-in for DA-V2 teacher when DA weights unavailable)",
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "mean_l1_to_teacher": float(np.mean(agree)),
        "student_ls_abs_rel_vs_gt": float(np.mean(vs_gt)),
        "constant_ls_abs_rel": m_const.abs_rel,
        "new_capability": "Distill relative depth from a strong teacher without GT depth labels.",
        "compare_to_fs10": "FS10 only runs teacher; FS11 trains a student on pseudo-labels (Depth Anything idea).",
        "compare_to_fs06": "FS06 needs GT depth; FS11 needs only RGB + teacher.",
        "artifacts": ["teacher_student.png"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert losses[-1] < losses[0] * 0.85
    assert float(np.mean(vs_gt)) < m_const.abs_rel * 0.9
    print("FS11 acceptance: PASSED")


if __name__ == "__main__":
    main()
