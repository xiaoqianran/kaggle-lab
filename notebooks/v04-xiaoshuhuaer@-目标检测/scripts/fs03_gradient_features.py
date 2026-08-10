#!/usr/bin/env python3
"""FS03: Hand-crafted gradient features vs pure color (HOG intuition).

Under brightness jitter, color template collapses; gradient magnitude template is stabler.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import COLORS, SynthConfig, generate_one  # noqa: E402
from fs01_sliding_window import sliding_window_detect  # noqa: E402
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs03_gradient_features"
OUT.mkdir(parents=True, exist_ok=True)


def to_gray(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def gradient_mag(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def sliding_window_gradient(
    img: np.ndarray, win: int = 16, stride: int = 4, score_thr: float = 0.35
):
    """Score windows by mean gradient energy + coarse color vote."""
    H, W = img.shape[:2]
    gm = gradient_mag(img)
    boxes, scores, labels = [], [], []
    n_win = 0
    protos = {k: np.array(v) for k, v in COLORS.items()}
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            n_win += 1
            patch = img[y : y + win, x : x + win]
            gpatch = gm[y : y + win, x : x + win]
            # high gradient relative to background
            gscore = float(np.clip(gpatch.mean() / 0.15, 0, 1))
            mean = patch.reshape(-1, 3).mean(0)
            best_c, best_cs = 0, -1.0
            for c, proto in protos.items():
                cs = float(np.exp(-np.linalg.norm(mean - proto) * 3.0))
                if cs > best_cs:
                    best_cs, best_c = cs, c
            s = 0.55 * gscore + 0.45 * best_cs
            if s >= score_thr:
                boxes.append([x, y, x + win, y + win])
                scores.append(s)
                labels.append(best_c)
    if boxes:
        return np.asarray(boxes, float), np.asarray(scores, float), np.asarray(labels, int), n_win
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), n_win


def brightness_jitter(img: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(img * scale, 0, 1).astype(np.float32)


def eval_detector(fn, n=35, bright_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    cfg = SynthConfig(img_size=64, min_size=12, max_size=22, max_objects=2)
    preds, gts = [], []
    for i in range(n):
        s = generate_one(rng, cfg)
        img = brightness_jitter(s["image"], bright_scale)
        b, sc, lab, _ = fn(img)
        if len(b):
            keep = nms_xyxy(b, sc, 0.3)
            b, sc, lab = b[keep], sc[keep], lab[keep]
        preds.append({"boxes": b, "scores": sc, "labels": lab})
        gts.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})
    return coco_style_ap(preds, gts)


def main() -> None:
    set_seed(0)
    rows = []
    for name, fn in [("color_template", sliding_window_detect), ("grad+color", sliding_window_gradient)]:
        for br, scale in [("bright_1.0", 1.0), ("bright_0.45", 0.45), ("bright_1.6", 1.6)]:
            m = eval_detector(fn, bright_scale=scale, seed=1)
            rows.append({"detector": name, "brightness": br, "ap50": m.ap50, "ap": m.ap})
            print(f"{name:14s} {br:12s} AP50={m.ap50:.3f}")

    # bars at dark condition
    dark = {r["detector"]: r["ap50"] for r in rows if r["brightness"] == "bright_0.45"}
    save_bar_compare(
        OUT / "ap50_under_dark.png",
        list(dark.keys()),
        list(dark.values()),
        "FS03 · AP50 at brightness×0.45",
    )

    # visual
    rng = np.random.default_rng(2)
    s = generate_one(rng, SynthConfig(img_size=64, min_size=14, max_size=20))
    dark_img = brightness_jitter(s["image"], 0.45)
    b1, sc1, l1, _ = sliding_window_detect(dark_img)
    b2, sc2, l2, _ = sliding_window_gradient(dark_img)
    if len(b1):
        k = nms_xyxy(b1, sc1, 0.3)
        b1, sc1, l1 = b1[k], sc1[k], l1[k]
    if len(b2):
        k = nms_xyxy(b2, sc2, 0.3)
        b2, sc2, l2 = b2[k], sc2[k], l2[k]
    save_det_panel(OUT / "dark_color.png", dark_img, s["boxes"], s["labels"], b1, sc1, l1, "FS03 color @ dark")
    save_det_panel(OUT / "dark_grad.png", dark_img, s["boxes"], s["labels"], b2, sc2, l2, "FS03 grad+color @ dark")

    analysis = {
        "step": "FS03",
        "concept": "Hand-crafted features (gradient) more lighting-robust than raw RGB mean",
        "rows": rows,
        "new_capability": "Feature engineering before deep learning era",
        "vs_previous": "FS01 pure color template fails under brightness shift; gradient helps",
        "artifacts": ["ap50_under_dark.png", "dark_color.png", "dark_grad.png"],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
