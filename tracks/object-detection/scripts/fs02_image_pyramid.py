#!/usr/bin/env python3
"""FS02: Multi-scale image pyramid sliding window vs single-scale (FS01).

Concept: scale is a first-class axis of search.
Compare AP and window count: single scale vs pyramid.
Honest lesson: naive pyramid can add FPs without score calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, generate_one  # noqa: E402
from fs01_sliding_window import sliding_window_detect  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs02_image_pyramid"
OUT.mkdir(parents=True, exist_ok=True)


def pyramid_detect(img: np.ndarray, scales=(0.75, 1.0, 1.35), win=16, stride=4):
    H, W = img.shape[:2]
    all_b, all_s, all_l = [], [], []
    n_win = 0
    for sc in scales:
        nh, nw = max(win, int(H * sc)), max(win, int(W * sc))
        ys = (np.linspace(0, H - 1, nh)).astype(int)
        xs = (np.linspace(0, W - 1, nw)).astype(int)
        small = img[ys][:, xs]
        boxes, scores, labels, nwins = sliding_window_detect(small, win=win, stride=stride)
        n_win += nwins
        if len(boxes):
            boxes = boxes / sc
            all_b.append(boxes)
            all_s.append(scores)
            all_l.append(labels)
    if all_b:
        b = np.concatenate(all_b)
        s = np.concatenate(all_s)
        l = np.concatenate(all_l)
        keep = nms_xyxy(b, s, 0.3)
        return b[keep], s[keep], l[keep], n_win
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), n_win


def eval_method(method: str, n=40):
    rng = np.random.default_rng(0)
    cfg = SynthConfig(img_size=64, min_size=8, max_size=28, max_objects=2)
    preds, gts = [], []
    wins = 0
    raw = 0
    for i in range(n):
        s = generate_one(rng, cfg)
        if method == "single":
            b, sc, lab, nw = sliding_window_detect(s["image"], win=16, stride=4)
            raw += len(b)
            if len(b):
                keep = nms_xyxy(b, sc, 0.3)
                b, sc, lab = b[keep], sc[keep], lab[keep]
            wins += nw
        else:
            b, sc, lab, nw = pyramid_detect(s["image"])
            # raw already nms'ed inside for pyramid; count windows only
            wins += nw
            raw += len(b)
        preds.append({"boxes": b, "scores": sc, "labels": lab})
        gts.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})
    m = coco_style_ap(preds, gts)
    return m, wins / n, raw / n


def main() -> None:
    set_seed(0)
    m1, w1, r1 = eval_method("single")
    m2, w2, r2 = eval_method("pyramid")
    analysis = {
        "single_scale": {**m1.as_dict(), "windows_per_image": w1, "dets_after_nms_mean": r1},
        "pyramid": {**m2.as_dict(), "windows_per_image": w2, "dets_after_nms_mean": r2},
        "delta_ap50": m2.ap50 - m1.ap50,
        "window_cost_ratio": w2 / max(w1, 1e-9),
        "findings": [
            "Pyramid multiplies compute (~n_scales) — scale coverage is bought with FLOPs.",
            "Naive fusion without score calibration / size-aware NMS can HURT AP by adding FPs.",
            "Historical lesson: multi-scale search is necessary but needs careful post-process "
            "(later solved better by feature pyramids, not image pyramids alone).",
            "Compare cost_ratio vs AP delta — if AP drops while cost rises, you felt the classic footgun.",
        ],
        "disease_targeted": "scale brittleness of fixed window",
        "disease_remaining_or_side_effect": "FP inflation + cost; motivates FPN (FS10) over image pyramid",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(
        f"single AP50={m1.ap50:.3f} wins={w1:.0f} | "
        f"pyramid AP50={m2.ap50:.3f} wins={w2:.0f} Δ={analysis['delta_ap50']:.3f}"
    )
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
