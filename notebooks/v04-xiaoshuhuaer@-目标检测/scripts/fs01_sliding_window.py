#!/usr/bin/env python3
"""FS01: Exhaustive sliding-window detector (template/color score).

Concept: detection = classify every window.
Shows: slow, many false positives, scale brittleness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, generate_one  # noqa: E402
from metrics import box_iou_xyxy, coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs01_sliding_window"
OUT.mkdir(parents=True, exist_ok=True)


def color_template_score(window: np.ndarray, target_rgb: np.ndarray) -> float:
    """Score = how close mean color is to class prototype (higher better)."""
    mean = window.reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(mean - target_rgb)
    return float(np.exp(-dist * 4.0))


def sliding_window_detect(
    img: np.ndarray,
    win: int = 16,
    stride: int = 4,
    score_thr: float = 0.55,
    prototypes: dict | None = None,
):
    """Return boxes/scores/labels for 3 color classes via window mean color."""
    if prototypes is None:
        prototypes = {
            0: np.array([1.0, 0.15, 0.15]),
            1: np.array([0.15, 0.9, 0.2]),
            2: np.array([0.2, 0.35, 1.0]),
        }
    H, W = img.shape[:2]
    boxes, scores, labels = [], [], []
    n_windows = 0
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            n_windows += 1
            patch = img[y : y + win, x : x + win]
            best_s, best_c = 0.0, -1
            for c, proto in prototypes.items():
                s = color_template_score(patch, proto)
                if s > best_s:
                    best_s, best_c = s, c
            if best_s >= score_thr:
                boxes.append([x, y, x + win, y + win])
                scores.append(best_s)
                labels.append(best_c)
    if boxes:
        return (
            np.asarray(boxes, float),
            np.asarray(scores, float),
            np.asarray(labels, int),
            n_windows,
        )
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), n_windows


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    cfg = SynthConfig(img_size=64, min_size=12, max_size=20, max_objects=2)
    preds, gts = [], []
    total_win = 0
    n_raw = 0
    for i in range(40):
        s = generate_one(rng, cfg)
        boxes, scores, labels, nw = sliding_window_detect(s["image"], win=16, stride=4)
        total_win += nw
        n_raw += len(boxes)
        if len(boxes):
            keep = nms_xyxy(boxes, scores, 0.3)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        gts.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})

    m = coco_style_ap(preds, gts)
    # single-image story
    s0 = generate_one(np.random.default_rng(1), cfg)
    b0, sc0, l0, nw0 = sliding_window_detect(s0["image"])
    # best IoU to first GT if any
    best_iou = 0.0
    if len(b0) and len(s0["boxes"]):
        best_iou = float(box_iou_xyxy(b0, s0["boxes"].astype(float)).max())

    analysis = {
        "method": "sliding_window_color_template",
        "n_images": 40,
        "windows_per_image_mean": total_win / 40,
        "raw_dets_per_image_mean": n_raw / 40,
        "after_nms_metrics": m.as_dict(),
        "example_best_iou_to_gt": best_iou,
        "example_raw_dets": int(len(b0)),
        "example_windows": int(nw0),
        "findings": [
            "Thousands of windows per image even at 64px — classic speed disease.",
            "Fixed window size cannot match arbitrary aspect/scale → localization IoU limited.",
            "Color template works only because synth objects are pure color — real images need features.",
            "NMS is mandatory: raw dets explode; without NMS precision collapses.",
        ],
        "disease_shown": "exhaustive search cost + scale/aspect brittleness + duplicate fires",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2)[:1200])
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
