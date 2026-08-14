#!/usr/bin/env python3
"""FS04: Region proposals vs exhaustive windows — count & recall."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, generate_one  # noqa: E402
from fs01_sliding_window import sliding_window_detect  # noqa: E402
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import box_iou_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs04_proposals"
OUT.mkdir(parents=True, exist_ok=True)


def make_proposals(img, gt_boxes, rng, n_random=15, n_jitter=3, include_gt=True):
    H, W = img.shape[:2]
    props = []
    for b in gt_boxes:
        for _ in range(n_jitter):
            bb = b.astype(float).copy()
            cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            cx += rng.normal(0, 0.1 * w)
            cy += rng.normal(0, 0.1 * h)
            w *= 1 + abs(rng.normal(0, 0.12))
            h *= 1 + abs(rng.normal(0, 0.12))
            props.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        if include_gt:
            props.append(b.astype(float).tolist())
    for _ in range(n_random):
        w, h = int(rng.integers(8, 20)), int(rng.integers(8, 20))
        x1, y1 = int(rng.integers(0, max(1, W - w))), int(rng.integers(0, max(1, H - h)))
        props.append([x1, y1, x1 + w, y1 + h])
    return np.asarray(props, float) if props else np.zeros((0, 4))


def proposal_recall(props, gts, iou_thr=0.5):
    if len(gts) == 0:
        return 1.0
    if len(props) == 0:
        return 0.0
    ious = box_iou_xyxy(props, gts)
    hit = 0
    for j in range(len(gts)):
        if ious[:, j].max() >= iou_thr:
            hit += 1
    return hit / len(gts)


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    cfg = SynthConfig(img_size=64, min_size=10, max_size=24, max_objects=3)
    n = 50
    recs, n_props, n_wins = [], [], []
    for i in range(n):
        s = generate_one(rng, cfg)
        props = make_proposals(s["image"], s["boxes"], rng, include_gt=True, n_jitter=3)
        recs.append(proposal_recall(props, s["boxes"].astype(float)))
        n_props.append(len(props))
        _, _, _, nw = sliding_window_detect(s["image"], win=16, stride=4)
        n_wins.append(nw)

    rec_bad = []
    for i in range(n):
        s = generate_one(np.random.default_rng(100 + i), cfg)
        # pure random — no GT, no jitter near GT
        props = make_proposals(
            s["image"],
            s["boxes"],
            np.random.default_rng(200 + i),
            n_jitter=0,
            n_random=25,
            include_gt=False,
        )
        rec_bad.append(proposal_recall(props, s["boxes"].astype(float)))

    analysis = {
        "step": "FS04",
        "mean_proposal_count": float(np.mean(n_props)),
        "mean_window_count": float(np.mean(n_wins)),
        "compute_ratio_windows_over_props": float(np.mean(n_wins) / max(np.mean(n_props), 1e-9)),
        "mean_proposal_recall_good": float(np.mean(recs)),
        "mean_proposal_recall_random_only": float(np.mean(rec_bad)),
        "new_capability": "Sparse WHERE candidates replace exhaustive pixel scan",
        "vs_previous": "FS01–02 search hundreds of windows; proposals ~tens with high recall if quality OK",
        "findings": [
            f"Windows/image ≈ {np.mean(n_wins):.0f} vs proposals ≈ {np.mean(n_props):.0f}",
            f"Good proposals (GT+jitter) recall@0.5 ≈ {np.mean(recs):.3f}",
            f"Random-only proposals recall ≈ {np.mean(rec_bad):.3f} — quality is destiny",
        ],
    }
    save_bar_compare(
        OUT / "counts.png",
        ["windows", "proposals"],
        [analysis["mean_window_count"], analysis["mean_proposal_count"]],
        "FS04 · candidates per image",
        ylabel="count",
    )
    save_bar_compare(
        OUT / "recall.png",
        ["good_props", "random_only"],
        [analysis["mean_proposal_recall_good"], analysis["mean_proposal_recall_random_only"]],
        "FS04 · proposal recall@0.5",
        ylabel="recall",
    )

    s = generate_one(np.random.default_rng(7), cfg)
    props = make_proposals(s["image"], s["boxes"], np.random.default_rng(8))
    save_det_panel(
        OUT / "proposals_vis.png",
        s["image"],
        gt_boxes=s["boxes"],
        gt_labels=s["labels"],
        pred_boxes=props,
        pred_scores=np.ones(len(props)) * 0.5,
        pred_labels=np.zeros(len(props), dtype=int),
        title="FS04 proposals (red) vs GT (green)",
        max_pred=40,
    )

    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2)[:1000])
    print("wrote", OUT)
    assert analysis["mean_proposal_recall_good"] > analysis["mean_proposal_recall_random_only"]


if __name__ == "__main__":
    main()
