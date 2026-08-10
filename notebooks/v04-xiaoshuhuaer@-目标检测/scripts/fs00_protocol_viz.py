#!/usr/bin/env python3
"""FS00: protocol experiment + visual board (wraps metrics scenarios)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, predictions_from_gt_jitter  # noqa: E402
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs00_protocol"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_seed(42)
    cfg = SynthConfig(img_size=64, seed=42)
    ds = SynthDetDataset(40, cfg, seed=0)
    scenarios = {
        "perfect": dict(iou_noise=0.0, drop_prob=0.0, fp_per_image=0, score_base=0.95),
        "loc_noise": dict(iou_noise=0.15, drop_prob=0.0, fp_per_image=0, score_base=0.9),
        "miss_gt": dict(iou_noise=0.0, drop_prob=0.35, fp_per_image=0, score_base=0.9),
        "false_pos": dict(iou_noise=0.0, drop_prob=0.0, fp_per_image=3, score_base=0.9),
    }
    rows = []
    labels, ap50s = [], []
    for name, kw in scenarios.items():
        preds, gts = predictions_from_gt_jitter(ds, seed=1, **kw)
        preds_n = []
        for p in preds:
            if len(p["boxes"]) == 0:
                preds_n.append(p)
                continue
            keep = nms_xyxy(p["boxes"], p["scores"], 0.5)
            preds_n.append(
                {
                    "boxes": p["boxes"][keep],
                    "scores": p["scores"][keep],
                    "labels": p["labels"][keep],
                }
            )
        m = coco_style_ap(preds_n, gts)
        rows.append({"scenario": name, **m.as_dict()})
        labels.append(name)
        ap50s.append(m.ap50)
        print(f"{name:12s} AP50={m.ap50:.3f} AP={m.ap:.3f}")

    save_bar_compare(OUT / "ap50_by_scenario.png", labels, ap50s, "FS00 · AP50 under error modes")

    # panel: loc noise example
    s0 = ds.samples[0]
    preds, _ = predictions_from_gt_jitter(
        type("D", (), {"samples": [s0], "cfg": cfg})(),
        seed=0,
        iou_noise=0.2,
        score_base=0.9,
    )
    p0 = preds[0]
    save_det_panel(
        OUT / "example_loc_noise.png",
        s0["image"],
        gt_boxes=s0["boxes"],
        gt_labels=s0["labels"],
        pred_boxes=p0["boxes"],
        pred_scores=p0["scores"],
        pred_labels=p0["labels"],
        title="FS00 loc_noise: green=GT dashed, color=pred",
    )

    analysis = {
        "step": "FS00",
        "concept": "IoU/NMS/mAP protocol; conf is part of protocol",
        "scenario_rows": rows,
        "new_capability": "Measure detectors fairly; see how error types move AP",
        "vs_previous": "Starting point — no detector yet",
        "artifacts": ["ap50_by_scenario.png", "example_loc_noise.png", "results.json"],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
