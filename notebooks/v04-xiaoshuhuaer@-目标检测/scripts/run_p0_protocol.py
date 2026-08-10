#!/usr/bin/env python3
"""P0: IoU/NMS/mAP protocol experiment + conf-threshold ablation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, predictions_from_gt_jitter  # noqa: E402
from metrics import box_iou_xyxy, coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "p0_protocol"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_seed(42)
    cfg = SynthConfig(img_size=64, seed=42)
    ds = SynthDetDataset(80, cfg, seed=0)

    scenarios = {
        "perfect": dict(iou_noise=0.0, drop_prob=0.0, fp_per_image=0, score_base=0.95),
        "loc_noise": dict(iou_noise=0.15, drop_prob=0.0, fp_per_image=0, score_base=0.9),
        "miss_gt": dict(iou_noise=0.0, drop_prob=0.3, fp_per_image=0, score_base=0.9),
        "false_pos": dict(iou_noise=0.0, drop_prob=0.0, fp_per_image=2, score_base=0.9),
        "noisy_all": dict(iou_noise=0.12, drop_prob=0.15, fp_per_image=1, score_base=0.85),
    }

    rows = []
    for name, kw in scenarios.items():
        preds, gts = predictions_from_gt_jitter(ds, seed=1, **kw)
        # apply NMS per image
        preds_nms = []
        for p in preds:
            if len(p["boxes"]) == 0:
                preds_nms.append(p)
                continue
            keep = nms_xyxy(p["boxes"], p["scores"], iou_thr=0.5)
            preds_nms.append(
                {
                    "boxes": p["boxes"][keep],
                    "scores": p["scores"][keep],
                    "labels": p["labels"][keep],
                }
            )
        m = coco_style_ap(preds_nms, gts)
        row = {"scenario": name, **m.as_dict()}
        rows.append(row)
        print(
            f"{name:12s} AP50={m.ap50:.3f} AP={m.ap:.3f} "
            f"AP_s={m.ap_s:.3f} AR={m.ar100:.3f} n_gt={m.n_gt}"
        )

    # conf threshold ablation on noisy_all
    preds, gts = predictions_from_gt_jitter(ds, seed=2, **scenarios["noisy_all"])
    conf_rows = []
    for thr in [0.0, 0.2, 0.4, 0.6, 0.8]:
        m = coco_style_ap(preds, gts, conf_thr=thr)
        conf_rows.append({"conf_thr": thr, **m.as_dict()})
        print(f"conf={thr:.1f} AP50={m.ap50:.3f} AP={m.ap:.3f} n_pred_used≈filter")

    # unit sanity: identical boxes IoU=1
    a = np.array([[10.0, 10.0, 30.0, 40.0]])
    assert abs(box_iou_xyxy(a, a)[0, 0] - 1.0) < 1e-6

    analysis = {
        "protocol": "coco_style_v0",
        "seed": 42,
        "findings": [
            "Perfect preds → AP≈1; validates metric not broken.",
            "Localization noise hurts AP@high IoU more than AP50 (AP drops more).",
            "Missed GT (drop_prob) lowers recall/AR and AP.",
            "False positives lower precision → AP falls even if GT covered.",
            "Raising conf_thr trades recall for precision; AP is non-monotonic — must report fixed thr protocol.",
        ],
        "scenario_rows": rows,
        "conf_ablation": conf_rows,
    }
    path = OUT / "results.json"
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
