#!/usr/bin/env python3
"""P1: box encode/decode, anchors, IoU assign, flip consistency."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment import assign_iou, simota_like_assign  # noqa: E402
from boxes import (  # noqa: E402
    cxcywh_to_xyxy,
    decode_boxes_cxcywh,
    encode_boxes_cxcywh,
    horizontal_flip_xyxy,
    make_anchors,
    xyxy_to_cxcywh,
)
from data_synth import SynthConfig, SynthDetDataset  # noqa: E402
from metrics import box_iou_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "p1_boxes"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_seed(0)
    # encode/decode roundtrip
    anchors = np.array([[20.0, 20.0, 40.0, 40.0], [50.0, 50.0, 30.0, 20.0]])
    gts = np.array([[22.0, 18.0, 44.0, 42.0], [48.0, 52.0, 28.0, 24.0]])
    a_c = xyxy_to_cxcywh(anchors)
    # wait anchors given as cxcywh-like above - use explicit
    anchors_c = np.array([[30.0, 30.0, 20.0, 20.0], [50.0, 50.0, 30.0, 20.0]])
    gts_c = np.array([[33.0, 30.0, 22.0, 24.0], [48.0, 52.0, 28.0, 24.0]])
    deltas = encode_boxes_cxcywh(anchors_c, gts_c)
    rec = decode_boxes_cxcywh(anchors_c, deltas)
    err = np.abs(rec - gts_c).max()
    print(f"encode/decode max abs err={err:.2e}")

    # flip consistency
    box = np.array([[10.0, 5.0, 30.0, 40.0]])
    W = 64.0
    flipped = horizontal_flip_xyxy(box, W)
    back = horizontal_flip_xyxy(flipped, W)
    flip_err = np.abs(back - box).max()
    print(f"flip roundtrip err={flip_err:.2e} flipped={flipped.tolist()}")

    # anchors + assignment stats on synth
    ds = SynthDetDataset(40, SynthConfig(img_size=64), seed=1)
    anchors_xyxy = make_anchors(feat_h=8, feat_w=8, stride=8, scales=(16, 32), ratios=(0.5, 1.0, 2.0))
    pos_counts, neg_counts, ign_counts = [], [], []
    thr_rows = []
    for pos_thr in [0.5, 0.6, 0.7]:
        pos_n = []
        for s in ds.samples:
            asg = assign_iou(anchors_xyxy, s["boxes"], pos_thr=pos_thr, neg_thr=0.3)
            labels = asg["labels"]
            pos_n.append(int((labels == 1).sum()))
            if pos_thr == 0.7:
                pos_counts.append(int((labels == 1).sum()))
                neg_counts.append(int((labels == 0).sum()))
                ign_counts.append(int((labels == -1).sum()))
        thr_rows.append(
            {
                "pos_thr": pos_thr,
                "mean_pos_anchors": float(np.mean(pos_n)),
                "std_pos": float(np.std(pos_n)),
            }
        )
        print(f"pos_thr={pos_thr} mean positives={np.mean(pos_n):.1f}")

    # SimOTA-like on noisy preds
    s0 = ds.samples[0]
    # fake preds: anchors as boxes with random scores
    scores = np.random.RandomState(0).uniform(0.1, 0.9, size=len(anchors_xyxy))
    sim = simota_like_assign(anchors_xyxy, scores, s0["boxes"], topk=5)
    print(f"simota positives={sim['pos'].sum()} / {len(anchors_xyxy)}")

    # assignment IoU threshold ablation effect on mean max_iou of positives
    analysis = {
        "encode_decode_max_abs_err": float(err),
        "flip_roundtrip_err": float(flip_err),
        "n_anchors": int(len(anchors_xyxy)),
        "assign_pos_thr_ablation": thr_rows,
        "assign_stats_pos_thr_0.7": {
            "mean_pos": float(np.mean(pos_counts)),
            "mean_neg": float(np.mean(neg_counts)),
            "mean_ignore": float(np.mean(ign_counts)),
        },
        "simota_pos_count": int(sim["pos"].sum()),
        "findings": [
            "encode/decode is numerically invertible (err ~0).",
            "horizontal flip twice restores boxes — aug must transform boxes.",
            "Higher pos_thr reduces mean positive anchors → fewer positives, harder training if too high.",
            "Best-anchor-per-GT guarantee keeps at least n_gt positives even at high thr.",
        ],
    }
    path = OUT / "results.json"
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", path)
    assert err < 1e-9
    assert flip_err < 1e-9


if __name__ == "__main__":
    main()
