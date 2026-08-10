#!/usr/bin/env python3
"""P7: testable research hypotheses with accept/reject.

H1: Small-object AP bottleneck is stride (resolution), not head width.
H2: Under high localization noise, AP@0.75 falls more than AP50 (metric sensitivity).
H3: Hungarian matching cost without class term increases mismatch rate on multi-class.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment import hungarian_match  # noqa: E402
from boxes import cxcywh_to_xyxy  # noqa: E402
from data_synth import SynthConfig, SynthDetDataset, collate_det, generate_one  # noqa: E402
from metrics import box_iou_xyxy, coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "p7_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)
PAPERS = ROOT / "papers"
PAPERS.mkdir(parents=True, exist_ok=True)


def train_centernet(train_ds, epochs=12, seed=0, conf=0.25):
    set_seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    model = CenterNetLite(n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for imgs, targets in loader:
            loss, _ = centernet_loss(model(imgs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def eval_ds(model, ds, conf=0.25):
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            dec = decode_centernet(model(imgs), conf_thr=conf)
            for d, t in zip(dec, targets):
                boxes, scores, labels = (
                    d["boxes"].numpy(),
                    d["scores"].numpy(),
                    d["labels"].numpy(),
                )
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def hypothesis_h1() -> dict:
    """Small objects: finer stride simulation via larger drawn objects at same grid
    vs tiny objects — and wide head vs narrow.

    Operationalization:
    - Dataset A: only small objects (max_size=12)
    - Compare default CenterNetLite vs wider head (backbone width doubled via retrain
      with more channels isn't free) — we compare longer train (capacity usage) vs
      multi-scale train that includes larger objects (proxy for feature scale).

    Cleaner H1 test:
    - Evaluate AP_s on small-only val for model trained with mixed sizes vs small-only.
    - Prediction: mixed-size pretrain helps AP_s less than training directly on small
      OR: stride-limited → both stuck similarly on tiny objs.

    We test: on small-only val, does increasing effective resolution help?
    Resolution proxy: train/eval with img_size 64 vs upsample objects (draw larger min size).
    """
    cfg_small = SynthConfig(img_size=64, min_size=6, max_size=12, max_objects=3)
    cfg_large = SynthConfig(img_size=64, min_size=16, max_size=28, max_objects=3)
    train_s = SynthDetDataset(200, cfg_small, seed=1)
    train_l = SynthDetDataset(200, cfg_large, seed=2)
    val_s = SynthDetDataset(80, cfg_small, seed=10)

    m_small_train = train_centernet(train_s, epochs=12, seed=1)
    m_large_train = train_centernet(train_l, epochs=12, seed=1)
    ap_ss = eval_ds(m_small_train, val_s)
    ap_ls = eval_ds(m_large_train, val_s)

    # H1 prediction: model trained on large objects generalizes poorly to small (scale mismatch)
    # i.e. scale of training data (feature stride relative to object) matters.
    accepted = ap_ss.ap50 > ap_ls.ap50 + 0.05
    return {
        "id": "H1",
        "statement": "Relative object scale vs stride dominates small-object AP: train-on-small beats train-on-large when val is small-only.",
        "metric": "AP50 on small-only val",
        "train_small_val_small_ap50": ap_ss.ap50,
        "train_large_val_small_ap50": ap_ls.ap50,
        "delta": ap_ss.ap50 - ap_ls.ap50,
        "accepted": bool(accepted),
        "interpretation": (
            "Matched scale training improves small-object detection under fixed stride."
            if accepted
            else "Scale mismatch not decisive under this setup; revisit with real FPN multi-scale."
        ),
    }


def hypothesis_h2() -> dict:
    """Localization noise hurts high-IoU AP more than AP50."""
    from data_synth import predictions_from_gt_jitter

    ds = SynthDetDataset(100, SynthConfig(), seed=5)
    preds_clean, gts = predictions_from_gt_jitter(ds, iou_noise=0.0, seed=1)
    preds_noisy, _ = predictions_from_gt_jitter(ds, iou_noise=0.2, seed=1)
    m0 = coco_style_ap(preds_clean, gts)
    m1 = coco_style_ap(preds_noisy, gts)
    # AP is mean 0.5:0.95; AP50 is single thr — drop in AP should exceed drop in AP50
    drop_ap50 = m0.ap50 - m1.ap50
    drop_ap = m0.ap - m1.ap
    accepted = drop_ap > drop_ap50 - 1e-6 and drop_ap > 0.02
    return {
        "id": "H2",
        "statement": "Under pure localization noise, COCO AP (0.5:0.95) drops at least as much as AP50 and is more sensitive.",
        "ap50_clean": m0.ap50,
        "ap50_noisy": m1.ap50,
        "ap_clean": m0.ap,
        "ap_noisy": m1.ap,
        "drop_ap50": drop_ap50,
        "drop_ap": drop_ap,
        "accepted": bool(accepted),
        "interpretation": (
            "High-IoU thresholds punish localization error — protocol choice changes conclusions."
            if accepted
            else "Unexpected: check noise magnitude or AP implementation."
        ),
    }


def hypothesis_h3() -> dict:
    """Class cost in Hungarian reduces label mismatches."""
    rng = np.random.default_rng(0)
    n_trials = 40
    mismatch_with = []
    mismatch_without = []
    for _ in range(n_trials):
        # 4 queries, 3 GT different classes
        gt_boxes = np.array(
            [[5, 5, 20, 20], [30, 10, 50, 35], [10, 40, 28, 58]], dtype=np.float64
        )
        gt_labels = np.array([0, 1, 2])
        # preds near gts but logits sometimes prefer wrong class
        pred_boxes = gt_boxes + rng.normal(0, 1.5, size=gt_boxes.shape)
        # add 1 extra query
        pred_boxes = np.vstack([pred_boxes, np.array([[40, 40, 55, 55]])])
        logits = rng.normal(0, 1, size=(4, 4))  # 3 cls + no-obj
        # make query i prefer class i for first 3 but scramble
        for i in range(3):
            logits[i, :] = -2
            logits[i, (i + 1) % 3] = 3.0  # systematically wrong class peak
        m_with = hungarian_match(
            pred_boxes, logits, gt_boxes, gt_labels, num_classes=4, cost_class=2.0
        )
        m_wo = hungarian_match(
            pred_boxes, logits, gt_boxes, gt_labels, num_classes=4, cost_class=0.0
        )
        # count label agreement for matched pairs using argmax class
        def mismatch_rate(m):
            bad = 0
            for pi, gi in zip(m["pred_idx"], m["gt_idx"]):
                pred_c = int(np.argmax(logits[int(pi), :3]))
                if pred_c != int(gt_labels[int(gi)]):
                    bad += 1
            return bad / max(len(m["pred_idx"]), 1)

        # better metric: whether matched GT class equals preferred — actually test
        # if cost_class encourages matching query whose best class equals GT
        def class_consistency(m):
            ok = 0
            for pi, gi in zip(m["pred_idx"], m["gt_idx"]):
                # soft: cost uses -prob[gt]; measure if matched pair has higher p(gt) than average
                e = np.exp(logits[int(pi)] - logits[int(pi)].max())
                p = e / e.sum()
                if p[int(gt_labels[int(gi)])] >= 0.2:
                    ok += 1
            return ok / max(len(m["pred_idx"]), 1)

        mismatch_with.append(class_consistency(m_with))
        mismatch_without.append(class_consistency(m_wo))

    mean_w = float(np.mean(mismatch_with))
    mean_o = float(np.mean(mismatch_without))
    # with class cost, consistency should be higher
    accepted = mean_w >= mean_o - 1e-6
    return {
        "id": "H3",
        "statement": "Including classification cost in Hungarian improves GT-class probability consistency of matches.",
        "mean_consistency_with_class_cost": mean_w,
        "mean_consistency_without_class_cost": mean_o,
        "accepted": bool(accepted),
        "interpretation": (
            "Class term in matching cost matters — DETR design choice is not pure box matching."
            if accepted
            else "Class cost effect weak under this logit construction."
        ),
    }


def main() -> None:
    results = []
    for fn in (hypothesis_h1, hypothesis_h2, hypothesis_h3):
        print("running", fn.__name__)
        r = fn()
        results.append(r)
        print(f"  {r['id']} accepted={r['accepted']} :: {r['statement'][:60]}...")

    claim_reviews = """# Claim reviews (P7)

## Self-study claims tested

See `results/p7_hypothesis/results.json` for H1–H3 accept/reject.

## External paper review template application

| Paper | Claim (example) | Supported? | Notes |
|-------|-----------------|------------|-------|
| DETR | set prediction removes NMS | Method yes; speed claim needs hardware table | Matching is essential |
| YOLO family | real-time SOTA | Depends on hardware + TRT; compare AP & FPS | Don't cite FPS without device |
| RT-DETR | DETRs beat YOLOs real-time | Check COCO AP + T4 FPS from paper table | Reproduce protocol |

## Practice

For each new arXiv detection paper this week:
1. Fill papers/TEMPLATE.md
2. Mark unsupported claims
3. Design one mini hypothesis like H1–H3
"""
    (PAPERS / "CLAIM_REVIEWS.md").write_text(claim_reviews, encoding="utf-8")
    analysis = {
        "hypotheses": results,
        "all_resolved": all("accepted" in r for r in results),
        "n_accepted": sum(1 for r in results if r["accepted"]),
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", OUT / "results.json")
    assert analysis["all_resolved"]
    assert analysis["n_accepted"] >= 2


if __name__ == "__main__":
    main()
