#!/usr/bin/env python3
"""FS15: research hypotheses H1–H3 with charts (from run_p7)."""
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
from data_synth import SynthConfig, SynthDetDataset, collate_det, predictions_from_gt_jitter  # noqa: E402
from fs_viz import save_bar_compare  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "fs15_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)


def train_centernet(train_ds, epochs=12, seed=0):
    set_seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    model = CenterNetLite()
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
                boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def main() -> None:
    # H1
    cfg_s = SynthConfig(img_size=64, min_size=6, max_size=12)
    cfg_l = SynthConfig(img_size=64, min_size=16, max_size=28)
    val_s = SynthDetDataset(80, cfg_s, seed=10)
    m_ss = eval_ds(train_centernet(SynthDetDataset(200, cfg_s, seed=1), seed=1), val_s)
    m_ls = eval_ds(train_centernet(SynthDetDataset(200, cfg_l, seed=2), seed=1), val_s)
    h1 = {
        "id": "H1",
        "statement": "Train-on-small beats train-on-large for small-only val (scale vs stride)",
        "train_small_ap50": m_ss.ap50,
        "train_large_ap50": m_ls.ap50,
        "accepted": bool(m_ss.ap50 > m_ls.ap50 + 0.05),
    }
    # H2
    ds = SynthDetDataset(100, SynthConfig(), seed=5)
    p0, g = predictions_from_gt_jitter(ds, iou_noise=0.0, seed=1)
    p1, _ = predictions_from_gt_jitter(ds, iou_noise=0.2, seed=1)
    c0, c1 = coco_style_ap(p0, g), coco_style_ap(p1, g)
    h2 = {
        "id": "H2",
        "statement": "Loc noise: AP drop >= AP50 drop",
        "drop_ap50": c0.ap50 - c1.ap50,
        "drop_ap": c0.ap - c1.ap,
        "accepted": bool((c0.ap - c1.ap) >= (c0.ap50 - c1.ap50) - 1e-6 and (c0.ap - c1.ap) > 0.02),
    }
    # H3
    rng = np.random.default_rng(0)
    cons_w, cons_o = [], []
    for _ in range(40):
        gt = np.array([[5, 5, 20, 20], [30, 10, 50, 35], [10, 40, 28, 58]], float)
        gl = np.array([0, 1, 2])
        pred = gt + rng.normal(0, 1.5, gt.shape)
        pred = np.vstack([pred, [[40, 40, 55, 55]]])
        logits = rng.normal(0, 1, (4, 4))
        for i in range(3):
            logits[i, :] = -2
            logits[i, (i + 1) % 3] = 3.0
        for cost_class, bucket in [(2.0, cons_w), (0.0, cons_o)]:
            m = hungarian_match(pred, logits, gt, gl, 4, cost_class=cost_class)
            ok = 0
            for pi, gi in zip(m["pred_idx"], m["gt_idx"]):
                e = np.exp(logits[int(pi)] - logits[int(pi)].max())
                p = e / e.sum()
                if p[int(gl[int(gi)])] >= 0.2:
                    ok += 1
            bucket.append(ok / max(len(m["pred_idx"]), 1))
    h3 = {
        "id": "H3",
        "statement": "Hungarian class cost improves match class consistency",
        "with_class_cost": float(np.mean(cons_w)),
        "without": float(np.mean(cons_o)),
        "accepted": bool(np.mean(cons_w) >= np.mean(cons_o) - 1e-6),
    }
    results = [h1, h2, h3]
    save_bar_compare(
        OUT / "h1.png",
        ["train_small", "train_large"],
        [h1["train_small_ap50"], h1["train_large_ap50"]],
        "FS15 H1 · AP50 on small val",
    )
    save_bar_compare(
        OUT / "h2.png",
        ["drop_AP50", "drop_AP"],
        [h2["drop_ap50"], h2["drop_ap"]],
        "FS15 H2 · metric sensitivity",
        ylabel="drop",
    )
    save_bar_compare(
        OUT / "h3.png",
        ["with_cls_cost", "no_cls_cost"],
        [h3["with_class_cost"], h3["without"]],
        "FS15 H3 · match consistency",
        ylabel="consistency",
    )
    analysis = {
        "step": "FS15",
        "concept": "Falsifiable hypotheses + claim discipline",
        "hypotheses": results,
        "n_accepted": sum(1 for r in results if r["accepted"]),
        "new_capability": "Research loop: predict → measure → accept/reject",
        "vs_previous": "FS00–14 build systems; FS15 judges contributions",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    # claim file
    (ROOT / "papers" / "CLAIM_REVIEWS.md").write_text(
        "# Claim reviews\n\nSee `results/fs15_hypothesis/results.json` for H1–H3.\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    assert analysis["n_accepted"] >= 2
    print("wrote", OUT)


if __name__ == "__main__":
    main()
