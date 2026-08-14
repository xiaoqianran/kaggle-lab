#!/usr/bin/env python3
"""P4: DETR-lite Hungarian matching + train; matching visualization stats."""
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
from data_synth import SynthConfig, SynthDetDataset, collate_det  # noqa: E402
from metrics import coco_style_ap, set_seed  # noqa: E402
from models import DETRLite, decode_detr, detr_lite_loss  # noqa: E402

OUT = ROOT / "results" / "p4_detr"
OUT.mkdir(parents=True, exist_ok=True)
PAPERS = ROOT / "papers" / "DETR"
PAPERS.mkdir(parents=True, exist_ok=True)


def eval_model(model, loader, conf_thr=0.3):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            logits, boxes = model(imgs)
            dec = decode_detr(logits, boxes, conf_thr=conf_thr)
            for d, t in zip(dec, targets):
                preds.append(
                    {
                        "boxes": d["boxes"].numpy(),
                        "scores": d["scores"].numpy(),
                        "labels": d["labels"].numpy(),
                    }
                )
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def main() -> None:
    set_seed(7)
    torch.manual_seed(7)
    cfg = SynthConfig(img_size=64)
    train_ds = SynthDetDataset(180, cfg, seed=3)
    val_ds = SynthDetDataset(50, cfg, seed=33)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_det)

    model = DETRLite(n_classes=3, n_queries=10, d_model=64)
    m0 = eval_model(model, val_loader)
    print(f"untrained AP50={m0.ap50:.3f}")

    # matching demo on random outputs
    s = train_ds.samples[0]
    with torch.no_grad():
        img = torch.from_numpy(s["image"]).permute(2, 0, 1).unsqueeze(0)
        logits, boxes = model(img)
        pb = boxes[0].numpy() * 64
        pb_xyxy = cxcywh_to_xyxy(pb)
        match = hungarian_match(
            pb_xyxy, logits[0].numpy(), s["boxes"], s["labels"], num_classes=4
        )
    print(
        "match pred_idx",
        match["pred_idx"].tolist(),
        "gt_idx",
        match["gt_idx"].tolist(),
        "costs",
        np.round(match["matched_cost"], 3).tolist(),
    )

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    history = []
    for epoch in range(20):
        model.train()
        losses = []
        for imgs, targets in train_loader:
            logits, boxes = model(imgs)
            loss, logs = detr_lite_loss(logits, boxes, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == 19:
            m = eval_model(model, val_loader)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), **m.as_dict()})
            print(f"epoch {epoch:02d} loss={history[-1]['loss']:.3f} AP50={m.ap50:.3f}")

    m_final = eval_model(model, val_loader)
    # query count ablation (reinit small train)
    query_rows = []
    for nq in [6, 10, 16]:
        set_seed(7)
        torch.manual_seed(7)
        mtmp = DETRLite(n_classes=3, n_queries=nq)
        opt = torch.optim.Adam(mtmp.parameters(), lr=2e-4)
        for epoch in range(12):
            mtmp.train()
            for imgs, targets in train_loader:
                logits, boxes = mtmp(imgs)
                loss, _ = detr_lite_loss(logits, boxes, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
        mm = eval_model(mtmp, val_loader)
        query_rows.append({"n_queries": nq, **mm.as_dict()})
        print(f"n_queries={nq} AP50={mm.ap50:.3f}")

    analysis = {
        "model": "DETRLite set prediction",
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": history,
        "example_match": {
            "pred_idx": match["pred_idx"].tolist(),
            "gt_idx": match["gt_idx"].tolist(),
            "matched_cost": match["matched_cost"].tolist(),
        },
        "query_ablation": query_rows,
        "findings": [
            "Hungarian matching assigns each GT to exactly one query (bipartite).",
            "Unmatched queries learn no-object class — DETR is not dense YOLO.",
            "More queries can help capacity but may need longer training; effect measured in ablation.",
            "Convergence is slower than CenterNetLite on same data — known DETR trait.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (PAPERS / "SOURCE_MAP.md").write_text(
        """# SOURCE_MAP · DETR

| Paper concept | DETRLite | Official facebookresearch/detr |
|---------------|----------|--------------------------------|
| Backbone | `TinyBackbone` | ResNet + position encoding |
| Transformer enc/dec | `nn.Transformer*` 1 layer | `models/transformer.py` |
| Object queries | `query_embed` | `query_embed` |
| Hungarian match | `assignment.hungarian_match` | `models/matcher.py` HungarianMatcher |
| Loss CE+L1(+GIoU) | `detr_lite_loss` | `models/detr.py` SetCriterion |
| no-object class | last logit | `num_classes` eos |

## Read order
1. DETR paper matching cost definition
2. `matcher.py` cost_class / cost_bbox / cost_giou
3. This script matching dump + train
""",
        encoding="utf-8",
    )
    print("wrote", OUT / "results.json")
    # acceptance: loss decreases or AP improves
    assert history[-1]["loss"] < history[0]["loss"] or m_final.ap50 >= m0.ap50


if __name__ == "__main__":
    main()
