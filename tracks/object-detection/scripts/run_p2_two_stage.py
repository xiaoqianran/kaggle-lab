#!/usr/bin/env python3
"""P2: two-stage lite train on synthetic data + SOURCE_MAP notes + mAP eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, collate_det  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import TwoStageLite, decode_two_stage, two_stage_lite_loss  # noqa: E402

OUT = ROOT / "results" / "p2_two_stage"
OUT.mkdir(parents=True, exist_ok=True)
PAPERS = ROOT / "papers" / "FasterRCNN"
PAPERS.mkdir(parents=True, exist_ok=True)


def eval_model(model, loader, conf_thr=0.25):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            rpn, head, _ = model(imgs)
            dec = decode_two_stage(rpn, head, conf_thr=conf_thr)
            for d, t in zip(dec, targets):
                boxes = d["boxes"].numpy()
                scores = d["scores"].numpy()
                labels = d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append(
                    {
                        "boxes": t["boxes"].numpy(),
                        "labels": t["labels"].numpy(),
                    }
                )
    return coco_style_ap(preds, gts)


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    cfg = SynthConfig(img_size=64, n_classes=3)
    train_ds = SynthDetDataset(200, cfg, seed=10)
    val_ds = SynthDetDataset(60, cfg, seed=99)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)

    model = TwoStageLite(n_classes=3)
    # untrained baseline
    m0 = eval_model(model, val_loader)
    print(f"untrained AP50={m0.ap50:.3f} AP={m0.ap:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    model.train()
    for epoch in range(12):
        losses = []
        for imgs, targets in train_loader:
            rpn, head, _ = model(imgs)
            loss, parts = two_stage_lite_loss(rpn, head, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(parts["loss"])
        if epoch % 3 == 0 or epoch == 11:
            m = eval_model(model, val_loader)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **m.as_dict(),
            }
            history.append(row)
            print(
                f"epoch {epoch:02d} loss={row['train_loss']:.3f} "
                f"AP50={m.ap50:.3f} AP={m.ap:.3f}"
            )

    m_final = eval_model(model, val_loader)
    # ablation: epochs proxy — compare first eval after e0 vs final
    analysis = {
        "model": "TwoStageLite (RPN obj + cls/reg head pedagogical)",
        "data": "SynthDet 64x64 colored shapes",
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": history,
        "acceptance": {
            "final_ap50_gt_untrained": m_final.ap50 > m0.ap50 + 0.15,
            "final_ap50": m_final.ap50,
            "untrained_ap50": m0.ap50,
        },
        "findings": [
            "RPN objectness + second head cls/reg is the two-stage DNA in miniature.",
            "Training reduces loss and lifts AP50 vs random init on easy synthetic.",
            "Not Faster R-CNN production; SOURCE_MAP maps concepts to Detectron2/torchvision paths.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    source_map = """# SOURCE_MAP · Faster R-CNN concepts → real code

| Concept (paper) | TwoStageLite (this repo) | Detectron2 | torchvision |
|-----------------|--------------------------|------------|-------------|
| Shared backbone | `TinyBackbone` | `build_resnet_fpn_backbone` | `fasterrcnn_resnet50_fpn` body |
| RPN objectness | `self.rpn` 1-ch map | `proposal_generator/rpn.py` | `RPNHead` |
| RoI feature | *simplified: same grid cell* | `roi_heads` + RoIAlign | `MultiScaleRoIAlign` |
| Box regressor | `head` last 4 ch | `box_head` / `box_predictor` | `FastRCNNPredictor` |
| Assign pos/neg | center cell (lite) / `assignment.assign_iou` | `Matcher` + `subsample_labels` | `box_ops` / anchors |
| NMS | `metrics.nms_xyxy` | `batched_nms` | `nms` |

## Read order
1. Faster R-CNN paper §§3.1–3.2 (RPN + loss)
2. Detectron2 `RPN` forward + losses
3. This `TwoStageLite` training loop

## Open questions
- Full anchor-based RPN with multi-scale anchors not trained here (CPU budget).
- RoIAlign omitted; pedagogical center assign only.
"""
    (PAPERS / "SOURCE_MAP.md").write_text(source_map, encoding="utf-8")
    print("wrote", OUT / "results.json")
    assert m_final.ap50 > m0.ap50 + 0.1, (m_final.ap50, m0.ap50)


if __name__ == "__main__":
    main()
