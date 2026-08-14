#!/usr/bin/env python3
"""P3: single-stage YOLO/FCOS-lite (CenterNetLite) train + imgsz/conf ablation."""
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
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "p3_yolo_lite"
OUT.mkdir(parents=True, exist_ok=True)
PAPERS = ROOT / "papers" / "YOLO_Ultralytics"
PAPERS.mkdir(parents=True, exist_ok=True)


def eval_model(model, loader, conf_thr=0.25):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs)
            dec = decode_centernet(out, conf_thr=conf_thr)
            for d, t in zip(dec, targets):
                boxes = d["boxes"].numpy()
                scores = d["scores"].numpy()
                labels = d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def train_once(seed=42, epochs=15, lr=1e-3):
    set_seed(seed)
    torch.manual_seed(seed)
    cfg = SynthConfig(img_size=64)
    train_ds = SynthDetDataset(220, cfg, seed=seed)
    val_ds = SynthDetDataset(70, cfg, seed=seed + 100)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = CenterNetLite(n_classes=3)
    m0 = eval_model(model, val_loader)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, targets in train_loader:
            out = model(imgs)
            loss, parts = centernet_loss(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(parts["loss"])
        if epoch % 4 == 0 or epoch == epochs - 1:
            m = eval_model(model, val_loader)
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)), **m.as_dict()})
            print(f"epoch {epoch:02d} loss={hist[-1]['loss']:.3f} AP50={m.ap50:.3f}")
    m_final = eval_model(model, val_loader)
    # conf ablation
    conf_rows = []
    for thr in [0.15, 0.25, 0.4, 0.6]:
        m = eval_model(model, val_loader, conf_thr=thr)
        conf_rows.append({"conf_thr": thr, **m.as_dict()})
    return {
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": hist,
        "conf_ablation": conf_rows,
        "model": model,
        "val_loader": val_loader,
    }


def main() -> None:
    r = train_once()
    analysis = {
        "model": "CenterNetLite (dense single-stage, stride=8)",
        "paradigm": "YOLO/FCOS-like center assign + ltrb regression",
        "untrained": r["untrained"],
        "final": r["final"],
        "history": r["history"],
        "conf_ablation": r["conf_ablation"],
        "findings": [
            "Single-stage dense head trains faster to high AP on easy synth than two-stage lite in same epochs budget.",
            "conf_thr ablation changes AP50; protocol must freeze thr when comparing models.",
            "Map to Ultralytics: metrics.py / loss / trainer — see SOURCE_MAP.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (PAPERS / "SOURCE_MAP.md").write_text(
        """# SOURCE_MAP · YOLO modern / Ultralytics

| Concept | CenterNetLite | Ultralytics (typical paths) |
|---------|---------------|-----------------------------|
| Backbone+Neck+Head | `TinyBackbone`+`head` | `nn/tasks.py` DetectionModel |
| Dense decode | `decode_centernet` | head decode in `ops` / predictor |
| Assign | center cell (lite) | TaskAligned / SimOTA family in loss |
| Loss obj/cls/box | `centernet_loss` | `utils/loss.py` (version-dependent) |
| Metrics mAP | `metrics.coco_style_ap` | `utils/metrics.py` DetMetrics |
| Trainer loop | this script | `engine/trainer.py` |

## Read order
1. RetinaNet Focal Loss (imbalance idea)
2. Ultralytics `utils/metrics.py` AP accumulation
3. This dense train loop

## Note
Full Ultralytics not vendored in CI; SOURCE_MAP is the contract for Kaggle GPU follow-up.
""",
        encoding="utf-8",
    )
    print("wrote", OUT / "results.json")
    assert r["final"]["ap50"] > r["untrained"]["ap50"] + 0.2


if __name__ == "__main__":
    main()
