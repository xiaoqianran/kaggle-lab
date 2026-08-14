#!/usr/bin/env python3
"""FS08: Dense one-stage detector train + viz."""
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
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "fs08_dense_yolo"
OUT.mkdir(parents=True, exist_ok=True)


def eval_model(model, loader, conf_thr=0.25):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            dec = decode_centernet(model(imgs), conf_thr=conf_thr)
            for d, t in zip(dec, targets):
                boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    cfg = SynthConfig(img_size=64)
    train_ds = SynthDetDataset(220, cfg, seed=42)
    val_ds = SynthDetDataset(70, cfg, seed=142)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = CenterNetLite()
    m0 = eval_model(model, val_loader)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for epoch in range(15):
        model.train()
        losses = []
        for imgs, targets in train_loader:
            loss, parts = centernet_loss(model(imgs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(parts["loss"])
        if epoch % 4 == 0 or epoch == 14:
            m = eval_model(model, val_loader)
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)), **m.as_dict()})
            print(f"epoch {epoch:02d} loss={hist[-1]['loss']:.3f} AP50={m.ap50:.3f}")
    m_final = eval_model(model, val_loader)
    conf_rows = []
    for thr in [0.15, 0.25, 0.4, 0.6]:
        mm = eval_model(model, val_loader, conf_thr=thr)
        conf_rows.append({"conf": thr, "ap50": mm.ap50})
    save_bar_compare(OUT / "ap50.png", ["untrained", "trained"], [m0.ap50, m_final.ap50], "FS08 dense AP50")
    save_bar_compare(
        OUT / "conf_ablation.png",
        [str(r["conf"]) for r in conf_rows],
        [r["ap50"] for r in conf_rows],
        "FS08 conf thr ablation",
    )
    img, t = val_ds[1]
    with torch.no_grad():
        d = decode_centernet(model(img.unsqueeze(0)), conf_thr=0.25)[0]
    boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
    if len(boxes):
        k = nms_xyxy(boxes, scores, 0.5)
        boxes, scores, labels = boxes[k], scores[k], labels[k]
    save_det_panel(
        OUT / "pred.png",
        img.permute(1, 2, 0).numpy(),
        t["boxes"].numpy(),
        t["labels"].numpy(),
        boxes,
        scores,
        labels,
        "FS08 dense grid predictions",
    )
    analysis = {
        "step": "FS08",
        "concept": "Single-shot dense prediction (YOLO/FCOS flavor)",
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": hist,
        "conf_ablation": conf_rows,
        "new_capability": "One forward → all boxes; no explicit RPN stage",
        "vs_previous": "FS07 two-stage; FS08 denser & usually faster to high AP on easy data",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    assert m_final.ap50 > m0.ap50 + 0.2
    print("wrote", OUT)


if __name__ == "__main__":
    main()
