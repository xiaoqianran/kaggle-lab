#!/usr/bin/env python3
"""FS07: Faster-RCNN-lite train + viz (wraps TwoStageLite)."""
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
from models import TwoStageLite, decode_two_stage, two_stage_lite_loss  # noqa: E402

OUT = ROOT / "results" / "fs07_two_stage"
OUT.mkdir(parents=True, exist_ok=True)


def eval_model(model, loader, conf_thr=0.25):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            rpn, head, _ = model(imgs)
            dec = decode_two_stage(rpn, head, conf_thr=conf_thr)
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
    train_ds = SynthDetDataset(200, cfg, seed=10)
    val_ds = SynthDetDataset(60, cfg, seed=99)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = TwoStageLite()
    m0 = eval_model(model, val_loader)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for epoch in range(12):
        model.train()
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
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)), **m.as_dict()})
            print(f"epoch {epoch:02d} loss={hist[-1]['loss']:.3f} AP50={m.ap50:.3f}")
    m_final = eval_model(model, val_loader)
    save_bar_compare(OUT / "ap50.png", ["untrained", "trained"], [m0.ap50, m_final.ap50], "FS07 TwoStageLite AP50")

    img, t = val_ds[0]
    with torch.no_grad():
        rpn, head, _ = model(img.unsqueeze(0))
        d = decode_two_stage(rpn, head, conf_thr=0.25)[0]
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
        "FS07 RPN+head predictions",
    )
    analysis = {
        "step": "FS07",
        "concept": "Learned proposals (RPN-like) + second stage cls/reg",
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": hist,
        "new_capability": "Network learns WHERE (objectness map) not just hand proposals",
        "vs_previous": "FS05 external proposals; FS06 share compute; FS07 learns proposals",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    assert m_final.ap50 > m0.ap50 + 0.1
    print("wrote", OUT)


if __name__ == "__main__":
    main()
