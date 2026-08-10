#!/usr/bin/env python3
"""FS12: DETR-lite + matching viz."""
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
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import coco_style_ap, set_seed  # noqa: E402
from models import DETRLite, decode_detr, detr_lite_loss  # noqa: E402

OUT = ROOT / "results" / "fs12_detr"
OUT.mkdir(parents=True, exist_ok=True)


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
    model = DETRLite(n_classes=3, n_queries=10)
    m0 = eval_model(model, val_loader)
    # matching demo
    s = train_ds.samples[0]
    with torch.no_grad():
        img = torch.from_numpy(s["image"]).permute(2, 0, 1).unsqueeze(0)
        logits, boxes = model(img)
        pb = cxcywh_to_xyxy(boxes[0].numpy() * 64)
        match = hungarian_match(pb, logits[0].numpy(), s["boxes"], s["labels"], 4)

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    hist = []
    for epoch in range(18):
        model.train()
        losses = []
        for imgs, targets in train_loader:
            logits, boxes = model(imgs)
            loss, _ = detr_lite_loss(logits, boxes, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch % 6 == 0 or epoch == 17:
            m = eval_model(model, val_loader)
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)), **m.as_dict()})
            print(f"epoch {epoch:02d} loss={hist[-1]['loss']:.3f} AP50={m.ap50:.3f}")
    m_final = eval_model(model, val_loader)
    save_bar_compare(
        OUT / "loss.png",
        [str(h["epoch"]) for h in hist],
        [h["loss"] for h in hist],
        "FS12 DETR loss vs epoch",
        ylabel="loss",
    )
    with torch.no_grad():
        d = decode_detr(*model(torch.from_numpy(s["image"]).permute(2, 0, 1).unsqueeze(0)), conf_thr=0.2)[0]
    save_det_panel(
        OUT / "pred.png",
        s["image"],
        s["boxes"],
        s["labels"],
        d["boxes"].numpy(),
        d["scores"].numpy(),
        d["labels"].numpy(),
        "FS12 DETR-lite preds",
    )
    analysis = {
        "step": "FS12",
        "concept": "Set prediction + Hungarian matching (no NMS paradigm)",
        "untrained": m0.as_dict(),
        "final": m_final.as_dict(),
        "history": hist,
        "example_match": {
            "pred_idx": match["pred_idx"].tolist(),
            "gt_idx": match["gt_idx"].tolist(),
            "costs": match["matched_cost"].tolist(),
        },
        "new_capability": "Bipartite matching end-to-end; queries as object slots",
        "vs_previous": "FS08 dense+NMS; FS12 set prediction, slower converge",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    assert hist[-1]["loss"] < hist[0]["loss"] or m_final.ap50 >= m0.ap50
    print("wrote", OUT)


if __name__ == "__main__":
    main()
