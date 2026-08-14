#!/usr/bin/env python3
"""FS09: Class imbalance — CE vs Focal on dense grid (RetinaNet insight).

Reports pos_fraction and both losses. On *easy* synth CE may win with default
focal hyperparams — that itself is a lesson (γ/α need tuning; COCO is harder).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, collate_det  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import (  # noqa: E402
    CenterNetLite,
    build_centernet_targets,
    decode_centernet,
)

OUT = ROOT / "results" / "fs09_focal_loss"
OUT.mkdir(parents=True, exist_ok=True)


def focal_bce_with_logits(logits, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def train_variant(use_focal: bool, epochs=14, seed=0, gamma=2.0, alpha=0.25):
    set_seed(seed)
    torch.manual_seed(seed)
    cfg = SynthConfig(img_size=64)
    train_ds = SynthDetDataset(200, cfg, seed=seed + 1)
    val_ds = SynthDetDataset(60, cfg, seed=seed + 50)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = CenterNetLite(n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for epoch in range(epochs):
        model.train()
        losses, pos_fracs = [], []
        for imgs, targets in train_loader:
            out = model(imgs)
            B, _, H, W = out.shape
            obj_t, cls_t, reg_t = build_centernet_targets(targets, H, W, 8, 3, 64)
            pos_fracs.append(float(obj_t.mean()))
            obj_p = out[:, 0]
            loss_obj = (
                focal_bce_with_logits(obj_p, obj_t, gamma=gamma, alpha=alpha)
                if use_focal
                else F.binary_cross_entropy_with_logits(obj_p, obj_t)
            )
            cls_p = out[:, 1:4]
            reg_p = out[:, 4:]
            pos = obj_t > 0.5
            if pos.any():
                loss_cls = F.cross_entropy(cls_p.permute(0, 2, 3, 1)[pos], cls_t[pos])
                loss_reg = F.l1_loss(
                    F.relu(reg_p.permute(0, 2, 3, 1)[pos]), reg_t.permute(0, 2, 3, 1)[pos]
                )
            else:
                loss_cls = out.sum() * 0
                loss_reg = out.sum() * 0
            loss = loss_obj + loss_cls + loss_reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        hist.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "pos_frac": float(np.mean(pos_fracs)),
            }
        )

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            dec = decode_centernet(model(imgs), conf_thr=0.25)
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
    return coco_style_ap(preds, gts), hist


def main() -> None:
    m_ce, h_ce = train_variant(False, seed=0)
    m_f, h_f = train_variant(True, seed=0, gamma=2.0, alpha=0.25)
    m_f2, h_f2 = train_variant(True, seed=0, gamma=1.0, alpha=0.5)
    analysis = {
        "pos_fraction_mean": h_ce[-1]["pos_frac"],
        "imbalance_note": "pos cells << 5% of 8x8 grid — classic dense-head imbalance",
        "ce": {"final": m_ce.as_dict(), "history": h_ce},
        "focal_g2_a025": {"final": m_f.as_dict(), "history": h_f},
        "focal_g1_a05": {"final": m_f2.as_dict(), "history": h_f2},
        "delta_ap50_focal_g2_minus_ce": m_f.ap50 - m_ce.ap50,
        "delta_ap50_focal_g1_minus_ce": m_f2.ap50 - m_ce.ap50,
        "findings": [
            f"Dense objectness positives ≈ {h_ce[-1]['pos_frac']*100:.2f}% of cells.",
            "Focal is the historical fix for easy-negative drowning (RetinaNet).",
            "On easy color-synth, default γ=2,α=0.25 may underperform CE — hyperparams matter; "
            "try milder focal (γ=1,α=0.5) as second arm.",
            "Lesson: a famous method is not a free +AP button; measure under fixed protocol.",
        ],
        "disease_targeted": "easy-negative gradient drowning on dense maps",
        "paper": "RetinaNet / Focal Loss (Lin et al. 2017)",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(
        f"CE AP50={m_ce.ap50:.3f} | Focalγ2={m_f.ap50:.3f} | Focalγ1={m_f2.ap50:.3f}"
    )
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
