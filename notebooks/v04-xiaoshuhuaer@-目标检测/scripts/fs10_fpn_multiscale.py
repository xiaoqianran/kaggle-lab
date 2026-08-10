#!/usr/bin/env python3
"""FS10: Multi-scale features intuition — single stride vs multi-stride heads.

Compare CenterNetLite stride-8 only vs dual-head (stride 8 + stride 16-like pooling)
for small vs large objects AP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, collate_det  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import (  # noqa: E402
    CenterNetLite,
    TinyBackbone,
    build_centernet_targets,
    centernet_loss,
    decode_centernet,
)

OUT = ROOT / "results" / "fs10_fpn_multiscale"
OUT.mkdir(parents=True, exist_ok=True)


class DualStrideLite(nn.Module):
    """Two heads at different resolutions (FPN intuition without lateral fuse)."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = TinyBackbone(64)  # -> 8x8
        self.head_p3 = nn.Conv2d(64, 1 + n_classes + 4, 1)  # fine
        self.down = nn.Conv2d(64, 64, 3, 2, 1)  # 4x4 coarse
        self.head_p4 = nn.Conv2d(64, 1 + n_classes + 4, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_p3(f), self.head_p4(self.down(f))


def decode_dual(p3, p4, n_classes=3, conf_thr=0.25):
    d3 = decode_centernet(p3, stride=8, conf_thr=conf_thr, n_classes=n_classes)
    d4 = decode_centernet(p4, stride=16, conf_thr=conf_thr, n_classes=n_classes)
    out = []
    for a, b in zip(d3, d4):
        if len(a["boxes"]) == 0 and len(b["boxes"]) == 0:
            out.append(a)
            continue
        boxes = torch.cat([a["boxes"], b["boxes"]], 0) if len(a["boxes"]) and len(b["boxes"]) else (
            a["boxes"] if len(a["boxes"]) else b["boxes"]
        )
        scores = torch.cat([a["scores"], b["scores"]], 0) if len(a["scores"]) and len(b["scores"]) else (
            a["scores"] if len(a["scores"]) else b["scores"]
        )
        labels = torch.cat([a["labels"], b["labels"]], 0) if len(a["labels"]) and len(b["labels"]) else (
            a["labels"] if len(a["labels"]) else b["labels"]
        )
        out.append({"boxes": boxes, "scores": scores, "labels": labels})
    return out


def dual_loss(p3, p4, targets, n_classes=3):
    # assign large objects preference to coarse by size
    loss3, parts3 = centernet_loss(p3, targets, n_classes=n_classes, stride=8)
    # build targets at stride 16
    B, _, H, W = p4.shape
    obj_t, cls_t, reg_t = build_centernet_targets(targets, H, W, 16, n_classes, 64)
    device = p4.device
    obj_t, cls_t, reg_t = obj_t.to(device), cls_t.to(device), reg_t.to(device)
    # only keep large-ish GT (area > 16^2) for coarse head — FPN size routing idea
    # mask by zeroing small centers: approximate via reg target sum
    loss_obj = F.binary_cross_entropy_with_logits(p4[:, 0], obj_t)
    pos = obj_t > 0.5
    if pos.any():
        loss_cls = F.cross_entropy(p4[:, 1 : 1 + n_classes].permute(0, 2, 3, 1)[pos], cls_t[pos])
        loss_reg = F.l1_loss(
            F.relu(p4[:, 1 + n_classes :].permute(0, 2, 3, 1)[pos]),
            reg_t.permute(0, 2, 3, 1)[pos],
        )
    else:
        loss_cls = p4.sum() * 0
        loss_reg = p4.sum() * 0
    loss4 = loss_obj + loss_cls + loss_reg
    return loss3 + 0.5 * loss4, {"loss": float((loss3 + 0.5 * loss4).detach())}


def eval_model(model, loader, dual=False):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            if dual:
                p3, p4 = model(imgs)
                dec = decode_dual(p3, p4)
            else:
                dec = decode_centernet(model(imgs), conf_thr=0.25)
            for d, t in zip(dec, targets):
                boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts)


def main() -> None:
    set_seed(1)
    torch.manual_seed(1)
    cfg = SynthConfig(img_size=64, min_size=6, max_size=30, max_objects=3)
    train_ds = SynthDetDataset(220, cfg, seed=2)
    val_ds = SynthDetDataset(70, cfg, seed=22)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)

    # single
    m1 = CenterNetLite()
    opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
    for epoch in range(12):
        m1.train()
        for imgs, targets in train_loader:
            loss, _ = centernet_loss(m1(imgs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    met1 = eval_model(m1, val_loader, dual=False)

    # dual
    m2 = DualStrideLite()
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    for epoch in range(12):
        m2.train()
        for imgs, targets in train_loader:
            p3, p4 = m2(imgs)
            loss, _ = dual_loss(p3, p4, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    met2 = eval_model(m2, val_loader, dual=True)

    analysis = {
        "single_stride8": met1.as_dict(),
        "dual_stride_8_16": met2.as_dict(),
        "delta_ap50": met2.ap50 - met1.ap50,
        "delta_ap_s": met2.ap_s - met1.ap_s,
        "delta_ap_l": met2.ap_l - met1.ap_l,
        "findings": [
            "FPN family: different object sizes need different feature strides.",
            "Dual head is a minimal multi-scale; full FPN adds top-down lateral fusion.",
            "Watch AP_s vs AP_l shifts — multi-scale is about size routing, not magic AP free lunch.",
        ],
        "disease_fixed": "single-feature-map scale blindness",
        "paper": "FPN (Lin et al. 2017)",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(
        f"single AP50={met1.ap50:.3f} APs={met1.ap_s:.3f} | "
        f"dual AP50={met2.ap50:.3f} APs={met2.ap_s:.3f}"
    )
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
