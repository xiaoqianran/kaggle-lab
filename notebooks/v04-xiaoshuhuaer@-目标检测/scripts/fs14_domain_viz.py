#!/usr/bin/env python3
"""FS14: domain shift + error modes + Kaggle playbook (visual)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, collate_det, generate_one  # noqa: E402
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import box_iou_xyxy, coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "fs14_domain"
OUT.mkdir(parents=True, exist_ok=True)


class DomainShiftDataset(torch.utils.data.Dataset):
    def __init__(self, n, seed=0):
        self.cfg = SynthConfig(img_size=64, min_size=5, max_size=14, max_objects=4, seed=seed)
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n):
            s = generate_one(rng, self.cfg)
            img = np.clip(s["image"] * 0.55 + 0.02, 0, 1)
            img[..., 1] = np.clip(img[..., 1] * 1.1, 0, 1)
            img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)
            s["image"] = img
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.from_numpy(s["image"]).permute(2, 0, 1), {
            "boxes": torch.from_numpy(s["boxes"]),
            "labels": torch.from_numpy(s["labels"]),
        }


def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            dec = decode_centernet(model(imgs), conf_thr=0.25)
            for d, t in zip(dec, targets):
                boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_style_ap(preds, gts), preds, gts


def error_analysis(preds, gts, iou_thr=0.5):
    loc_err = cls_err = miss = fp = 0
    for p, g in zip(preds, gts):
        pb, pl = p["boxes"], p["labels"]
        gb, gl = g["boxes"], g["labels"]
        if len(gb) == 0:
            fp += len(pb)
            continue
        if len(pb) == 0:
            miss += len(gb)
            continue
        ious = box_iou_xyxy(pb, gb)
        matched = set()
        for i in range(len(pb)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thr and j not in matched:
                matched.add(j)
                if pl[i] != gl[j]:
                    cls_err += 1
                elif ious[i, j] < 0.75:
                    loc_err += 1
            else:
                fp += 1
        miss += len(gb) - len(matched)
    return {"loc_err": loc_err, "cls_err": cls_err, "miss": miss, "fp": fp}


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    train_ds = SynthDetDataset(240, SynthConfig(img_size=64), seed=1)
    id_val = SynthDetDataset(80, SynthConfig(img_size=64), seed=2)
    ood_val = DomainShiftDataset(80, seed=3)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    id_loader = DataLoader(id_val, batch_size=16, shuffle=False, collate_fn=collate_det)
    ood_loader = DataLoader(ood_val, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = CenterNetLite()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(14):
        model.train()
        for imgs, targets in train_loader:
            loss, _ = centernet_loss(model(imgs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m_id, _, _ = evaluate(model, id_loader)
    m_ood, preds_ood, gts_ood = evaluate(model, ood_loader)
    err = error_analysis(preds_ood, gts_ood)
    # FT
    for epoch in range(6):
        model.train()
        for imgs, targets in ood_loader:
            loss, _ = centernet_loss(model(imgs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m_ft, _, _ = evaluate(model, ood_loader)

    save_bar_compare(
        OUT / "domain_ap50.png",
        ["ID", "OOD", "OOD+FT"],
        [m_id.ap50, m_ood.ap50, m_ft.ap50],
        "FS14 · domain shift AP50",
    )
    save_bar_compare(
        OUT / "error_modes.png",
        list(err.keys()),
        list(err.values()),
        "FS14 · OOD error counts",
        ylabel="count",
    )
    img, t = ood_val[0]
    with torch.no_grad():
        d = decode_centernet(model(img.unsqueeze(0)), conf_thr=0.25)[0]
    boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
    if len(boxes):
        k = nms_xyxy(boxes, scores, 0.5)
        boxes, scores, labels = boxes[k], scores[k], labels[k]
    save_det_panel(
        OUT / "ood_pred.png",
        img.permute(1, 2, 0).numpy(),
        t["boxes"].numpy(),
        t["labels"].numpy(),
        boxes,
        scores,
        labels,
        "FS14 OOD after FT",
    )

    catalog = json.loads((ROOT / "catalog.json").read_text())
    analysis = {
        "step": "FS14",
        "concept": "Domain shift / real Kaggle domains; open-vocab is next frontier setting",
        "in_domain": m_id.as_dict(),
        "ood": m_ood.as_dict(),
        "ood_ft": m_ft.as_dict(),
        "error_modes": err,
        "kaggle_active": [c["ref"] for c in catalog.get("competitions", []) if c.get("status") == "active"][:6],
        "new_capability": "Diagnose domain failure modes; adapt with target FT",
        "vs_previous": "FS13 closed-set synth; FS14 shows beautiful AP can collapse under shift",
        "frontier": "OWL-ViT / Grounding DINO for text-open classes (inference on Kaggle GPU)",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"ID={m_id.ap50:.3f} OOD={m_ood.ap50:.3f} FT={m_ft.ap50:.3f} err={err}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
