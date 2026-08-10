#!/usr/bin/env python3
"""P5: training recipe ablation on fixed data/protocol (lr, epochs, hflip aug)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from boxes import horizontal_flip_xyxy  # noqa: E402
from data_synth import SynthConfig, SynthDetDataset, collate_det  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "p5_recipe"
OUT.mkdir(parents=True, exist_ok=True)


class FlipDataset(torch.utils.data.Dataset):
    def __init__(self, base: SynthDetDataset, p=0.5, seed=0):
        self.base = base
        self.p = p
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if self.rng.rand() < self.p:
            img = torch.flip(img, dims=[2])  # W axis
            boxes = target["boxes"].numpy()
            boxes = horizontal_flip_xyxy(boxes, width=float(img.shape[2]))
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": target["labels"],
            }
        return img, target


def train_eval(epochs, lr, use_flip, seed=0):
    set_seed(seed)
    torch.manual_seed(seed)
    cfg = SynthConfig(img_size=64)
    train_base = SynthDetDataset(200, cfg, seed=seed + 1)
    val_ds = SynthDetDataset(60, cfg, seed=seed + 50)
    train_ds = FlipDataset(train_base, p=0.5 if use_flip else 0.0, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_det)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_det)
    model = CenterNetLite(n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for imgs, targets in train_loader:
            out = model(imgs)
            loss, _ = centernet_loss(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            out = model(imgs)
            dec = decode_centernet(out, conf_thr=0.25)
            for d, t in zip(dec, targets):
                boxes = d["boxes"].numpy()
                scores = d["scores"].numpy()
                labels = d["labels"].numpy()
                if len(boxes):
                    keep = nms_xyxy(boxes, scores, 0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    m = coco_style_ap(preds, gts)
    return m


def main() -> None:
    recipes = [
        {"name": "baseline", "epochs": 10, "lr": 1e-3, "use_flip": False},
        {"name": "longer", "epochs": 18, "lr": 1e-3, "use_flip": False},
        {"name": "low_lr", "epochs": 10, "lr": 1e-4, "use_flip": False},
        {"name": "flip_aug", "epochs": 10, "lr": 1e-3, "use_flip": True},
        {"name": "flip_longer", "epochs": 18, "lr": 1e-3, "use_flip": True},
    ]
    rows = []
    for r in recipes:
        print("running", r["name"], "...")
        m = train_eval(r["epochs"], r["lr"], r["use_flip"], seed=11)
        row = {**r, **m.as_dict()}
        rows.append(row)
        print(f"  {r['name']:12s} AP50={m.ap50:.3f} AP={m.ap:.3f}")

    best = max(rows, key=lambda x: x["ap50"])
    analysis = {
        "protocol": "coco_style_v0 conf=0.25 fixed val split",
        "rows": rows,
        "best": best["name"],
        "findings": [
            "Compare only under fixed val + conf — recipe claims otherwise invalid.",
            "Longer training often helps on easy synth; low_lr may underfit at same epochs.",
            "HFlip with correct box transform is free diversity; broken flip would destroy boxes (tested in P1).",
            f"Best recipe by AP50: {best['name']} AP50={best['ap50']:.3f}",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("wrote", OUT / "results.json")
    assert max(r["ap50"] for r in rows) > 0.3


if __name__ == "__main__":
    main()
