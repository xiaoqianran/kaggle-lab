#!/usr/bin/env python3
"""FS05: R-CNN idea — classify region proposals (crops), not every window.

Proposals = GT jitter + random background boxes (simulates selective search quality mix).
CNN: tiny conv net on 32x32 crops. Compare vs FS01 sliding window metrics on same val set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, generate_one  # noqa: E402
from fs01_sliding_window import sliding_window_detect  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402

OUT = ROOT / "results" / "fs05_rcnn_crops"
OUT.mkdir(parents=True, exist_ok=True)


class TinyCropNet(nn.Module):
    def __init__(self, n_classes=4):  # 3 obj + bg
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def crop_resize(img: np.ndarray, box, size=32) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        patch = np.zeros((size, size, 3), np.float32)
    else:
        patch = img[y1:y2, x1:x2]
        ys = (np.linspace(0, patch.shape[0] - 1, size)).astype(int)
        xs = (np.linspace(0, patch.shape[1] - 1, size)).astype(int)
        patch = patch[ys][:, xs].astype(np.float32)
    return patch


def make_proposals(img, gt_boxes, gt_labels, rng, n_bg=8):
    """GT-like proposals + random bg — teaches positive/negative crops."""
    H, W = img.shape[:2]
    props, labs = [], []
    for b, lab in zip(gt_boxes, gt_labels):
        # jittered positive
        for _ in range(2):
            bb = b.astype(float).copy()
            cx = 0.5 * (bb[0] + bb[2])
            cy = 0.5 * (bb[1] + bb[3])
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            cx += rng.normal(0, 0.08 * w)
            cy += rng.normal(0, 0.08 * h)
            w *= 1 + rng.normal(0, 0.1)
            h *= 1 + rng.normal(0, 0.1)
            props.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            labs.append(int(lab))
        # exact GT
        props.append(b.astype(float).tolist())
        labs.append(int(lab))
    for _ in range(n_bg):
        w = rng.integers(8, 18)
        h = rng.integers(8, 18)
        x1 = rng.integers(0, max(1, W - w))
        y1 = rng.integers(0, max(1, H - h))
        props.append([x1, y1, x1 + w, y1 + h])
        labs.append(3)  # bg
    return np.asarray(props, float), np.asarray(labs, int)


class CropDataset(Dataset):
    def __init__(self, n=300, seed=0):
        self.rng = np.random.default_rng(seed)
        self.cfg = SynthConfig(img_size=64, min_size=10, max_size=24)
        self.items = []
        for _ in range(n):
            s = generate_one(self.rng, self.cfg)
            props, labs = make_proposals(s["image"], s["boxes"], s["labels"], self.rng)
            for p, lab in zip(props, labs):
                crop = crop_resize(s["image"], p)
                self.items.append((crop, lab, p, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop, lab, prop, s = self.items[idx]
        x = torch.from_numpy(crop).permute(2, 0, 1)
        return x, lab, prop, s


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    # train crop classifier
    ds = CropDataset(n=120, seed=1)
    # flatten to tensors
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    model = TinyCropNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(25):
        model.train()
        # mini batches
        perm = torch.randperm(len(xs))
        total = 0.0
        for i in range(0, len(xs), 64):
            idx = perm[i : i + 64]
            logits = model(xs[idx])
            loss = F.cross_entropy(logits, ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        if epoch % 5 == 0:
            print(f"epoch {epoch} loss={total:.3f}")

    # evaluate on 40 images: proposals → classify → NMS
    rng = np.random.default_rng(9)
    cfg = SynthConfig(img_size=64, min_size=10, max_size=24)
    preds_rcnn, preds_sw, gts = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(40):
            s = generate_one(rng, cfg)
            props, _ = make_proposals(s["image"], s["boxes"], s["labels"], rng, n_bg=12)
            # also add a few pure random (missed selective search)
            crops = torch.stack(
                [torch.from_numpy(crop_resize(s["image"], p)).permute(2, 0, 1) for p in props]
            )
            logits = model(crops)
            prob = torch.softmax(logits, dim=-1).numpy()
            # drop bg class 3
            scores = prob[:, :3].max(axis=1)
            labels = prob[:, :3].argmax(axis=1)
            keep = scores > 0.45
            boxes = props[keep]
            scores = scores[keep]
            labels = labels[keep]
            if len(boxes):
                k = nms_xyxy(boxes, scores, 0.4)
                boxes, scores, labels = boxes[k], scores[k], labels[k]
            preds_rcnn.append({"boxes": boxes, "scores": scores, "labels": labels})

            b, sc, lab, _ = sliding_window_detect(s["image"])
            if len(b):
                k = nms_xyxy(b, sc, 0.3)
                b, sc, lab = b[k], sc[k], lab[k]
            preds_sw.append({"boxes": b, "scores": sc, "labels": lab})
            gts.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})

    m_rcnn = coco_style_ap(preds_rcnn, gts)
    m_sw = coco_style_ap(preds_sw, gts)
    analysis = {
        "rcnn_style_proposals_plus_cnn": m_rcnn.as_dict(),
        "sliding_window_baseline_fs01": m_sw.as_dict(),
        "delta_ap50": m_rcnn.ap50 - m_sw.ap50,
        "findings": [
            "R-CNN decouples WHERE (proposals) from WHAT (CNN on crop).",
            "Far fewer CNN forwards than sliding windows if proposals are sparse.",
            "Proposal quality is destiny: missing GT regions ⇒ recall hard-caps (classic R-CNN pain).",
            "Still no shared computation across crops — Fast R-CNN fixes that next.",
        ],
        "disease_fixed_vs_fs01": "exhaustive per-window deep scoring",
        "disease_remaining": "redundant CNN per crop + external proposal module",
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"R-CNN-lite AP50={m_rcnn.ap50:.3f} | sliding-window AP50={m_sw.ap50:.3f}")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
