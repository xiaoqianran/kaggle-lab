"""Synthetic detection datasets: colored shapes on canvas (CPU-friendly, learnable)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

Array = np.ndarray


@dataclass
class SynthConfig:
    img_size: int = 64
    n_classes: int = 3  # 0=red square, 1=green circle-ish, 2=blue rect
    min_size: int = 8
    max_size: int = 28
    max_objects: int = 3
    seed: int = 42


def _draw_square(img: Array, x1: int, y1: int, x2: int, y2: int, color: Tuple[float, float, float]):
    img[y1:y2, x1:x2] = color


def _draw_rect(img: Array, x1: int, y1: int, x2: int, y2: int, color: Tuple[float, float, float]):
    img[y1:y2, x1:x2] = color


def _draw_blob(img: Array, cx: int, cy: int, r: int, color: Tuple[float, float, float]):
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[mask] = color


COLORS = {
    0: (1.0, 0.15, 0.15),  # red square
    1: (0.15, 0.9, 0.2),  # green circle
    2: (0.2, 0.35, 1.0),  # blue rect
}


def generate_one(rng: np.random.Generator, cfg: SynthConfig) -> Dict:
    s = cfg.img_size
    img = np.zeros((s, s, 3), dtype=np.float32)
    # background noise
    img += rng.normal(0.08, 0.02, size=img.shape).astype(np.float32)
    n = int(rng.integers(1, cfg.max_objects + 1))
    boxes = []
    labels = []
    for _ in range(n):
        cls = int(rng.integers(0, cfg.n_classes))
        w = int(rng.integers(cfg.min_size, cfg.max_size + 1))
        h = int(rng.integers(cfg.min_size, cfg.max_size + 1))
        if cls == 1:  # circle: square box around radius
            r = max(cfg.min_size // 2, w // 2)
            cx = int(rng.integers(r + 1, s - r - 1))
            cy = int(rng.integers(r + 1, s - r - 1))
            _draw_blob(img, cx, cy, r, COLORS[cls])
            x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        elif cls == 0:
            x1 = int(rng.integers(0, s - w))
            y1 = int(rng.integers(0, s - h))
            x2, y2 = x1 + w, y1 + h
            side = min(w, h)
            x2, y2 = x1 + side, y1 + side
            _draw_square(img, x1, y1, x2, y2, COLORS[cls])
        else:
            x1 = int(rng.integers(0, s - w))
            y1 = int(rng.integers(0, s - h))
            x2, y2 = x1 + w, y1 + h
            _draw_rect(img, x1, y1, x2, y2, COLORS[cls])
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
    img = np.clip(img, 0, 1)
    return {
        "image": img,
        "boxes": np.asarray(boxes, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }


class SynthDetDataset(Dataset):
    def __init__(self, n: int, cfg: SynthConfig | None = None, seed: int = 0):
        self.cfg = cfg or SynthConfig()
        self.n = n
        self.rng = np.random.default_rng(seed)
        # pre-generate for reproducibility
        self.samples = [generate_one(self.rng, self.cfg) for _ in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = torch.from_numpy(s["image"]).permute(2, 0, 1)  # C,H,W
        target = {
            "boxes": torch.from_numpy(s["boxes"]),
            "labels": torch.from_numpy(s["labels"]),
        }
        return img, target


def collate_det(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


def predictions_from_gt_jitter(
    dataset: SynthDetDataset,
    iou_noise: float = 0.0,
    score_base: float = 0.9,
    drop_prob: float = 0.0,
    fp_per_image: int = 0,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """Build pred/gt lists for metric protocol experiments."""
    rng = np.random.default_rng(seed)
    preds, gts = [], []
    for s in dataset.samples:
        gt_boxes = s["boxes"].astype(np.float64)
        gt_labels = s["labels"].astype(np.int64)
        gts.append({"boxes": gt_boxes, "labels": gt_labels})
        keep = []
        for i, box in enumerate(gt_boxes):
            if rng.random() < drop_prob:
                continue
            b = box.copy()
            if iou_noise > 0:
                # shift/scale a bit
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
                w = b[2] - b[0]
                h = b[3] - b[1]
                cx += rng.normal(0, iou_noise * w)
                cy += rng.normal(0, iou_noise * h)
                w *= 1 + rng.normal(0, iou_noise)
                h *= 1 + rng.normal(0, iou_noise)
                b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            keep.append((b, gt_labels[i], score_base + rng.uniform(-0.05, 0.05)))
        # false positives
        s_img = dataset.cfg.img_size
        for _ in range(fp_per_image):
            w = rng.integers(6, 16)
            h = rng.integers(6, 16)
            x1 = rng.integers(0, s_img - w)
            y1 = rng.integers(0, s_img - h)
            keep.append(
                (
                    np.array([x1, y1, x1 + w, y1 + h], dtype=np.float64),
                    int(rng.integers(0, dataset.cfg.n_classes)),
                    rng.uniform(0.2, 0.6),
                )
            )
        if keep:
            boxes = np.stack([k[0] for k in keep])
            labels = np.array([k[1] for k in keep], dtype=np.int64)
            scores = np.array([k[2] for k in keep], dtype=np.float64)
        else:
            boxes = np.zeros((0, 4))
            labels = np.zeros((0,), dtype=np.int64)
            scores = np.zeros((0,))
        preds.append({"boxes": boxes, "scores": scores, "labels": labels})
    return preds, gts
