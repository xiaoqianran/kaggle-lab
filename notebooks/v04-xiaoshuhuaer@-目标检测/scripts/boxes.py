"""Box representations and geometric transforms for detection protocol."""
from __future__ import annotations

from typing import Tuple

import numpy as np

Array = np.ndarray


def xyxy_to_xywh(boxes: Array) -> Array:
    """xyxy -> xywh (x,y,w,h) with w=x2-x1, h=y2-y1."""
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0]
    out[..., 1] = boxes[..., 1]
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def xywh_to_xyxy(boxes: Array) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0]
    out[..., 1] = boxes[..., 1]
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def xyxy_to_cxcywh(boxes: Array) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    out[..., 0] = boxes[..., 0] + 0.5 * out[..., 2]
    out[..., 1] = boxes[..., 1] + 0.5 * out[..., 3]
    return out


def cxcywh_to_xyxy(boxes: Array) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - 0.5 * boxes[..., 2]
    out[..., 1] = boxes[..., 1] - 0.5 * boxes[..., 3]
    out[..., 2] = boxes[..., 0] + 0.5 * boxes[..., 2]
    out[..., 3] = boxes[..., 1] + 0.5 * boxes[..., 3]
    return out


def clip_boxes_xyxy(boxes: Array, width: float, height: float) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
    return boxes


def box_area_xyxy(boxes: Array) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.maximum(0.0, boxes[..., 2] - boxes[..., 0]) * np.maximum(
        0.0, boxes[..., 3] - boxes[..., 1]
    )


def horizontal_flip_xyxy(boxes: Array, width: float) -> Array:
    """Flip boxes horizontally; keep xyxy ordering."""
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    x1 = width - boxes[..., 2]
    x2 = width - boxes[..., 0]
    boxes[..., 0] = x1
    boxes[..., 2] = x2
    return boxes


def scale_xyxy(boxes: Array, sx: float, sy: float) -> Array:
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    boxes[..., 0] *= sx
    boxes[..., 2] *= sx
    boxes[..., 1] *= sy
    boxes[..., 3] *= sy
    return boxes


def encode_boxes_cxcywh(
    anchors_cxcywh: Array, gt_cxcywh: Array, eps: float = 1e-7
) -> Array:
    """Standard R-CNN style offsets tx,ty,tw,th relative to anchors."""
    a = np.asarray(anchors_cxcywh, dtype=np.float64)
    g = np.asarray(gt_cxcywh, dtype=np.float64)
    tx = (g[..., 0] - a[..., 0]) / np.maximum(a[..., 2], eps)
    ty = (g[..., 1] - a[..., 1]) / np.maximum(a[..., 3], eps)
    tw = np.log(np.maximum(g[..., 2], eps) / np.maximum(a[..., 2], eps))
    th = np.log(np.maximum(g[..., 3], eps) / np.maximum(a[..., 3], eps))
    return np.stack([tx, ty, tw, th], axis=-1)


def decode_boxes_cxcywh(
    anchors_cxcywh: Array, deltas: Array, eps: float = 1e-7
) -> Array:
    a = np.asarray(anchors_cxcywh, dtype=np.float64)
    d = np.asarray(deltas, dtype=np.float64)
    cx = d[..., 0] * a[..., 2] + a[..., 0]
    cy = d[..., 1] * a[..., 3] + a[..., 1]
    w = np.exp(d[..., 2]) * np.maximum(a[..., 2], eps)
    h = np.exp(d[..., 3]) * np.maximum(a[..., 3], eps)
    return np.stack([cx, cy, w, h], axis=-1)


def make_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    scales: Tuple[float, ...] = (32.0, 64.0, 128.0),
    ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> Array:
    """Generate anchors on a feature grid in image xyxy coordinates."""
    anchors = []
    for y in range(feat_h):
        for x in range(feat_w):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for s in scales:
                for r in ratios:
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.asarray(anchors, dtype=np.float64)
