#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment import assign_iou
from boxes import (
    decode_boxes_cxcywh,
    encode_boxes_cxcywh,
    horizontal_flip_xyxy,
    make_anchors,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
)


def test_xyxy_cxcywh_roundtrip():
    b = np.array([[1.0, 2.0, 5.0, 8.0], [0.0, 0.0, 10.0, 10.0]])
    rec = cxcywh_to_xyxy(xyxy_to_cxcywh(b))
    assert np.allclose(rec, b)


def test_encode_decode():
    a = np.array([[10.0, 10.0, 20.0, 20.0]])
    g = np.array([[12.0, 11.0, 22.0, 24.0]])
    d = encode_boxes_cxcywh(a, g)
    rec = decode_boxes_cxcywh(a, d)
    assert np.allclose(rec, g, atol=1e-9)


def test_flip():
    b = np.array([[10.0, 5.0, 30.0, 40.0]])
    assert np.allclose(horizontal_flip_xyxy(horizontal_flip_xyxy(b, 64), 64), b)


def test_assign_min_one_pos_per_gt():
    anchors = make_anchors(4, 4, stride=8, scales=(16,), ratios=(1.0,))
    gt = np.array([[5.0, 5.0, 25.0, 25.0]])
    asg = assign_iou(anchors, gt, pos_thr=0.9, neg_thr=0.3)
    assert (asg["labels"] == 1).sum() >= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all boxes tests passed")
