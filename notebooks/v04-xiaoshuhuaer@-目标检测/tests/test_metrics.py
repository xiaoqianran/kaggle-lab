#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metrics import average_precision_pr, box_giou_xyxy, box_iou_xyxy, coco_style_ap, nms_xyxy


def test_iou_identical():
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert abs(box_iou_xyxy(b, b)[0, 0] - 1.0) < 1e-9


def test_iou_no_overlap():
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[2.0, 2.0, 3.0, 3.0]])
    assert box_iou_xyxy(a, b)[0, 0] == 0.0


def test_iou_partial():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[1.0, 1.0, 3.0, 3.0]])
    # inter=1, union=4+4-1=7
    assert abs(box_iou_xyxy(a, b)[0, 0] - 1.0 / 7.0) < 1e-9


def test_giou_less_equal_iou():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[1.0, 1.0, 3.0, 3.0]])
    iou = box_iou_xyxy(a, b)[0, 0]
    giou = box_giou_xyxy(a, b)[0, 0]
    assert giou <= iou + 1e-9


def test_nms_suppresses_overlap():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms_xyxy(boxes, scores, iou_thr=0.5)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_ap_perfect():
    gt = [{"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "labels": np.array([0])}]
    pred = [
        {
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
            "scores": np.array([0.99]),
            "labels": np.array([0]),
        }
    ]
    m = coco_style_ap(pred, gt)
    assert m.ap50 > 0.99
    assert m.ap > 0.99


def test_ap_pr_envelope():
    # synthetic PR
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 0.5, 0.25])
    ap = average_precision_pr(rec, prec)
    assert 0.0 <= ap <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all metrics tests passed")
