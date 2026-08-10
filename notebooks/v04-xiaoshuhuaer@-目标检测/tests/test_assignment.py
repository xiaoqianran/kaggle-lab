#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment import hungarian_match


def test_hungarian_bijection():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
    gl = np.array([0, 1])
    pred = gt.copy()
    logits = np.zeros((2, 3))
    logits[0, 0] = 5
    logits[1, 1] = 5
    m = hungarian_match(pred, logits, gt, gl, num_classes=3)
    assert set(m["gt_idx"].tolist()) == {0, 1}
    assert len(m["pred_idx"]) == 2


if __name__ == "__main__":
    test_hungarian_bijection()
    print("ok test_hungarian_bijection")
    print("all assignment tests passed")
