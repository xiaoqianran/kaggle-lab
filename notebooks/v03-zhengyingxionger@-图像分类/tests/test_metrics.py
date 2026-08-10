"""P0 acceptance: metrics + leakage + seed stability."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.metrics import (  # noqa: E402
    detect_split_leakage,
    evaluate_classification,
    expected_calibration_error,
    macro_f1_score,
    softmax,
    topk_accuracy,
)
from scripts.seed import set_seed  # noqa: E402


def test_topk_perfect():
    logits = np.eye(5) * 10.0
    y = np.arange(5)
    assert topk_accuracy(logits, y, k=1) == 1.0
    assert topk_accuracy(logits, y, k=5) == 1.0


def test_topk_none():
    logits = np.zeros((4, 3))
    logits[:, 0] = 5  # always predict 0
    y = np.array([1, 1, 2, 2])
    assert topk_accuracy(logits, y, k=1) == 0.0


def test_macro_f1_balanced():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    assert abs(macro_f1_score(y, p, n_classes=2) - 1.0) < 1e-9


def test_ece_perfect_calibration():
    # one-hot confident correct preds -> ECE ~ 0
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(probs, y, n_bins=10) < 1e-9


def test_ece_overconfident_wrong():
    probs = np.array([[0.95, 0.05], [0.9, 0.1]])
    y = np.array([1, 1])  # model always prefers class 0
    ece = expected_calibration_error(probs, y, n_bins=10)
    assert ece > 0.5


def test_evaluate_bundle():
    rng = np.random.RandomState(0)
    logits = rng.randn(200, 10)
    y = rng.randint(0, 10, size=200)
    rep = evaluate_classification(logits, y)
    assert 0.0 <= rep.top1 <= 1.0
    assert rep.top5 is not None and rep.top1 <= rep.top5 + 1e-9
    assert 0.0 <= rep.macro_f1 <= 1.0
    assert 0.0 <= rep.ece <= 1.0
    assert rep.n == 200


def test_leakage_detector():
    clean = detect_split_leakage([1, 2, 3], [4, 5], [6])
    assert clean["leaky"] is False
    dirty = detect_split_leakage([1, 2, 3], [3, 4], [4, 5])
    assert dirty["leaky"] is True
    assert dirty["train_val_overlap"] == 1
    assert dirty["val_test_overlap"] == 1


def test_seed_reproducible_numpy():
    set_seed(123)
    a = np.random.randn(10)
    set_seed(123)
    b = np.random.randn(10)
    assert np.allclose(a, b)


def test_softmax_rows_sum():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x)
    assert np.allclose(p.sum(axis=1), 1.0)
