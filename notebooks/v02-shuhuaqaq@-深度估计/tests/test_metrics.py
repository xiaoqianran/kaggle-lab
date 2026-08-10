"""P0 acceptance tests for metrics and alignment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metrics import (  # noqa: E402
    align_median,
    align_scale_shift,
    compute_metrics,
    least_squares_scale_shift,
    median_scale,
    si_rmse,
    silog_loss,
    valid_mask,
)


def test_perfect_prediction_zero_error():
    gt = np.full((32, 48), 5.0)
    pred = gt.copy()
    m = compute_metrics(pred, gt, align="none")
    assert m.n_valid == 32 * 48
    assert m.abs_rel < 1e-8
    assert m.rmse < 1e-8
    assert m.si_rmse < 1e-8
    assert m.delta1 > 0.999


def test_median_scale_recovers_scaled_prediction():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 10.0, size=(40, 60))
    pred = gt / 3.7  # wrong scale
    s = median_scale(pred, gt)
    assert abs(s - 3.7) < 1e-6
    m = compute_metrics(pred, gt, align="median")
    assert m.abs_rel < 1e-5
    assert m.rmse < 1e-5


def test_scale_shift_alignment():
    rng = np.random.default_rng(1)
    gt = rng.uniform(0.5, 8.0, size=(50, 50))
    pred = 0.3 * gt + 1.2
    s, t = least_squares_scale_shift(pred, gt)
    # pred = 0.3*gt + 1.2 => gt = (pred - 1.2)/0.3 = (1/0.3)pred - 4
    # We fit s*pred + t ≈ gt, so s≈1/0.3, t≈-1.2/0.3
    assert abs(s - (1.0 / 0.3)) < 1e-4
    assert abs(t - (-1.2 / 0.3)) < 1e-4
    aligned = align_scale_shift(pred, gt)
    m = compute_metrics(aligned, gt, align="none")
    assert m.abs_rel < 1e-4


def test_si_rmse_invariant_to_global_scale():
    rng = np.random.default_rng(2)
    gt = rng.uniform(1.0, 20.0, size=(30, 30))
    pred = gt * 2.5
    # SI-RMSE of scaled pred vs gt should be ~0
    assert si_rmse(pred, gt) < 1e-8
    # SI-RMSE with structured error should be > 0
    pred2 = gt + 0.5
    assert si_rmse(pred2, gt) > 0.01


def test_valid_mask_caps_and_nans():
    gt = np.array([[0.0, 5.0, 100.0], [np.nan, 2.0, 3.0]])
    pred = np.array([[1.0, 5.0, 10.0], [2.0, 2.0, 3.0]])
    m = valid_mask(gt, pred, min_depth=0.1, max_depth=80.0)
    # only (0,1) and (1,1),(1,2) valid? gt 100 capped out; 0 invalid; nan invalid
    assert m[0, 0] == False
    assert m[0, 1] == True
    assert m[0, 2] == False
    assert m[1, 0] == False
    assert m[1, 1] == True
    assert m[1, 2] == True


def test_silog_loss_minimum_at_identity():
    gt = np.linspace(1, 10, 100).reshape(10, 10)
    loss0 = silog_loss(gt, gt)
    loss1 = silog_loss(gt * 1.2, gt)
    assert loss0 < 1e-10
    assert loss1 > loss0


def test_shape_mismatch_raises():
    try:
        compute_metrics(np.ones((2, 2)), np.ones((3, 3)))
        assert False, "should raise"
    except ValueError:
        pass


if __name__ == "__main__":
    test_perfect_prediction_zero_error()
    test_median_scale_recovers_scaled_prediction()
    test_scale_shift_alignment()
    test_si_rmse_invariant_to_global_scale()
    test_valid_mask_caps_and_nans()
    test_silog_loss_minimum_at_identity()
    test_shape_mismatch_raises()
    print("P0 test_metrics: ALL PASSED")
