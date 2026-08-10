"""Depth estimation evaluation metrics and alignment utilities.

Implements standard MDE metrics used in Eigen / KITTI / NYU protocols:
AbsRel, SqRel, RMSE, RMSE_log, SI-RMSE, delta thresholds, and
scale / scale-shift alignment for relative depth evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class MetricResult:
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    si_rmse: float
    delta1: float
    delta2: float
    delta3: float
    n_valid: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "si_rmse": self.si_rmse,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
            "n_valid": float(self.n_valid),
        }


def valid_mask(
    gt: np.ndarray,
    pred: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Boolean mask for finite, positive, capped depths (both gt and pred)."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = (
        np.isfinite(gt)
        & np.isfinite(pred)
        & (gt > min_depth)
        & (pred > min_depth)
        & (gt <= max_depth)
        & (pred <= max_depth)
    )
    return mask


def median_scale(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Median scaling: s = median(gt) / median(pred) on valid pixels."""
    if mask is None:
        mask = valid_mask(gt, pred)
    p = pred[mask]
    g = gt[mask]
    if p.size == 0:
        return 1.0
    return float(np.median(g) / (np.median(p) + 1e-12))


def least_squares_scale_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Align pred ≈ s * pred + t to gt in least-squares sense (MiDaS/DPT style).

    Solves min_{s,t} || s * pred + t - gt ||^2 on valid pixels.
    Returns (scale, shift).
    """
    if mask is None:
        mask = np.isfinite(gt) & np.isfinite(pred)
    p = pred[mask].astype(np.float64).ravel()
    g = gt[mask].astype(np.float64).ravel()
    if p.size < 2:
        return 1.0, 0.0
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    return float(sol[0]), float(sol[1])


def align_median(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    s = median_scale(pred, gt, mask)
    return pred * s


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    s, t = least_squares_scale_shift(pred, gt, mask)
    return s * pred + t


def si_rmse(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Scale-invariant RMSE on log depth (Eigen / ETHZ SI-RMSE style).

    SI-RMSE = sqrt( max(0, (1/n) sum d_i^2 - ((1/n)sum d_i)^2) )
    where d_i = log(pred_i) - log(gt_i).
    """
    if mask is None:
        mask = valid_mask(gt, pred)
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    if p.size == 0:
        return float("nan")
    d = np.log(p) - np.log(g)
    var = np.mean(d ** 2) - (np.mean(d) ** 2)
    # numerical guard: pure global scale => var ~ 0 within float noise
    return float(np.sqrt(max(var, 0.0)))


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    align: str = "none",
) -> MetricResult:
    """Compute standard depth metrics.

    align:
      - "none": use pred as-is (metric models)
      - "median": median scale (Eigen monocular protocol)
      - "least_squares": scale+shift (relative / affine-invariant)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch pred {pred.shape} vs gt {gt.shape}")

    mask = valid_mask(gt, pred, min_depth=min_depth, max_depth=max_depth)

    if align == "none":
        pred_a = pred
    elif align == "median":
        pred_a = align_median(pred, gt, mask)
        mask = valid_mask(gt, pred_a, min_depth=min_depth, max_depth=max_depth)
    elif align in ("least_squares", "scale_shift", "ss"):
        pred_a = align_scale_shift(pred, gt, mask)
        pred_a = np.clip(pred_a, min_depth, max_depth)
        mask = valid_mask(gt, pred_a, min_depth=min_depth, max_depth=max_depth)
    else:
        raise ValueError(f"unknown align mode: {align}")

    p = pred_a[mask]
    g = gt[mask]
    n = int(p.size)
    if n == 0:
        return MetricResult(
            abs_rel=float("nan"),
            sq_rel=float("nan"),
            rmse=float("nan"),
            rmse_log=float("nan"),
            si_rmse=float("nan"),
            delta1=float("nan"),
            delta2=float("nan"),
            delta3=float("nan"),
            n_valid=0,
        )

    thresh = np.maximum(p / g, g / p)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    si = si_rmse(pred_a, gt, mask)

    return MetricResult(
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        rmse=rmse,
        rmse_log=rmse_log,
        si_rmse=si,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
        n_valid=n,
    )


def silog_loss(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None, lamb: float = 0.85) -> float:
    """Scale-invariant log loss (training objective, Eigen-style variant)."""
    if mask is None:
        mask = valid_mask(gt, pred)
    p = pred[mask]
    g = gt[mask]
    if p.size == 0:
        return float("nan")
    d = np.log(p) - np.log(g)
    return float(np.mean(d ** 2) - lamb * (np.mean(d) ** 2))
