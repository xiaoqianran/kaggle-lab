"""Classification metrics for research-grade evaluation (P0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MetricReport:
    top1: float
    top5: Optional[float]
    macro_f1: float
    ece: float
    n: int
    per_class_recall: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, float]:
        d = {
            "top1": float(self.top1),
            "macro_f1": float(self.macro_f1),
            "ece": float(self.ece),
            "n": float(self.n),
        }
        if self.top5 is not None:
            d["top5"] = float(self.top5)
        return d


def _as_1d_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.reshape(-1)
    return y.astype(np.int64)


def topk_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    k: int = 1,
) -> float:
    """Fraction of samples whose true label is in top-k predicted classes."""
    logits = np.asarray(logits)
    targets = _as_1d_int(targets)
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets length mismatch")
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> float:
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE with equal-width confidence bins (Naeini et al.)."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = _as_1d_int(targets)
    if probs.ndim != 2:
        raise ValueError("probs must be [N, C]")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def evaluate_classification(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    top5: bool = True,
    n_bins: int = 15,
) -> MetricReport:
    logits = np.asarray(logits)
    targets = _as_1d_int(targets)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    n_classes = logits.shape[1]
    t1 = topk_accuracy(logits, targets, k=1)
    t5 = topk_accuracy(logits, targets, k=5) if top5 and n_classes >= 5 else None
    f1 = macro_f1_score(targets, pred, n_classes=n_classes)
    ece = expected_calibration_error(probs, targets, n_bins=n_bins)
    cm = confusion_matrix(targets, pred, n_classes)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return MetricReport(
        top1=t1,
        top5=t5,
        macro_f1=f1,
        ece=ece,
        n=len(targets),
        per_class_recall=recall,
    )


def detect_split_leakage(
    train_ids: Sequence,
    val_ids: Sequence,
    test_ids: Optional[Sequence] = None,
) -> Dict[str, object]:
    """Report ID overlap between splits (data leakage detector)."""
    tr, va = set(train_ids), set(val_ids)
    te = set(test_ids) if test_ids is not None else set()
    tv = tr & va
    tt = tr & te
    vt = va & te
    return {
        "train_val_overlap": len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap": len(vt),
        "leaky": bool(tv or tt or vt),
        "examples": {
            "train_val": list(tv)[:5],
            "train_test": list(tt)[:5],
            "val_test": list(vt)[:5],
        },
    }
