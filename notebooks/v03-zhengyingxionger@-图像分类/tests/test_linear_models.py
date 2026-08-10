"""P1 acceptance: numpy softmax classifier learns separable data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.linear_models import KNNClassifier, SoftmaxClassifier, cross_entropy  # noqa: E402
from scripts.seed import set_seed  # noqa: E402


def _blobs(n=300, d=8, c=3, seed=0):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for i in range(c):
        X_list.append(rng.randn(n // c, d) * 0.4 + i * 2.0)
        y_list.append(np.full(n // c, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X.astype(np.float64), y.astype(np.int64)


def test_softmax_learns_blobs():
    set_seed(0)
    X, y = _blobs()
    clf = SoftmaxClassifier(X.shape[1], 3, lr=0.5, reg=1e-4)
    hist = clf.fit(X, y, epochs=40, batch_size=64)
    acc = (clf.predict(X) == y).mean()
    assert hist[-1] < hist[0]
    assert acc > 0.9


def test_cross_entropy_minimum_on_correct():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    y = np.array([0, 1])
    assert cross_entropy(logits, y) < 0.01


def test_knn_predicts():
    set_seed(1)
    X, y = _blobs(n=150, d=4, c=2)
    knn = KNNClassifier(k=3).fit(X, y)
    acc = (knn.predict(X) == y).mean()
    assert acc > 0.85
