"""From-scratch softmax / linear / kNN classifiers (P1)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    n = logits.shape[0]
    p = softmax_rows(logits)
    return float(-np.log(p[np.arange(n), y] + 1e-12).mean())


class SoftmaxClassifier:
    """Multinomial logistic regression trained with SGD (numpy)."""

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1, reg: float = 1e-4):
        self.W = 0.001 * np.random.randn(n_features, n_classes)
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.reg = reg

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        n = X.shape[0]
        scores = self.logits(X)
        p = softmax_rows(scores)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * self.reg * np.sum(self.W**2)
        p[np.arange(n), y] -= 1.0
        p /= n
        dW = X.T @ p + self.reg * self.W
        db = p.sum(axis=0)
        return float(loss), dW, db

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 256) -> list:
        history = []
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            total = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                loss, dW, db = self.loss_and_grad(X[idx], y[idx])
                self.W -= self.lr * dW
                self.b -= self.lr * db
                total += loss
                steps += 1
            history.append(total / max(steps, 1))
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax_rows(self.logits(X))


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.X is not None and self.y is not None
        X = X.astype(np.float32)
        # chunked L2 for memory
        preds = []
        bs = 256
        for i in range(0, len(X), bs):
            q = X[i : i + bs]
            # ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q x^T
            q2 = (q**2).sum(axis=1, keepdims=True)
            x2 = (self.X**2).sum(axis=1)
            d = q2 + x2 - 2 * q @ self.X.T
            nn = np.argpartition(d, self.k, axis=1)[:, : self.k]
            labels = self.y[nn]
            # majority vote
            out = []
            for row in labels:
                vals, counts = np.unique(row, return_counts=True)
                out.append(vals[counts.argmax()])
            preds.append(np.array(out))
        return np.concatenate(preds)
