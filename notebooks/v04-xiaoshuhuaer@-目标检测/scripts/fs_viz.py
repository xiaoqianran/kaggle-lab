"""Save detection visualizations as PNG (no GUI)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# avoid display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]


def save_det_panel(
    path: Path,
    image: np.ndarray,
    gt_boxes: Optional[np.ndarray] = None,
    gt_labels: Optional[np.ndarray] = None,
    pred_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    title: str = "",
    max_pred: int = 30,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    img = np.clip(image, 0, 1)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    if gt_boxes is not None and len(gt_boxes):
        for i, b in enumerate(gt_boxes):
            x1, y1, x2, y2 = b
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            lab = int(gt_labels[i]) if gt_labels is not None else -1
            ax.text(x1, max(0, y1 - 1), f"GT{lab}", color="lime", fontsize=7)
    if pred_boxes is not None and len(pred_boxes):
        order = np.arange(len(pred_boxes))
        if pred_scores is not None:
            order = np.argsort(-np.asarray(pred_scores))[:max_pred]
        for j in order:
            b = pred_boxes[j]
            x1, y1, x2, y2 = b
            c = COLORS[int(pred_labels[j]) % len(COLORS)] if pred_labels is not None else "red"
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=c, facecolor="none"
            )
            ax.add_patch(rect)
            sc = f"{pred_scores[j]:.2f}" if pred_scores is not None else ""
            lab = int(pred_labels[j]) if pred_labels is not None else -1
            ax.text(x1, y2 + 1, f"{lab}:{sc}", color=c, fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_bar_compare(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str = "AP50",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color=["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
