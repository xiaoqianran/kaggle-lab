"""Visualization helpers for from-scratch ladder."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

VIZ = Path(__file__).resolve().parent / "results" / "viz"
VIZ.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> str:
    path = VIZ / name
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(Path(__file__).resolve().parent.parent))


def plot_images_grid(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    name: str = "grid.png",
    cmap: str = "gray",
    ncols: int = 5,
) -> str:
    n = len(images)
    ncols = min(ncols, n) if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            im = images[i]
            if im.ndim == 2:
                ax.imshow(im, cmap=cmap)
            else:
                ax.imshow(np.clip(im, 0, 1) if im.max() <= 1.5 else im.astype(np.uint8))
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    return savefig(name)


def plot_curve(ys: Sequence[float], name: str, title: str, ylabel: str = "loss") -> str:
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(ys) + 1), ys, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return savefig(name)


def plot_curves(
    series: dict,
    name: str,
    title: str,
    ylabel: str = "value",
) -> str:
    plt.figure(figsize=(5.5, 3.2))
    for label, ys in series.items():
        plt.plot(range(1, len(ys) + 1), ys, marker="o", label=label)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return savefig(name)


def plot_bar(labels: Sequence[str], values: Sequence[float], name: str, title: str) -> str:
    plt.figure(figsize=(max(5, 0.7 * len(labels)), 3.2))
    plt.bar(labels, values, color="#3b82f6")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.ylabel("top1")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    return savefig(name)


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], name: str, title: str) -> str:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title(title)
    return savefig(name)


def plot_templates(W: np.ndarray, class_names: Sequence[str], name: str) -> str:
    # W: [D, C] or [C, H, W]
    if W.ndim == 2:
        c = W.shape[1]
        side = int(np.sqrt(W.shape[0]))
        mats = [W[:, i].reshape(side, side) for i in range(c)]
    else:
        mats = [W[i] for i in range(W.shape[0])]
        c = len(mats)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < c:
            m = mats[i]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            ax.imshow(m, cmap="magma")
            ax.set_title(class_names[i], fontsize=8)
    return savefig(name)
