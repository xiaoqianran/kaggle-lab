"""Save simple PNG visualizations without display backend."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_panels(
    path: Path,
    images: Sequence[Tuple[str, np.ndarray]],
    cmap_depth: str = "magma",
    ncols: int = 3,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (title, img) in zip(axes, images):
        img = np.asarray(img)
        ax.set_title(title, fontsize=10)
        if img.ndim == 2:
            im = ax.imshow(img, cmap=cmap_depth)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif img.ndim == 3 and img.shape[-1] in (3, 4):
            x = img
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1)
            ax.imshow(x)
        else:
            ax.imshow(img.squeeze(), cmap=cmap_depth)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def depth_to_rgb(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    x = (d - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    # simple magma-ish via matplotlib
    cm = plt.get_cmap("magma")
    return cm(x)[..., :3]
