"""Reproducibility helpers (P0)."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix Python / NumPy / PyTorch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for stable augmentations."""
    worker_seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int = 42):
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
