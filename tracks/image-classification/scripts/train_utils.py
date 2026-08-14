"""Shared training utilities (P2–P6). Default dataset: FashionMNIST (fast, offline-ready)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .metrics import evaluate_classification
from .seed import dataloader_generator, seed_worker, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# FashionMNIST class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToRGB:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.size(0) == 1:
            return t.repeat(3, 1, 1)
        return t


def fashion_loaders(
    *,
    batch_size: int = 128,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    strong_aug: bool = False,
    data_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """FashionMNIST resized to 32×32 RGB — ImageNet-style pipeline practice without 170MB CIFAR pull."""
    root = Path(data_root or DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    mean = (0.2860, 0.2860, 0.2860)
    std = (0.3530, 0.3530, 0.3530)
    if strong_aug:
        train_tf = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(mean, std),
            ]
        )
    test_tf = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            ToRGB(),
            transforms.Normalize(mean, std),
        ]
    )
    train_set = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=train_tf)
    test_set = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=test_tf)
    if train_subset is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_set), size=min(train_subset, len(train_set)), replace=False)
        train_set = Subset(train_set, idx.tolist())
    if test_subset is not None:
        rng = np.random.RandomState(seed + 1)
        idx = rng.choice(len(test_set), size=min(test_subset, len(test_set)), replace=False)
        test_set = Subset(test_set, idx.tolist())
    g = dataloader_generator(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, 10


# backward-compatible alias used by run_phase
cifar10_loaders = fashion_loaders


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(x.size(0), device=device)
            mixed = lam * x + (1 - lam) * x[idx]
            logits = model(mixed)
            loss = lam * F.cross_entropy(logits, y, label_smoothing=label_smoothing) + (
                1 - lam
            ) * F.cross_entropy(logits, y[idx], label_smoothing=label_smoothing)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 5,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    seed: int = 42,
    exp_id: str = "exp",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, opt, device,
            label_smoothing=label_smoothing, mixup_alpha=mixup_alpha,
        )
        logits, targets = collect_logits(model, test_loader, device)
        report = evaluate_classification(logits, targets, top5=True)
        history.append({"epoch": ep, "train_loss": tr_loss, **report.as_dict()})
        sched.step()
        print(f"  [{exp_id}] ep {ep}/{epochs} loss={tr_loss:.4f} top1={report.top1:.4f}")
    elapsed = time.time() - t0
    logits, targets = collect_logits(model, test_loader, device)
    final = evaluate_classification(logits, targets, top5=True)
    out = {
        "exp_id": exp_id,
        "device": str(device),
        "dataset": "FashionMNIST-32RGB",
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "metrics": final.as_dict(),
        "history": history,
        "meta": meta or {},
        "seed": seed,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{exp_id}.json"
    path.write_text(json.dumps(out, indent=2))
    return out
