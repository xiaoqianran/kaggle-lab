#!/usr/bin/env python3
"""Image classification from-scratch ladder runner.

Usage:
  python from_scratch/run_ladder.py
  python from_scratch/run_ladder.py S00 S02 S05 S08
  python from_scratch/run_ladder.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.linear_models import KNNClassifier, SoftmaxClassifier  # noqa: E402
from scripts.metrics import evaluate_classification  # noqa: E402
from scripts.models import MiniCNN, MiniConvNeXt, MiniViT, ResNetCIFAR  # noqa: E402
from scripts.seed import set_seed  # noqa: E402
from scripts.train_utils import (  # noqa: E402
    CLASS_NAMES,
    ToRGB,
    collect_logits,
    fit_classifier,
    get_device,
)

FS_RESULTS = Path(__file__).resolve().parent / "results"
FS_RESULTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

MEAN = (0.2860, 0.2860, 0.2860)
STD = (0.3530, 0.3530, 0.3530)


def save_step(step_id: str, payload: dict) -> Path:
    path = FS_RESULTS / f"{step_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  [saved] {path.name}")
    return path


def fashion_raw(train: bool = True):
    return datasets.FashionMNIST(str(DATA), train=train, download=True)


def fashion_tensor_loaders(
    train_n: int,
    test_n: int,
    batch_size: int = 128,
    seed: int = 0,
    strong_aug: bool = False,
):
    if strong_aug:
        train_tf = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(MEAN, STD),
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
                transforms.Normalize(MEAN, STD),
            ]
        )
    test_tf = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            ToRGB(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    tr = datasets.FashionMNIST(str(DATA), train=True, download=True, transform=train_tf)
    te = datasets.FashionMNIST(str(DATA), train=False, download=True, transform=test_tf)
    rng = np.random.RandomState(seed)
    tr = Subset(tr, rng.choice(len(tr), min(train_n, len(tr)), replace=False).tolist())
    te = Subset(te, rng.choice(len(te), min(test_n, len(te)), replace=False).tolist())
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=batch_size, shuffle=False),
    )


def flatten_subset(n_train=2000, n_test=500, seed=0):
    tr, te = fashion_raw(True), fashion_raw(False)
    rng = np.random.RandomState(seed)
    tri = rng.choice(len(tr), n_train, replace=False)
    tei = rng.choice(len(te), n_test, replace=False)
    Xtr = np.stack([np.array(tr[i][0], dtype=np.float32).reshape(-1) / 255.0 for i in tri])
    ytr = np.array([tr[i][1] for i in tri], dtype=np.int64)
    Xte = np.stack([np.array(te[i][0], dtype=np.float32).reshape(-1) / 255.0 for i in tei])
    yte = np.array([te[i][1] for i in tei], dtype=np.int64)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mean) / std, ytr, (Xte - mean) / std, yte


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_S00() -> dict:
    set_seed(0)
    ds = fashion_raw(True)
    x, y = ds[0]
    arr = np.array(x, dtype=np.float32)
    # class counts
    # sample 2000 labels for hist
    idx = np.random.RandomState(0).choice(len(ds), 2000, replace=False)
    labels = [ds[i][1] for i in idx]
    counts = {CLASS_NAMES[c]: int(labels.count(c)) for c in range(10)}
    out = {
        "step": "S00",
        "title": "pixels as tensors",
        "input": "FashionMNIST PIL image",
        "observable": {
            "sample_shape_hw": list(arr.shape),
            "sample_min_max": [float(arr.min()), float(arr.max())],
            "sample_mean": float(arr.mean()),
            "label": int(y),
            "label_name": CLASS_NAMES[y],
            "class_hist_n2000": counts,
            "rgb32_note": "downstream steps resize to 32 and repeat to 3ch",
        },
        "metric": None,
        "metric_kind": "none",
        "takeaway": "Model input is a numeric tensor + discrete label; always check shape/scale first.",
    }
    save_step("S00", out)
    return out


def step_S01() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte = flatten_subset(1500, 400, seed=0)
    # random projection for speed
    R = np.random.RandomState(0).randn(Xtr.shape[1], 64).astype(np.float32) / 8.0
    knn = KNNClassifier(k=3).fit(Xtr @ R, ytr)
    pred = knn.predict(Xte @ R)
    top1 = float((pred == yte).mean())
    chance = 0.1
    out = {
        "step": "S01",
        "title": "kNN baseline (no learning)",
        "input": "flattened pixels -> 64d random projection",
        "observable": {"top1": top1, "chance": chance, "k": 3},
        "metric": top1,
        "delta_vs_prev": top1 - chance,
        "takeaway": "Data has structure: kNN >> chance without any training.",
    }
    save_step("S01", out)
    return out


def step_S02() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte = flatten_subset(3000, 800, seed=0)
    clf = SoftmaxClassifier(Xtr.shape[1], 10, lr=0.2, reg=1e-3)
    hist = clf.fit(Xtr, ytr, epochs=25, batch_size=256)
    logits = clf.logits(Xte)
    rep = evaluate_classification(logits, yte)
    # class templates energy
    templates = clf.W.T.reshape(10, 28, 28)
    t_energy = np.abs(templates).mean(axis=(1, 2)).tolist()
    out = {
        "step": "S02",
        "title": "linear softmax from scratch",
        "input": "standardized flat pixels",
        "observable": {
            "loss_start": hist[0],
            "loss_end": hist[-1],
            "metrics": rep.as_dict(),
            "template_abs_energy_per_class": t_energy,
        },
        "metric": rep.top1,
        "takeaway": "Learning finds class templates (rows of W); still no spatial prior.",
    }
    save_step("S02", out)
    return out


class MLP(nn.Module):
    def __init__(self, d_in=28 * 28, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def step_S03() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte = flatten_subset(3000, 800, seed=0)
    device = get_device()
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    losses = []
    for ep in range(15):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i : i + 128]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(Xtr_t))
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device)).cpu().numpy()
    rep = evaluate_classification(logits, yte)
    out = {
        "step": "S03",
        "title": "MLP on flat pixels",
        "observable": {"loss_curve": losses, "metrics": rep.as_dict()},
        "metric": rep.top1,
        "takeaway": "Nonlinearity helps some; still treats pixels as bag of values → need convolution.",
    }
    save_step("S03", out)
    return out


def step_S04() -> dict:
    set_seed(0)
    ds = fashion_raw(False)
    img, y = ds[0]
    x = torch.tensor(np.array(img, dtype=np.float32) / 255.0).view(1, 1, 28, 28)
    # hand-specified Sobel-like kernels + learnable contrast via conv
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    fx = F.conv2d(x, sobel_x, padding=1)
    fy = F.conv2d(x, sobel_y, padding=1)
    edge = torch.sqrt(fx**2 + fy**2)
    # param count comparison: dense vs conv for 3x3 mapping 1->8
    dense_params = 28 * 28 * 8 * 28 * 28  # absurd full connect same spatial
    conv_params = 8 * 1 * 3 * 3
    out = {
        "step": "S04",
        "title": "hand convolution + feature maps",
        "input": f"one test image label={CLASS_NAMES[y]}",
        "observable": {
            "input_shape": list(x.shape),
            "edge_map_mean": float(edge.mean()),
            "edge_map_max": float(edge.max()),
            "conv_params_1to8_k3": conv_params,
            "naive_dense_same_spatial_params": dense_params,
            "param_ratio_dense_over_conv": dense_params / conv_params,
        },
        "metric": float(edge.mean()),  # not accuracy — signal energy
        "metric_kind": "feature_energy",
        "takeaway": "Conv reuses local filters: orders-of-magnitude fewer params than dense spatial maps; edges emerge from simple kernels.",
    }
    save_step("S04", out)
    return out


def step_S05() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2500, 800, seed=0)
    r = fit_classifier(MiniCNN(10), tr, te, epochs=3, lr=0.05, exp_id="fs_s05_cnn")
    out = {
        "step": "S05",
        "title": "shallow CNN (LeNet spirit)",
        "observable": r["metrics"],
        "metric": r["metrics"]["top1"],
        "takeaway": "End-to-end CNN uses spatial structure; should beat flat MLP under similar budget.",
    }
    save_step("S05", out)
    return out


class CNNNoBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def step_S06() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=1)
    with_bn = fit_classifier(MiniCNN(10), tr, te, epochs=3, lr=0.1, exp_id="fs_s06_bn")
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=1)
    no_bn = fit_classifier(CNNNoBN(10), tr, te, epochs=3, lr=0.1, exp_id="fs_s06_nobn")
    out = {
        "step": "S06",
        "title": "BatchNorm ablation",
        "observable": {"with_bn": with_bn["metrics"], "no_bn": no_bn["metrics"]},
        "metric": with_bn["metrics"]["top1"],
        "delta_internal": with_bn["metrics"]["top1"] - no_bn["metrics"]["top1"],
        "takeaway": "BN changes trainability/lr robustness; compare curves not only final top1.",
    }
    save_step("S06", out)
    return out


class PlainDeep(nn.Module):
    """Deep stack WITHOUT residuals — for degradation demo."""

    def __init__(self, num_classes=10, depth=8, ch=32):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def step_S08() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2500, 800, seed=2)
    res = fit_classifier(ResNetCIFAR(10), tr, te, epochs=3, lr=0.05, exp_id="fs_s08_res")
    set_seed(0)
    tr, te = fashion_tensor_loaders(2500, 800, seed=2)
    plain = fit_classifier(PlainDeep(10, depth=10, ch=32), tr, te, epochs=3, lr=0.05, exp_id="fs_s08_plain")
    out = {
        "step": "S08",
        "title": "residual vs plain deep",
        "observable": {"residual": res["metrics"], "plain_deep": plain["metrics"]},
        "metric": res["metrics"]["top1"],
        "delta_internal": res["metrics"]["top1"] - plain["metrics"]["top1"],
        "takeaway": "Residual path keeps deep nets optimizable — key 2015 breakthrough.",
    }
    save_step("S08", out)
    return out


class DWSeparableCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def dw_pw(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.BatchNorm2d(cin),
                nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            dw_pw(32, 64, 2),
            dw_pw(64, 128, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def step_S09() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=3)
    dw = fit_classifier(DWSeparableCNN(10), tr, te, epochs=3, lr=0.05, exp_id="fs_s09_dw")
    n_dw = sum(p.numel() for p in DWSeparableCNN(10).parameters())
    n_cnn = sum(p.numel() for p in MiniCNN(10).parameters())
    out = {
        "step": "S09",
        "title": "depthwise separable conv",
        "observable": {
            "metrics": dw["metrics"],
            "params_dw": n_dw,
            "params_minicnn": n_cnn,
        },
        "metric": dw["metrics"]["top1"],
        "takeaway": "Factorizing conv buys efficiency; report params/latency with accuracy.",
    }
    save_step("S09", out)
    return out


def step_S10() -> dict:
    set_seed(0)
    rows = []
    for name, strong, ls, mix in [
        ("base", False, 0.0, 0.0),
        ("aug", True, 0.0, 0.0),
        ("aug_ls", True, 0.1, 0.0),
        ("aug_mix", True, 0.0, 0.2),
    ]:
        set_seed(0)
        tr, te = fashion_tensor_loaders(2000, 600, seed=4, strong_aug=strong)
        r = fit_classifier(
            MiniCNN(10), tr, te, epochs=3, lr=0.05,
            label_smoothing=ls, mixup_alpha=mix, exp_id=f"fs_s10_{name}",
        )
        rows.append({"name": name, "strong_aug": strong, "ls": ls, "mixup": mix, **r["metrics"]})
    best = max(rows, key=lambda x: x["top1"])
    out = {
        "step": "S10",
        "title": "training recipe ablation",
        "observable": {"table": rows, "best": best["name"]},
        "metric": best["top1"],
        "takeaway": "Same model, different recipe → different top1/ECE; modern gains are often recipe.",
    }
    save_step("S10", out)
    return out


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x.mean((2, 3)))
        return x * w.view(b, c, 1, 1)


class CNNSE(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.se1 = SEBlock(32)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.se2 = SEBlock(64)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        x = self.conv3(x).flatten(1)
        return self.head(x)


def step_S11() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=5)
    se = fit_classifier(CNNSE(10), tr, te, epochs=3, lr=0.05, exp_id="fs_s11_se")
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=5)
    base = fit_classifier(MiniCNN(10), tr, te, epochs=3, lr=0.05, exp_id="fs_s11_base")
    out = {
        "step": "S11",
        "title": "Squeeze-and-Excitation",
        "observable": {"se": se["metrics"], "base": base["metrics"]},
        "metric": se["metrics"]["top1"],
        "delta_internal": se["metrics"]["top1"] - base["metrics"]["top1"],
        "takeaway": "Channel attention is a cheap pre-ViT way to reweight features.",
    }
    save_step("S11", out)
    return out


def step_S12() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=6)
    vit = fit_classifier(
        MiniViT(10, depth=2, embed_dim=96, num_heads=3),
        tr, te, epochs=3, lr=0.01, exp_id="fs_s12_vit",
    )
    out = {
        "step": "S12",
        "title": "minimal Vision Transformer",
        "observable": vit["metrics"],
        "metric": vit["metrics"]["top1"],
        "takeaway": "Global attention works but is data-hungry without strong recipe/pretrain.",
    }
    save_step("S12", out)
    return out


def step_S13() -> dict:
    set_seed(0)
    tr, te = fashion_tensor_loaders(2000, 600, seed=6)
    cx = fit_classifier(MiniConvNeXt(10), tr, te, epochs=3, lr=0.03, exp_id="fs_s13_cx")
    out = {
        "step": "S13",
        "title": "Mini ConvNeXt",
        "observable": cx["metrics"],
        "metric": cx["metrics"]["top1"],
        "takeaway": "Modernized CNNs compete with ViTs; architecture is not settled on attention-only.",
    }
    save_step("S13", out)
    return out


def nt_xent(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    n = z1.size(0)
    sim = (z @ z.T) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), -1e9)
    targets = torch.cat([torch.arange(n, device=z.device) + n, torch.arange(n, device=z.device)])
    return F.cross_entropy(sim, targets)


def step_S14() -> dict:
    set_seed(0)
    device = get_device()
    from scripts.models import EncoderForSSL

    tf = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ToRGB(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    ds = fashion_raw(True)
    idx = np.random.RandomState(0).choice(len(ds), 2000, replace=False)

    class TwoView(torch.utils.data.Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            img, y = ds[idx[i]]
            return tf(img), tf(img)

    loader = DataLoader(TwoView(), batch_size=128, shuffle=True)
    enc = EncoderForSSL(MiniCNN(10), proj_dim=64).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        total = n = 0
        enc.train()
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad()
            loss = nt_xent(enc(v1), enc(v2))
            loss.backward()
            opt.step()
            total += loss.item() * v1.size(0)
            n += v1.size(0)
        losses.append(total / max(n, 1))
    # linear probe
    probe_tf = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), ToRGB(), transforms.Normalize(MEAN, STD)]
    )
    tr = datasets.FashionMNIST(str(DATA), train=True, download=True, transform=probe_tf)
    te = datasets.FashionMNIST(str(DATA), train=False, download=True, transform=probe_tf)
    enc.eval()

    def emb(split, n_s, seed):
        id_ = np.random.RandomState(seed).choice(len(split), n_s, replace=False)
        f, y = [], []
        with torch.no_grad():
            for i in id_:
                x, lab = split[i]
                f.append(enc.forward_features(x.unsqueeze(0).to(device)).cpu().numpy())
                y.append(lab)
        return np.concatenate(f), np.array(y)

    Xtr, ytr = emb(tr, 1500, 1)
    Xte, yte = emb(te, 500, 2)
    clf = SoftmaxClassifier(Xtr.shape[1], 10, lr=0.3, reg=1e-3)
    clf.fit(Xtr, ytr, epochs=25, batch_size=128)
    probe = float((clf.predict(Xte) == yte).mean())
    out = {
        "step": "S14",
        "title": "SimCLR-style contrastive SSL",
        "observable": {"ssl_loss": losses, "linear_probe_top1": probe},
        "metric": probe,
        "takeaway": "Labels optional: contrastive pretrain + linear probe measures representation quality.",
    }
    save_step("S14", out)
    return out


def step_S15() -> dict:
    set_seed(0)
    # use supervised CNN features as stand-in encoder then class prototypes
    tr, te = fashion_tensor_loaders(2500, 800, seed=7)
    device = get_device()
    model = MiniCNN(10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for _ in range(3):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in tr:
            h = model.forward_features(x.to(device)).cpu().numpy()
            feats.append(h)
            labs.append(y.numpy())
    Xtr = np.concatenate(feats)
    ytr = np.concatenate(labs)
    feats, labs = [], []
    with torch.no_grad():
        for x, y in te:
            h = model.forward_features(x.to(device)).cpu().numpy()
            feats.append(h)
            labs.append(y.numpy())
    Xte = np.concatenate(feats)
    yte = np.concatenate(labs)
    protos = np.stack([Xtr[ytr == c].mean(0) for c in range(10)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    zs = float(((Xn @ protos.T).argmax(1) == yte).mean())
    # closed-set head
    with torch.no_grad():
        logits, yt = collect_logits(model, te, device)
    closed = evaluate_classification(logits, yt).top1
    out = {
        "step": "S15",
        "title": "prototype cosine (CLIP intuition)",
        "observable": {"prototype_top1": zs, "closed_head_top1": closed},
        "metric": zs,
        "takeaway": "Cosine-to-prototype is the interface of open-vocab models; usually trails closed heads.",
    }
    save_step("S15", out)
    return out


def step_S16() -> dict:
    set_seed(0)
    device = get_device()
    # pretrain on larger subset
    tr_full, te = fashion_tensor_loaders(4000, 600, seed=8)
    backbone = MiniCNN(10).to(device)
    opt = torch.optim.SGD(backbone.parameters(), lr=0.05, momentum=0.9)
    for _ in range(3):
        backbone.train()
        for x, y in tr_full:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(backbone(x), y).backward()
            opt.step()
    # few-shot subset
    tr_few, _ = fashion_tensor_loaders(500, 600, seed=9)

    def eval_model(m):
        logits, y = collect_logits(m, te, device)
        return evaluate_classification(logits, y).top1

    # linear probe: freeze features, new head
    probe = MiniCNN(10).to(device)
    probe.load_state_dict(backbone.state_dict())
    for p in probe.features.parameters():
        p.requires_grad = False
    probe.head = nn.Linear(128, 10).to(device)
    opt = torch.optim.SGD(probe.head.parameters(), lr=0.1, momentum=0.9)
    for _ in range(5):
        probe.train()
        for x, y in tr_few:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(probe(x), y).backward()
            opt.step()
    probe_acc = eval_model(probe)

    # from scratch on few
    scratch = MiniCNN(10).to(device)
    opt = torch.optim.SGD(scratch.parameters(), lr=0.05, momentum=0.9)
    for _ in range(5):
        scratch.train()
        for x, y in tr_few:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(scratch(x), y).backward()
            opt.step()
    scratch_acc = eval_model(scratch)

    # full FT from pretrained on few
    ft = MiniCNN(10).to(device)
    ft.load_state_dict(backbone.state_dict())
    opt = torch.optim.SGD(ft.parameters(), lr=0.01, momentum=0.9)
    for _ in range(5):
        ft.train()
        for x, y in tr_few:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(ft(x), y).backward()
            opt.step()
    ft_acc = eval_model(ft)

    out = {
        "step": "S16",
        "title": "transfer protocols",
        "observable": {
            "linear_probe_fewshot": probe_acc,
            "scratch_fewshot": scratch_acc,
            "full_finetune_fewshot": ft_acc,
        },
        "metric": probe_acc,
        "takeaway": "With few labels, reusing pretrained features (probe/FT) beats training from scratch.",
    }
    save_step("S16", out)
    return out


STEPS: Dict[str, Callable[[], dict]] = {
    "S00": step_S00,
    "S01": step_S01,
    "S02": step_S02,
    "S03": step_S03,
    "S04": step_S04,
    "S05": step_S05,
    "S06": step_S06,
    "S08": step_S08,
    "S09": step_S09,
    "S10": step_S10,
    "S11": step_S11,
    "S12": step_S12,
    "S13": step_S13,
    "S14": step_S14,
    "S15": step_S15,
    "S16": step_S16,
}

# Default core path (skips S07 which is narrative-only in MAP; S08 covers depth)
DEFAULT = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]


def main():
    ap = argparse.ArgumentParser(description="Image classification from-scratch ladder")
    ap.add_argument("steps", nargs="*", help="Step ids e.g. S00 S05 S08")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k in STEPS:
            print(k, STEPS[k].__doc__ or STEPS[k].__name__)
        print("default:", " ".join(DEFAULT))
        return

    todo = args.steps or DEFAULT
    unknown = [s for s in todo if s not in STEPS]
    if unknown:
        print("Unknown steps:", unknown)
        print("Available:", list(STEPS))
        sys.exit(2)

    ladder = []
    prev_metric: Optional[float] = None
    for sid in todo:
        print(f"\n======== {sid} ========", flush=True)
        t0 = time.time()
        out = STEPS[sid]()
        elapsed = time.time() - t0
        metric = out.get("metric")
        kind = out.get("metric_kind", "top1" if out.get("step") not in {"S00", "S04"} else "other")
        out["metric_kind"] = kind
        delta = None
        if kind == "top1" and metric is not None and prev_metric is not None:
            delta = float(metric) - float(prev_metric)
            out["delta_vs_prev_metric"] = delta
        if kind == "top1" and metric is not None:
            prev_metric = float(metric)
        out["elapsed_sec"] = elapsed
        save_step(sid, out)  # rewrite with delta
        ladder.append(
            {
                "step": sid,
                "title": out.get("title"),
                "metric": metric,
                "delta_vs_prev_metric": delta,
                "takeaway": out.get("takeaway"),
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"{sid} | metric={metric} | delta={delta} | {elapsed:.1f}s | {out.get('takeaway', '')[:80]}",
            flush=True,
        )

    summary = {
        "ladder": ladder,
        "dataset": "FashionMNIST-32RGB",
        "map": "from_scratch/MAP.md",
    }
    path = FS_RESULTS / "ladder.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n[ladder complete] {path}")
    # markdown table
    lines = [
        "# From-Scratch Ladder Results",
        "",
        "| Step | Title | Metric | Δ prev | Takeaway |",
        "|------|-------|--------|--------|----------|",
    ]
    for r in ladder:
        m = f"{r['metric']:.4f}" if isinstance(r["metric"], float) else str(r["metric"])
        d = f"{r['delta_vs_prev_metric']:+.4f}" if r["delta_vs_prev_metric"] is not None else "—"
        lines.append(f"| {r['step']} | {r['title']} | {m} | {d} | {r['takeaway']} |")
    (FS_RESULTS / "LADDER_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote results/LADDER_RESULTS.md")


if __name__ == "__main__":
    main()
