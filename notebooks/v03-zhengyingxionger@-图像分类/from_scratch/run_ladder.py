#!/usr/bin/env python3
"""From-scratch ladder: S00→S20 continuous execution with viz + comparisons.

  python from_scratch/run_ladder.py
  python from_scratch/run_ladder.py S00 S08 S17
  python from_scratch/run_ladder.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
FS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FS))

from scripts.linear_models import KNNClassifier, SoftmaxClassifier  # noqa: E402
from scripts.metrics import evaluate_classification, expected_calibration_error, softmax  # noqa: E402
from scripts.models import MiniCNN, MiniConvNeXt, MiniViT, ResNetCIFAR  # noqa: E402
from scripts.seed import set_seed  # noqa: E402
from scripts.train_utils import CLASS_NAMES, ToRGB, collect_logits, fit_classifier, get_device  # noqa: E402
from viz_utils import (  # noqa: E402
    plot_bar,
    plot_confusion,
    plot_curve,
    plot_curves,
    plot_images_grid,
    plot_templates,
)

FS_RESULTS = FS / "results"
FS_RESULTS.mkdir(parents=True, exist_ok=True)
NOTES = FS / "notes"
NOTES.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
MEAN = (0.2860, 0.2860, 0.2860)
STD = (0.3530, 0.3530, 0.3530)

_PREV_TOP1: Optional[float] = None
_HISTORY: List[dict] = []


def save_step(step_id: str, payload: dict) -> Path:
    global _PREV_TOP1
    kind = payload.get("metric_kind", "top1")
    metric = payload.get("metric")
    if kind == "top1" and metric is not None and _PREV_TOP1 is not None:
        payload["delta_vs_prev_top1"] = float(metric) - float(_PREV_TOP1)
    if kind == "top1" and metric is not None:
        _PREV_TOP1 = float(metric)
    path = FS_RESULTS / f"{step_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    note = NOTES / f"{step_id}.md"
    note.write_text(
        f"""# {step_id} · {payload.get('title','')}

## 概念
{payload.get('concept','')}

## 输入
{payload.get('input','')}

## 输出 / 观测
```json
{json.dumps(payload.get('observable',{}), indent=2, default=str)[:2000]}
```

## 指标
- metric: `{payload.get('metric')}` ({kind})
- Δ top1 vs prev: `{payload.get('delta_vs_prev_top1')}`

## 新增能力
{payload.get('takeaway','')}

## 可视化
{payload.get('viz', [])}
"""
    )
    print(f"  [saved] {path.name} + notes/{step_id}.md")
    return payload


def fashion_raw(train: bool = True):
    return datasets.FashionMNIST(str(DATA), train=train, download=True)


def fashion_loaders(train_n, test_n, batch_size=128, seed=0, strong_aug=False):
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
        [transforms.Resize(32), transforms.ToTensor(), ToRGB(), transforms.Normalize(MEAN, STD)]
    )
    tr = datasets.FashionMNIST(str(DATA), train=True, download=True, transform=train_tf)
    te = datasets.FashionMNIST(str(DATA), train=False, download=True, transform=test_tf)
    rng = np.random.RandomState(seed)
    tr = Subset(tr, rng.choice(len(tr), min(train_n, len(tr)), replace=False).tolist())
    te = Subset(te, rng.choice(len(te), min(test_n, len(te)), replace=False).tolist())
    return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(te, batch_size=batch_size)


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
    return (Xtr - mean) / std, ytr, (Xte - mean) / std, yte, tri, tei


# ========================= STEPS =========================


def step_S00() -> dict:
    set_seed(0)
    ds = fashion_raw(True)
    imgs, titles = [], []
    for c in range(10):
        for i in range(len(ds)):
            x, y = ds[i]
            if y == c:
                imgs.append(np.array(x, dtype=np.float32) / 255.0)
                titles.append(CLASS_NAMES[c])
                break
    viz = plot_images_grid(imgs, titles, "S00_class_samples.png")
    arr = np.array(ds[0][0], dtype=np.float32)
    return save_payload(
        "S00",
        title="pixels as tensors",
        concept="图像是 HxW 数值网格；标签是离散类 id。归一化影响优化。",
        input="FashionMNIST raw PIL",
        observable={
            "shape": list(arr.shape),
            "min_max": [float(arr.min()), float(arr.max())],
            "mean": float(arr.mean()),
            "n_train": len(ds),
            "n_classes": 10,
        },
        metric=None,
        metric_kind="none",
        takeaway="能力#0：把像素当数据——先看 shape/scale/类分布，再谈模型。",
        viz=[viz],
    )


def step_S01() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte, _, tei = flatten_subset(1500, 400, 0)
    R = np.random.RandomState(0).randn(Xtr.shape[1], 64).astype(np.float32) / 8.0
    knn = KNNClassifier(k=3).fit(Xtr @ R, ytr)
    pred = knn.predict(Xte @ R)
    top1 = float((pred == yte).mean())
    # show 4 errors: query
    ds = fashion_raw(False)
    err = np.where(pred != yte)[0][:4]
    imgs, titles = [], []
    for j in err:
        imgs.append(np.array(ds[tei[j]][0], dtype=np.float32) / 255.0)
        titles.append(f"T:{CLASS_NAMES[yte[j]][:4]} P:{CLASS_NAMES[pred[j]][:4]}")
    viz = plot_images_grid(imgs, titles, "S01_knn_errors.png") if imgs else ""
    viz2 = plot_bar(["chance", "kNN"], [0.1, top1], "S01_vs_chance.png", "S01 kNN vs chance")
    return save_payload(
        "S01",
        title="kNN baseline (no learning)",
        concept="预测=最近训练样本标签；无参数，距离定义一切。",
        input="flat pixels → 64d random projection",
        observable={"top1": top1, "chance": 0.1, "k": 3, "n_errors_shown": len(imgs)},
        metric=top1,
        takeaway="能力#1：无学习基线。kNN>>随机 ⇒ 数据有可分结构，后续模型必须超过它。",
        viz=[viz, viz2],
    )


def step_S02() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte, _, _ = flatten_subset(3000, 800, 0)
    clf = SoftmaxClassifier(Xtr.shape[1], 10, lr=0.2, reg=1e-3)
    hist = clf.fit(Xtr, ytr, epochs=25, batch_size=256)
    rep = evaluate_classification(clf.logits(Xte), yte)
    viz1 = plot_curve(hist, "S02_loss.png", "S02 Softmax CE loss")
    viz2 = plot_templates(clf.W, CLASS_NAMES, "S02_class_templates.png")
    return save_payload(
        "S02",
        title="linear softmax from scratch",
        concept="z=Wx+b, p=softmax(z), 最小化交叉熵；W 的每列是类模板。",
        input="standardized flat 28x28 pixels",
        observable={"loss": [hist[0], hist[-1]], "metrics": rep.as_dict()},
        metric=rep.top1,
        takeaway="能力#2：参数学习+概率输出。仍是超平面决策，无空间结构。",
        viz=[viz1, viz2],
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(True), nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def step_S03() -> dict:
    set_seed(0)
    Xtr, ytr, Xte, yte, _, _ = flatten_subset(3000, 800, 0)
    device = get_device()
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    losses, train_accs = [], []
    for _ in range(15):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = correct = n = 0
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i : i + 128]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad()
            logits = model(xb.float())
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(idx)
        losses.append(total / n)
        train_accs.append(correct / n)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte).float().to(device)).cpu().numpy()
    rep = evaluate_classification(logits, yte)
    viz = plot_curves({"loss": losses, "train_acc": train_accs}, "S03_mlp_curves.png", "S03 MLP", "value")
    return save_payload(
        "S03",
        title="MLP on flat pixels",
        concept="多层 ReLU 提供非线性；像素仍被当作独立维度。",
        input="same flat features as S02",
        observable={"metrics": rep.as_dict(), "final_train_acc": train_accs[-1]},
        metric=rep.top1,
        takeaway="能力#3：深度非线性。常略好于线性，但过拟合与无视空间 → 需要卷积。",
        viz=[viz],
    )


def step_S04() -> dict:
    set_seed(0)
    ds = fashion_raw(False)
    # pick a boot image
    img, y = None, None
    for i in range(len(ds)):
        im, lab = ds[i]
        if lab == 9:
            img, y = im, lab
            break
    x = torch.tensor(np.array(img, dtype=np.float32) / 255.0).view(1, 1, 28, 28)
    sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sy = torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    fx, fy = F.conv2d(x, sx, padding=1), F.conv2d(x, sy, padding=1)
    edge = torch.sqrt(fx**2 + fy**2)[0, 0].numpy()
    blur_k = torch.ones(1, 1, 5, 5) / 25
    blur = F.conv2d(x, blur_k, padding=2)[0, 0].numpy()
    viz = plot_images_grid(
        [x[0, 0].numpy(), fx[0, 0].numpy(), fy[0, 0].numpy(), edge, blur],
        ["input", "sobel_x", "sobel_y", "edge", "blur"],
        "S04_feature_maps.png",
        ncols=5,
    )
    conv_p, dense_p = 8 * 1 * 3 * 3, 28 * 28 * 8 * 28 * 28
    return save_payload(
        "S04",
        title="hand-written convolution + feature maps",
        concept="局部连接+权值共享；同一滤波器扫描全图产生特征图。",
        input=f"one image class={CLASS_NAMES[y]}",
        observable={
            "edge_mean": float(edge.mean()),
            "conv_params_1to8": conv_p,
            "dense_params_same_map": dense_p,
            "ratio": dense_p / conv_p,
        },
        metric=float(edge.mean()),
        metric_kind="feature_energy",
        takeaway="能力#4：局部模式提取。参数效率数量级优势；第一次‘看见’边缘。",
        viz=[viz],
    )


def step_S05() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2500, 800, seed=0)
    # manual train to capture history
    device = get_device()
    model = MiniCNN(10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    hist = []
    for ep in range(4):
        model.train()
        tot = n = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        logits, yt = collect_logits(model, te, device)
        rep = evaluate_classification(logits, yt)
        hist.append({"loss": tot / n, "top1": rep.top1})
        print(f"  [S05] ep{ep+1} top1={rep.top1:.3f}")
    pred = logits.argmax(1)
    cm = np.zeros((10, 10), int)
    for t, p in zip(yt, pred):
        cm[int(t), int(p)] += 1
    viz1 = plot_curve([h["top1"] for h in hist], "S05_top1.png", "S05 CNN top1", "top1")
    viz2 = plot_confusion(cm, [c[:4] for c in CLASS_NAMES], "S05_cm.png", "S05 confusion")
    return save_payload(
        "S05",
        title="shallow CNN end-to-end",
        concept="Conv-Pool 堆叠 + 分类头；端到端反传。",
        input="32x32 RGB FashionMNIST loaders",
        observable={"history": hist, "final": rep.as_dict()},
        metric=rep.top1,
        takeaway="能力#5：完整卷积分类器。空间归纳偏置进入闭环。",
        viz=[viz1, viz2],
    )


class CNNNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def _train_hist(model, tr, te, epochs, lr, tag):
    device = get_device()
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    losses, tops = [], []
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        logits, yt = collect_logits(model, te, device)
        rep = evaluate_classification(logits, yt)
        losses.append(tot / n)
        tops.append(rep.top1)
        print(f"  [{tag}] ep{ep+1} loss={losses[-1]:.3f} top1={tops[-1]:.3f}")
    return model, losses, tops, rep


def step_S06() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=1)
    _, lb, tb, rb = _train_hist(MiniCNN(10), tr, te, 4, 0.1, "bn")
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=1)
    _, ln, tn, rn = _train_hist(CNNNoBN(), tr, te, 4, 0.1, "nobn")
    viz = plot_curves({"bn_top1": tb, "no_bn_top1": tn}, "S06_bn_ablation.png", "S06 BN ablation", "top1")
    return save_payload(
        "S06",
        title="BatchNorm ablation",
        concept="BN 标准化激活分布，提高 lr 容忍与收敛稳定性。",
        input="same arch family ± BN, lr=0.1",
        observable={"bn": rb.as_dict(), "no_bn": rn.as_dict(), "bn_hist": tb, "nobn_hist": tn},
        metric=rb.top1,
        takeaway="能力#6：训练稳定性技术。同架构下 BN 通常更快更稳。",
        viz=[viz],
    )


class VGGish(nn.Module):
    def __init__(self, depth=4, ch=32, num_classes=10):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for i in range(depth - 1):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True)]
            if i in (0, 2):
                layers.append(nn.MaxPool2d(2))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def step_S07() -> dict:
    set_seed(0)
    rows = []
    for depth in [3, 5, 7]:
        set_seed(0)
        tr, te = fashion_loaders(2000, 600, seed=2)
        _, losses, tops, rep = _train_hist(VGGish(depth=depth), tr, te, 3, 0.05, f"vgg{depth}")
        rows.append({"depth": depth, "top1": rep.top1, "final_loss": losses[-1], "curve": tops})
    viz = plot_bar([f"d{r['depth']}" for r in rows], [r["top1"] for r in rows], "S07_depth.png", "S07 VGG-style depth")
    return save_payload(
        "S07",
        title="3x3 stacks (VGG intuition)",
        concept="小卷积核堆叠扩大感受野并增加非线性；更深不一定更好。",
        input="depth ∈ {3,5,7} same channel",
        observable={"rows": rows},
        metric=max(r["top1"] for r in rows),
        takeaway="能力#7：深度设计直觉。过深可能退化 → 引出残差。",
        viz=[viz],
    )


class PlainDeep(nn.Module):
    def __init__(self, depth=16, ch=32):
        super().__init__()
        # No BN + very deep: classic degradation setup vs residual
        layers: List[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(depth):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(True)]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def step_S08() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2500, 800, seed=2)
    _, lr, tr_top, rr = _train_hist(ResNetCIFAR(10), tr, te, 4, 0.05, "res")
    set_seed(0)
    tr, te = fashion_loaders(2500, 800, seed=2)
    _, lp, tp, rp = _train_hist(PlainDeep(), tr, te, 4, 0.05, "plain")
    viz = plot_curves({"residual": tr_top, "plain": tp}, "S08_residual.png", "S08 residual vs plain", "top1")
    return save_payload(
        "S08",
        title="ResNet residual breakthrough",
        concept="y=F(x)+x，学习残差使极深网络可优化。",
        input="same data budget; residual vs plain deep",
        observable={"residual": rr.as_dict(), "plain": rp.as_dict()},
        metric=rr.top1,
        takeaway="能力#8：可训练的深度。2015 分水岭——现代主干默认残差。",
        viz=[viz],
    )


class DWSeparableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.BatchNorm2d(cin),
                nn.ReLU(True),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            block(32, 64, 2),
            block(64, 128, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def step_S09() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=3)
    _, _, tops, rep = _train_hist(DWSeparableCNN(), tr, te, 3, 0.05, "dw")
    n_dw = sum(p.numel() for p in DWSeparableCNN().parameters())
    n_cnn = sum(p.numel() for p in MiniCNN(10).parameters())
    viz = plot_bar(["MiniCNN_params/1e3", "DW_params/1e3"], [n_cnn / 1e3, n_dw / 1e3], "S09_params.png", "S09 param count")
    return save_payload(
        "S09",
        title="depthwise separable conv",
        concept="DW+PW 分解标准卷积，降参数/FLOPs。",
        input="MobileNet-style blocks",
        observable={"metrics": rep.as_dict(), "params_dw": n_dw, "params_cnn": n_cnn},
        metric=rep.top1,
        takeaway="能力#9：效率轴。研究也要报成本，不只报 top1。",
        viz=[viz],
    )


def step_S10() -> dict:
    rows = []
    for name, strong, ls, mix in [
        ("base", False, 0.0, 0.0),
        ("aug", True, 0.0, 0.0),
        ("aug_ls", True, 0.1, 0.0),
        ("aug_mix", True, 0.0, 0.2),
    ]:
        set_seed(0)
        tr, te = fashion_loaders(2000, 600, seed=4, strong_aug=strong)
        r = fit_classifier(
            MiniCNN(10), tr, te, epochs=3, lr=0.05,
            label_smoothing=ls, mixup_alpha=mix, exp_id=f"fs_s10_{name}",
        )
        rows.append({"name": name, **r["metrics"]})
        print(f"  [S10] {name} top1={r['metrics']['top1']:.3f}")
    best = max(rows, key=lambda x: x["top1"])
    viz = plot_bar([r["name"] for r in rows], [r["top1"] for r in rows], "S10_recipe.png", "S10 recipe ablation")
    return save_payload(
        "S10",
        title="modern training recipe",
        concept="增强/LS/Mixup 等配方与架构同样重要。",
        input="fixed MiniCNN, 4 recipes",
        observable={"table": rows, "best": best["name"]},
        metric=best["top1"],
        takeaway="能力#10：工业级训练配方。SOTA 提升常来自 recipe。",
        viz=[viz],
    )


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(True), nn.Linear(max(ch // r, 4), ch), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x.mean((2, 3)))
        return x * w.view(x.size(0), x.size(1), 1, 1)


class CNNSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2))
        self.se1 = SEBlock(32)
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2))
        self.se2 = SEBlock(64)
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = self.se1(self.c1(x))
        x = self.se2(self.c2(x))
        return self.head(self.c3(x).flatten(1))


def step_S11() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=5)
    _, _, ts, rs = _train_hist(CNNSE(), tr, te, 3, 0.05, "se")
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=5)
    _, _, tb, rb = _train_hist(MiniCNN(10), tr, te, 3, 0.05, "base")
    viz = plot_bar(["base", "SE"], [rb.top1, rs.top1], "S11_se.png", "S11 SE vs base")
    return save_payload(
        "S11",
        title="Squeeze-and-Excitation",
        concept="全局池化→FC→通道权重，轻量注意力。",
        input="CNN ± SE blocks",
        observable={"se": rs.as_dict(), "base": rb.as_dict()},
        metric=rs.top1,
        takeaway="能力#11：通道注意力。ViT 之前的实用注意形态。",
        viz=[viz],
    )


def step_S12() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=6)
    _, _, tops, rep = _train_hist(
        MiniViT(10, depth=2, embed_dim=96, num_heads=3), tr, te, 3, 0.01, "vit"
    )
    viz = plot_curve(tops, "S12_vit.png", "S12 MiniViT top1", "top1")
    return save_payload(
        "S12",
        title="minimal Vision Transformer",
        concept="图像切 patch → 序列 Transformer → cls 分类。",
        input="small data on purpose",
        observable={"metrics": rep.as_dict(), "curve": tops},
        metric=rep.top1,
        takeaway="能力#12：全局自注意力。小数据常弱于 CNN → 需配方/预训练。",
        viz=[viz],
    )


def step_S13() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=6)
    _, _, tops, rep = _train_hist(MiniConvNeXt(10), tr, te, 3, 0.03, "cx")
    # compare file from S12 if exists
    s12 = FS_RESULTS / "S12.json"
    vit_top = json.loads(s12.read_text())["metric"] if s12.exists() else None
    viz = plot_bar(
        ["ViT", "ConvNeXt"],
        [vit_top or 0.0, rep.top1],
        "S13_vs_vit.png",
        "S13 ConvNeXt vs ViT (same budget)",
    )
    return save_payload(
        "S13",
        title="ConvNeXt modern CNN",
        concept="大核 DWConv + LN + 阶段设计，把 Transformer 课搬回 CNN。",
        input="same budget as S12",
        observable={"metrics": rep.as_dict(), "vit_top1_ref": vit_top},
        metric=rep.top1,
        takeaway="能力#13：现代 CNN 双路线。注意力非唯一答案。",
        viz=[viz],
    )


def nt_xent(z1, z2, t=0.5):
    z = torch.cat([z1, z2], 0)
    n = z1.size(0)
    sim = (z @ z.T) / t
    sim = sim.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), -1e9)
    tgt = torch.cat([torch.arange(n, device=z.device) + n, torch.arange(n, device=z.device)])
    return F.cross_entropy(sim, tgt)


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

    class TV(torch.utils.data.Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            img, _ = ds[idx[i]]
            return tf(img), tf(img)

    loader = DataLoader(TV(), batch_size=128, shuffle=True)
    enc = EncoderForSSL(MiniCNN(10), 64).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    losses = []
    for ep in range(4):
        tot = n = 0
        enc.train()
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad()
            loss = nt_xent(enc(v1), enc(v2))
            loss.backward()
            opt.step()
            tot += loss.item() * v1.size(0)
            n += v1.size(0)
        losses.append(tot / n)
        print(f"  [S14] ssl ep{ep+1} loss={losses[-1]:.3f}")
    probe_tf = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), ToRGB(), transforms.Normalize(MEAN, STD)]
    )
    tr = datasets.FashionMNIST(str(DATA), True, download=True, transform=probe_tf)
    te = datasets.FashionMNIST(str(DATA), False, download=True, transform=probe_tf)
    enc.eval()

    def emb(split, ns, seed):
        id_ = np.random.RandomState(seed).choice(len(split), ns, replace=False)
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
    viz = plot_curve(losses, "S14_ssl_loss.png", "S14 NT-Xent loss")
    viz2 = plot_bar(["chance", "linear_probe"], [0.1, probe], "S14_probe.png", "S14 linear probe")
    return save_payload(
        "S14",
        title="SimCLR-style contrastive SSL",
        concept="双视图正样本，batch 内负样本；投影头+NT-Xent。",
        input="unlabeled two-view augmentations",
        observable={"ssl_loss": losses, "linear_probe_top1": probe},
        metric=probe,
        takeaway="能力#14：无标签表征学习。linear probe 是标准度量。",
        viz=[viz, viz2],
    )


def step_S15() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(2500, 800, seed=7)
    device = get_device()
    model, _, _, rep = _train_hist(MiniCNN(10), tr, te, 3, 0.05, "sup")
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in tr:
            feats.append(model.forward_features(x.to(device)).cpu().numpy())
            labs.append(y.numpy())
    Xtr, ytr = np.concatenate(feats), np.concatenate(labs)
    feats, labs = [], []
    with torch.no_grad():
        for x, y in te:
            feats.append(model.forward_features(x.to(device)).cpu().numpy())
            labs.append(y.numpy())
    Xte, yte = np.concatenate(feats), np.concatenate(labs)
    protos = np.stack([Xtr[ytr == c].mean(0) for c in range(10)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    zs = float(((Xn @ protos.T).argmax(1) == yte).mean())
    viz = plot_bar(["closed_head", "prototype"], [rep.top1, zs], "S15_proto.png", "S15 closed vs prototype")
    return save_payload(
        "S15",
        title="prototype cosine (CLIP interface)",
        concept="图特征与类原型余弦相似度分类；开放词汇接口的雏形。",
        input="supervised features + class mean prototypes",
        observable={"closed_top1": rep.top1, "prototype_top1": zs},
        metric=zs,
        takeaway="能力#15：开放集接口。换原型即可扩类（真 CLIP 用文本塔）。",
        viz=[viz],
    )


def step_S16() -> dict:
    set_seed(0)
    device = get_device()
    tr_full, te = fashion_loaders(4000, 600, seed=8)
    backbone, _, _, _ = _train_hist(MiniCNN(10), tr_full, te, 3, 0.05, "pre")
    tr_few, _ = fashion_loaders(500, 600, seed=9)

    def acc(m):
        logits, y = collect_logits(m, te, device)
        return evaluate_classification(logits, y).top1

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
    a_probe = acc(probe)

    scratch = MiniCNN(10).to(device)
    opt = torch.optim.SGD(scratch.parameters(), lr=0.05, momentum=0.9)
    for _ in range(5):
        scratch.train()
        for x, y in tr_few:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(scratch(x), y).backward()
            opt.step()
    a_scratch = acc(scratch)

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
    a_ft = acc(ft)
    viz = plot_bar(
        ["scratch", "probe", "finetune"],
        [a_scratch, a_probe, a_ft],
        "S16_transfer.png",
        "S16 few-shot transfer protocols",
    )
    return save_payload(
        "S16",
        title="transfer learning protocols",
        concept="linear probe / full FT / scratch；少样本时复用预训练。",
        input="pretrain 4k then few-shot 500",
        observable={"scratch": a_scratch, "probe": a_probe, "finetune": a_ft},
        metric=max(a_probe, a_ft),
        takeaway="能力#16：迁移协议。少标签下 probe/FT 通常碾压从头训。",
        viz=[viz],
    )


# ---------- Frontier S17–S20 ----------


class TinyMAE(nn.Module):
    """Toy MAE: patchify, mask, reconstruct pixels with tiny transformer."""

    def __init__(self, img=32, patch=4, dim=64, depth=2, heads=4, mask_ratio=0.75):
        super().__init__()
        self.patch = patch
        self.n = (img // patch) ** 2
        self.dim = dim
        self.mask_ratio = mask_ratio
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.pred = nn.Linear(dim, patch * patch * 3)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def patchify(self, x):
        # x: B,3,H,W -> B,N,P*P*3
        p = self.patch
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(b, -1, p * p * c)

    def forward(self, x):
        # encode visible patches only (simplified: encode all then zero mask tokens effect via loss mask)
        tokens = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        b, n, d = tokens.shape
        n_mask = int(n * self.mask_ratio)
        noise = torch.rand(b, n, device=x.device)
        ids = torch.argsort(noise, dim=1)
        ids_keep = ids[:, n_mask:]
        ids_mask = ids[:, :n_mask]
        keep = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        encoded = self.encoder(keep)
        # scatter back
        full = torch.zeros_like(tokens)
        full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, d), encoded)
        pred = self.pred(full)
        target = self.patchify(x)
        # loss on masked only
        loss = 0.0
        for i in range(b):
            m = ids_mask[i]
            loss = loss + F.mse_loss(pred[i, m], target[i, m])
        loss = loss / b
        return loss, pred, target, ids_mask


def step_S17() -> dict:
    set_seed(0)
    device = get_device()
    tr, _ = fashion_loaders(512, 128, seed=10)
    model = TinyMAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    # overfit a few batches then general
    for ep in range(6):
        model.train()
        tot = n = 0
        for x, _ in tr:
            x = x.to(device)
            opt.zero_grad()
            loss, _, _, _ = model(x)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        losses.append(tot / max(n, 1))
        print(f"  [S17] mae ep{ep+1} loss={losses[-1]:.4f}")
    # visualize reconstruction for one image
    model.eval()
    x, _ = next(iter(tr))
    x = x[:1].to(device)
    with torch.no_grad():
        loss, pred, target, ids_mask = model(x)
    # reconstruct image from pred patches (masked shown as pred, visible as original)
    p = 4
    recon = target.clone()
    recon[0, ids_mask[0]] = pred[0, ids_mask[0]]
    # unpatchify
    def unpatch(patches):
        b, n, pp = patches.shape
        h = w = int(np.sqrt(n))
        patches = patches.reshape(b, h, w, p, p, 3)
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        return patches.reshape(b, 3, h * p, w * p)

    orig = x[0].cpu()
    # denorm roughly for viz
    def den(t):
        t = t.cpu().numpy().transpose(1, 2, 0)
        t = t * np.array(STD) + np.array(MEAN)
        return np.clip(t, 0, 1)

    rec = unpatch(recon)[0]
    masked = target.clone()
    masked[0, ids_mask[0]] = 0.5  # gray holes
    mask_img = unpatch(masked)[0]
    viz = plot_images_grid(
        [den(orig), den(mask_img), den(rec)],
        ["original", "masked_target", "recon"],
        "S17_mae_recon.png",
        ncols=3,
    )
    viz2 = plot_curve(losses, "S17_mae_loss.png", "S17 toy MAE loss")
    return save_payload(
        "S17",
        title="toy MAE (masked autoencoder)",
        concept="高比例 mask patch，编码器只看可见块，预测被遮像素——自监督重建。",
        input="Fashion 32RGB batches, mask_ratio=0.75",
        observable={"loss_curve": losses, "final_loss": losses[-1]},
        metric=float(1.0 / (1.0 + losses[-1])),  # proxy: higher better as loss drops
        metric_kind="recon_quality_proxy",
        takeaway="能力#17：掩码图像建模。MAE 路线与对比学习并列的 SSL 支柱。",
        viz=[viz, viz2],
    )


def step_S18() -> dict:
    """DeiT-style recipe: bare ViT vs ViT + strong aug + longer (here more epochs) + higher wd."""
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=11, strong_aug=False)
    _, _, t_bare, r_bare = _train_hist(
        MiniViT(10, depth=2, embed_dim=96, num_heads=3), tr, te, 2, 0.01, "vit_bare"
    )
    set_seed(0)
    tr, te = fashion_loaders(2000, 600, seed=11, strong_aug=True)
    device = get_device()
    model = MiniViT(10, depth=2, embed_dim=96, num_heads=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    tops = []
    for ep in range(5):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y, label_smoothing=0.1).backward()
            opt.step()
        logits, yt = collect_logits(model, te, device)
        rep = evaluate_classification(logits, yt)
        tops.append(rep.top1)
        print(f"  [S18] deit-ish ep{ep+1} top1={rep.top1:.3f}")
    viz = plot_curves({"bare_vit": t_bare, "deit_recipe": tops}, "S18_deit_recipe.png", "S18 DeiT-style recipe", "top1")
    return save_payload(
        "S18",
        title="DeiT-style data-efficient ViT recipe",
        concept="ViT 靠强增强、LS、AdamW、更长训在小数据上可战 CNN。",
        input="bare ViT 2ep vs aug+LS+AdamW 5ep",
        observable={"bare": r_bare.as_dict(), "deit_ish_top1": tops[-1], "curves": {"bare": t_bare, "deit": tops}},
        metric=tops[-1],
        takeaway="能力#18：数据高效 ViT 配方。架构+配方缺一不可。",
        viz=[viz],
    )


class MiniCLIP(nn.Module):
    """Dual encoder: image CNN + text embedding table for class names (from-scratch CLIP)."""

    def __init__(self, n_classes=10, dim=64):
        super().__init__()
        self.img = MiniCNN(dim)  # logits dim as embed
        # repurpose: use features
        self.img.head = nn.Linear(128, dim)
        self.txt = nn.Embedding(n_classes, dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def encode_image(self, x):
        return F.normalize(self.img(x), dim=-1)

    def encode_text(self, y):
        return F.normalize(self.txt(y), dim=-1)

    def forward(self, x, y):
        zi = self.encode_image(x)
        zt = self.encode_text(y)
        scale = self.logit_scale.exp().clamp(1e-3, 100)
        logits = scale * zi @ zt.T  # B x B when y is batch labels as ids... use all class texts
        return zi, zt, scale


def step_S19() -> dict:
    set_seed(0)
    device = get_device()
    tr, te = fashion_loaders(3000, 800, seed=12, strong_aug=True)
    model = MiniCLIP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    for ep in range(5):
        model.train()
        tot = n = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            zi = model.encode_image(x)
            # text for all classes
            all_y = torch.arange(10, device=device)
            zt = model.encode_text(all_y)
            scale = model.logit_scale.exp().clamp(1e-3, 100)
            logits = scale * zi @ zt.T  # B x 10
            loss = F.cross_entropy(logits, y)
            # also symmetric batch contrastive among images with matching labels soft — skip, CE enough for toy
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        losses.append(tot / n)
        print(f"  [S19] clip ep{ep+1} loss={losses[-1]:.3f}")
    # zero-shot style: encode image, compare to class embeddings
    model.eval()
    correct = total = 0
    with torch.no_grad():
        zt = model.encode_text(torch.arange(10, device=device))
        for x, y in te:
            x, y = x.to(device), y.to(device)
            zi = model.encode_image(x)
            pred = (zi @ zt.T).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    zs = correct / total
    # supervised linear head baseline on same budget from S05-like
    viz = plot_curve(losses, "S19_clip_loss.png", "S19 MiniCLIP loss")
    viz2 = plot_bar(["miniCLIP_zeroshot"], [zs], "S19_zs.png", f"S19 zero-shot-ish top1={zs:.3f}")
    return save_payload(
        "S19",
        title="MiniCLIP dual-encoder (from scratch)",
        concept="图像塔+文本(类)塔共享空间；推理时用余弦对齐，无需分类头。",
        input="class id embeddings as text tower surrogate",
        observable={"loss": losses, "zero_shot_top1": zs},
        metric=zs,
        takeaway="能力#19：多模态对齐接口。真 CLIP 换文本编码器即可扩展任意类名。",
        viz=[viz, viz2],
    )


def step_S20() -> dict:
    set_seed(0)
    tr, te = fashion_loaders(3000, 1000, seed=13, strong_aug=True)
    model, _, tops, rep = _train_hist(ResNetCIFAR(10), tr, te, 4, 0.05, "final")
    device = get_device()
    logits, y = collect_logits(model, te, device)
    probs = softmax(logits)
    ece = expected_calibration_error(probs, y)
    # OOD: gaussian noise images
    ood_logits = []
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            noise = torch.randn(128, 3, 32, 32, device=device)
            # normalize-ish
            ood_logits.append(model(noise).cpu().numpy())
    ood_logits = np.concatenate(ood_logits)
    ood_conf = softmax(ood_logits).max(1)
    id_conf = probs.max(1)
    # simple OOD score: 1 - max prob; AUROC-like separation via mean conf gap
    conf_gap = float(id_conf.mean() - ood_conf.mean())
    pred = logits.argmax(1)
    cm = np.zeros((10, 10), int)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    viz1 = plot_confusion(cm, [c[:4] for c in CLASS_NAMES], "S20_cm.png", "S20 final confusion")
    viz2 = plot_bar(
        ["top1", "macro_f1", "ece", "conf_gap"],
        [rep.top1, rep.macro_f1, ece, max(conf_gap, 0)],
        "S20_protocol.png",
        "S20 modern eval protocol",
    )
    # confidence hist
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 3))
    plt.hist(id_conf, bins=20, alpha=0.6, label="ID max-prob")
    plt.hist(ood_conf, bins=20, alpha=0.6, label="OOD noise max-prob")
    plt.legend()
    plt.title("S20 confidence ID vs OOD")
    from viz_utils import savefig

    viz3 = savefig("S20_ood_conf.png")
    return save_payload(
        "S20",
        title="modern evaluation protocol",
        concept="Top-1 + macro-F1 + ECE + 简单 OOD 置信度分离——研究级评测最小集。",
        input="ResNet final model + Gaussian noise OOD",
        observable={
            "metrics": rep.as_dict(),
            "ece": ece,
            "id_mean_conf": float(id_conf.mean()),
            "ood_mean_conf": float(ood_conf.mean()),
            "conf_gap": conf_gap,
        },
        metric=rep.top1,
        takeaway="能力#20：完整评测协议。只有 top1 不够，校准与 OOD 同属现代系统。",
        viz=[viz1, viz2, viz3],
    )


def save_payload(step, title, concept, input, observable, metric, takeaway, viz=None, metric_kind="top1"):
    payload = {
        "step": step,
        "title": title,
        "concept": concept,
        "input": input,
        "observable": observable,
        "metric": metric,
        "metric_kind": metric_kind,
        "takeaway": takeaway,
        "viz": viz or [],
        "new_capability": takeaway,
    }
    save_step(step, payload)
    return payload


STEPS: Dict[str, Callable[[], dict]] = {
    "S00": step_S00,
    "S01": step_S01,
    "S02": step_S02,
    "S03": step_S03,
    "S04": step_S04,
    "S05": step_S05,
    "S06": step_S06,
    "S07": step_S07,
    "S08": step_S08,
    "S09": step_S09,
    "S10": step_S10,
    "S11": step_S11,
    "S12": step_S12,
    "S13": step_S13,
    "S14": step_S14,
    "S15": step_S15,
    "S16": step_S16,
    "S17": step_S17,
    "S18": step_S18,
    "S19": step_S19,
    "S20": step_S20,
}

DEFAULT = list(STEPS.keys())


def rebuild_summary(ladder_rows: List[dict]):
    summary = {
        "ladder": ladder_rows,
        "dataset": "FashionMNIST-32RGB",
        "map": "from_scratch/MAP.md",
        "complete": [r["step"] for r in ladder_rows],
    }
    (FS_RESULTS / "ladder.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# From-Scratch Ladder Results",
        "",
        "| Step | Title | Metric | Δ top1 | New capability |",
        "|------|-------|--------|--------|----------------|",
    ]
    for r in ladder_rows:
        m = r.get("metric")
        ms = f"{m:.4f}" if isinstance(m, float) else str(m)
        d = r.get("delta_vs_prev_top1")
        ds = f"{d:+.4f}" if isinstance(d, float) else "—"
        lines.append(f"| {r['step']} | {r.get('title','')} | {ms} | {ds} | {r.get('takeaway','')} |")
    (FS_RESULTS / "LADDER_RESULTS.md").write_text("\n".join(lines) + "\n")


def update_progress(done: List[str], current: str = ""):
    path = ROOT / "PROGRESS.md"
    base = path.read_text() if path.exists() else "# PROGRESS\n"
    # rewrite from-scratch section
    marker = "## From Scratch 阶梯执行"
    block = f"""
## From Scratch 阶梯执行

- **地图:** `from_scratch/MAP.md`
- **运行:** `python from_scratch/run_ladder.py`
- **已完成:** {', '.join(done) if done else '(none)'}
- **当前:** {current or 'idle'}
- **可视化:** `from_scratch/results/viz/`
- **分步笔记:** `from_scratch/notes/`
- **总表:** `from_scratch/results/LADDER_RESULTS.md`

| Step | Status |
|------|--------|
""" + "\n".join(
        f"| {s} | {'DONE' if s in done else ('RUN' if s == current else 'TODO')} |" for s in DEFAULT
    )
    if marker in base:
        pre = base.split(marker)[0].rstrip()
        path.write_text(pre + "\n\n" + block + "\n")
    else:
        path.write_text(base.rstrip() + "\n\n" + block + "\n")


def main():
    global _PREV_TOP1, _HISTORY
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--continue-from", type=str, default=None, help="start from this step id")
    args = ap.parse_args()
    if args.list:
        for k in STEPS:
            print(k)
        return

    todo = args.steps or DEFAULT
    if args.continue_from:
        if args.continue_from not in DEFAULT:
            raise SystemExit(f"unknown {args.continue_from}")
        todo = DEFAULT[DEFAULT.index(args.continue_from) :]

    # restore prev top1 from last completed top1 step if partial
    _PREV_TOP1 = None
    _HISTORY = []
    done = []
    for sid in todo:
        if sid not in STEPS:
            print("skip unknown", sid)
            continue
        print(f"\n======== {sid} ========", flush=True)
        update_progress(done, sid)
        t0 = time.time()
        out = STEPS[sid]()
        # re-read saved for delta
        saved = json.loads((FS_RESULTS / f"{sid}.json").read_text())
        elapsed = time.time() - t0
        saved["elapsed_sec"] = elapsed
        (FS_RESULTS / f"{sid}.json").write_text(json.dumps(saved, indent=2, default=str))
        row = {
            "step": sid,
            "title": saved.get("title"),
            "metric": saved.get("metric"),
            "delta_vs_prev_top1": saved.get("delta_vs_prev_top1"),
            "takeaway": saved.get("takeaway"),
            "elapsed_sec": elapsed,
            "viz": saved.get("viz"),
        }
        _HISTORY.append(row)
        done.append(sid)
        rebuild_summary(_HISTORY)
        update_progress(done, "")
        print(
            f"{sid} DONE metric={saved.get('metric')} delta={saved.get('delta_vs_prev_top1')} "
            f"{elapsed:.1f}s | {saved.get('takeaway','')[:70]}",
            flush=True,
        )

    print("\nALL REQUESTED STEPS COMPLETE:", done)
    print("See from_scratch/results/LADDER_RESULTS.md and results/viz/")


if __name__ == "__main__":
    main()
