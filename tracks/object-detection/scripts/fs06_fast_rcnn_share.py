#!/usr/bin/env python3
"""FS06: Shared backbone vs per-crop forwards — compute story.

Uses many fake proposals (~500) to simulate classic R-CNN cost.
Reports: n_forwards proxy + wall clock + AP.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_synth import SynthConfig, SynthDetDataset, generate_one  # noqa: E402
from fs05_rcnn_crops import TinyCropNet, crop_resize, make_proposals  # noqa: E402
from fs_viz import save_bar_compare, save_det_panel  # noqa: E402
from metrics import coco_style_ap, nms_xyxy, set_seed  # noqa: E402
from models import CenterNetLite, centernet_loss, decode_centernet  # noqa: E402

OUT = ROOT / "results" / "fs06_fast_rcnn_share"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_seed(0)
    torch.manual_seed(0)
    cfg = SynthConfig(img_size=64)
    rng = np.random.default_rng(1)
    xs, ys = [], []
    for _ in range(80):
        s = generate_one(rng, cfg)
        props, labs = make_proposals(s["image"], s["boxes"], s["labels"], rng)
        for p, lab in zip(props, labs):
            xs.append(torch.from_numpy(crop_resize(s["image"], p)).permute(2, 0, 1))
            ys.append(lab)
    xs, ys = torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    crop_net = TinyCropNet()
    opt = torch.optim.Adam(crop_net.parameters(), lr=1e-3)
    for epoch in range(15):
        perm = torch.randperm(len(xs))
        for i in range(0, len(xs), 64):
            idx = perm[i : i + 64]
            loss = F.cross_entropy(crop_net(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    train_ds = SynthDetDataset(160, cfg, seed=3)
    shared = CenterNetLite()
    opt2 = torch.optim.Adam(shared.parameters(), lr=1e-3)
    for epoch in range(12):
        shared.train()
        for i in range(0, len(train_ds), 16):
            batch = [train_ds[j] for j in range(i, min(i + 16, len(train_ds)))]
            imgs = torch.stack([b[0] for b in batch])
            targets = [b[1] for b in batch]
            loss, _ = centernet_loss(shared(imgs), targets)
            opt2.zero_grad()
            loss.backward()
            opt2.step()

    val_n = 30
    n_props_sim = 500  # classic selective-search order of magnitude (scaled)
    preds_rcnn, preds_share, gts = [], [], []
    crop_net.eval()
    t0 = time.perf_counter()
    total_crop_forwards = 0
    with torch.no_grad():
        for i in range(val_n):
            s = generate_one(np.random.default_rng(50 + i), cfg)
            # simulate many proposals: base good props + pad random
            props, _ = make_proposals(s["image"], s["boxes"], s["labels"], np.random.default_rng(60 + i), n_bg=20)
            while len(props) < n_props_sim:
                extra = np.random.default_rng(70 + i).integers(0, 50, size=(n_props_sim - len(props), 4)).astype(float)
                extra[:, 2] = extra[:, 0] + 12
                extra[:, 3] = extra[:, 1] + 12
                props = np.vstack([props, extra])
            props = props[:n_props_sim]
            total_crop_forwards += len(props)
            # batch crops for speed but count as N forwards conceptually
            crops = torch.stack(
                [torch.from_numpy(crop_resize(s["image"], p)).permute(2, 0, 1) for p in props[:64]]
            )  # score first 64 for AP (good ones first)
            # re-make with good props only for AP quality
            props_q, _ = make_proposals(s["image"], s["boxes"], s["labels"], np.random.default_rng(60 + i))
            crops_q = torch.stack(
                [torch.from_numpy(crop_resize(s["image"], p)).permute(2, 0, 1) for p in props_q]
            )
            # time the large simulated batch in chunks
            for st in range(0, n_props_sim, 64):
                chunk = []
                for p in props[st : st + 64]:
                    chunk.append(torch.from_numpy(crop_resize(s["image"], p)).permute(2, 0, 1))
                _ = crop_net(torch.stack(chunk))
            prob = torch.softmax(crop_net(crops_q), 1).numpy()
            scores, labels = prob[:, :3].max(1), prob[:, :3].argmax(1)
            keep = scores > 0.45
            boxes, scores, labels = props_q[keep], scores[keep], labels[keep]
            if len(boxes):
                k = nms_xyxy(boxes, scores, 0.4)
                boxes, scores, labels = boxes[k], scores[k], labels[k]
            preds_rcnn.append({"boxes": boxes, "scores": scores, "labels": labels})
            gts.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})
    t_rcnn = time.perf_counter() - t0

    t1 = time.perf_counter()
    shared.eval()
    gts2 = []
    with torch.no_grad():
        for i in range(val_n):
            s = generate_one(np.random.default_rng(50 + i), cfg)
            img = torch.from_numpy(s["image"]).permute(2, 0, 1).unsqueeze(0)
            _ = shared(img)  # 1 forward
            d = decode_centernet(shared(img), conf_thr=0.25)[0]
            boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
            if len(boxes):
                k = nms_xyxy(boxes, scores, 0.5)
                boxes, scores, labels = boxes[k], scores[k], labels[k]
            preds_share.append({"boxes": boxes, "scores": scores, "labels": labels})
            gts2.append({"boxes": s["boxes"].astype(float), "labels": s["labels"]})
    t_share = time.perf_counter() - t1

    m_r = coco_style_ap(preds_rcnn, gts)
    m_s = coco_style_ap(preds_share, gts2)
    analysis = {
        "step": "FS06",
        "rcnn_style": {
            **m_r.as_dict(),
            "seconds_for_val": t_rcnn,
            "sec_per_image": t_rcnn / val_n,
            "cnn_forwards_per_image": n_props_sim,
            "total_forwards": total_crop_forwards,
        },
        "shared_backbone": {
            **m_s.as_dict(),
            "seconds_for_val": t_share,
            "sec_per_image": t_share / val_n,
            "cnn_forwards_per_image": 1,
        },
        "forward_ratio": n_props_sim,
        "wall_speedup": t_rcnn / max(t_share, 1e-9),
        "new_capability": "Amortize features: 1 backbone / image vs N crops",
        "vs_previous": "FS05 correctness of crop-CNN; FS06 system cost of sharing",
        "findings": [
            f"R-CNN-style ~{n_props_sim} crop CNN forwards/image vs shared 1 forward",
            f"Wall-clock speedup on this CPU run ≈ {t_rcnn/max(t_share,1e-9):.1f}x",
            "AP differs (different heads); the lesson is compute structure",
        ],
    }
    save_bar_compare(
        OUT / "forwards.png",
        ["R-CNN crops", "shared"],
        [n_props_sim, 1],
        "FS06 · CNN forwards per image",
        ylabel="forwards",
    )
    save_bar_compare(
        OUT / "time_ms.png",
        ["R-CNN crops", "shared"],
        [t_rcnn / val_n * 1000, t_share / val_n * 1000],
        "FS06 · ms per image",
        ylabel="ms",
    )
    s = generate_one(np.random.default_rng(0), cfg)
    with torch.no_grad():
        d = decode_centernet(
            shared(torch.from_numpy(s["image"]).permute(2, 0, 1).unsqueeze(0)), conf_thr=0.25
        )[0]
    boxes, scores, labels = d["boxes"].numpy(), d["scores"].numpy(), d["labels"].numpy()
    if len(boxes):
        k = nms_xyxy(boxes, scores, 0.5)
        boxes, scores, labels = boxes[k], scores[k], labels[k]
    save_det_panel(
        OUT / "shared_pred.png",
        s["image"],
        s["boxes"],
        s["labels"],
        boxes,
        scores,
        labels,
        "FS06 shared-backbone preds",
    )
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"forwards {n_props_sim}:1  wall_speedup={analysis['wall_speedup']:.1f}x")
    print("wrote", OUT)
    assert analysis["wall_speedup"] > 1.5


if __name__ == "__main__":
    main()
