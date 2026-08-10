#!/usr/bin/env python3
"""FS11: Anchor matching stats vs center (anchor-free) assignment coverage.

Does not retrain two full detectors; measures:
- how many anchors become positive under IoU assign
- center-cell coverage for GT (anchor-free)
Shows the hyperparameter surface of anchors vs simplicity of centers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment import assign_iou  # noqa: E402
from boxes import make_anchors  # noqa: E402
from data_synth import SynthConfig, SynthDetDataset  # noqa: E402
from fs_viz import save_bar_compare  # noqa: E402
from metrics import set_seed  # noqa: E402

OUT = ROOT / "results" / "fs11_anchor_free"
OUT.mkdir(parents=True, exist_ok=True)


def center_assign_coverage(gt_boxes, stride=8, feat=8, img=64):
    """Fraction of GTs that land in a unique center cell."""
    cells = set()
    hit = 0
    for b in gt_boxes:
        cx = 0.5 * (b[0] + b[2])
        cy = 0.5 * (b[1] + b[3])
        gx, gy = int(cx / stride), int(cy / stride)
        if 0 <= gx < feat and 0 <= gy < feat:
            hit += 1
            cells.add((gx, gy))
    return hit / max(len(gt_boxes), 1), len(cells)


def main() -> None:
    set_seed(0)
    ds = SynthDetDataset(80, SynthConfig(img_size=64, min_size=8, max_size=28), seed=0)
    anchors = make_anchors(8, 8, stride=8, scales=(16, 32), ratios=(0.5, 1.0, 2.0))
    pos_fracs = []
    center_cov = []
    multi_gt_cell = 0
    for s in ds.samples:
        asg = assign_iou(anchors, s["boxes"], pos_thr=0.5, neg_thr=0.3)
        pos_fracs.append(float((asg["labels"] == 1).mean()))
        cov, ncell = center_assign_coverage(s["boxes"])
        center_cov.append(cov)
        if len(s["boxes"]) > ncell:
            multi_gt_cell += 1

    # thr sweep
    thr_rows = []
    for thr in [0.4, 0.5, 0.6, 0.7]:
        pf = []
        for s in ds.samples:
            asg = assign_iou(anchors, s["boxes"], pos_thr=thr, neg_thr=0.3)
            pf.append(float((asg["labels"] == 1).mean()))
        thr_rows.append({"pos_thr": thr, "mean_pos_frac": float(np.mean(pf))})

    analysis = {
        "step": "FS11",
        "n_anchors": int(len(anchors)),
        "mean_anchor_pos_frac@0.5": float(np.mean(pos_fracs)),
        "mean_center_gt_coverage": float(np.mean(center_cov)),
        "images_with_center_collision": multi_gt_cell,
        "pos_thr_sweep": thr_rows,
        "new_capability": "Anchor-free center assign removes scales/ratios knobs",
        "vs_previous": "FS07/anchors need thr & shapes; FS08/11 center is simpler but collisions exist",
        "findings": [
            f"Anchors={len(anchors)}; mean positive fraction={np.mean(pos_fracs):.4f}",
            f"Center assign covers {np.mean(center_cov)*100:.1f}% GT centers in-grid",
            "Raising pos_thr shrinks positives — classic anchor hyperparam pain",
        ],
    }
    save_bar_compare(
        OUT / "pos_thr.png",
        [str(r["pos_thr"]) for r in thr_rows],
        [r["mean_pos_frac"] for r in thr_rows],
        "FS11 · anchor positive fraction vs IoU thr",
        ylabel="pos frac",
    )
    save_bar_compare(
        OUT / "coverage.png",
        ["anchor_pos_frac", "center_cov"],
        [analysis["mean_anchor_pos_frac@0.5"], analysis["mean_center_gt_coverage"]],
        "FS11 · assign regimes",
        ylabel="fraction",
    )
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2)[:900])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
