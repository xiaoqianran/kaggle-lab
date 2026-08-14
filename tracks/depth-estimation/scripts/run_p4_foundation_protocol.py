#!/usr/bin/env python3
"""P4: foundation-model *protocol* experiment without multi-GB weights.

Simulates relative-depth predictors (affine-invariant) vs metric predictors and
shows evaluation must match representation. Optionally tries torch.hub MiDaS if online.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from synth_data import make_scene  # noqa: E402

OUT = ROOT / "results" / "p4_foundation"
OUT.mkdir(parents=True, exist_ok=True)


def relative_depth_from_metric(depth: np.ndarray, mode: str = "disparity") -> np.ndarray:
    if mode == "disparity":
        return 1.0 / np.maximum(depth, 1e-3)
    if mode == "log":
        return np.log(np.maximum(depth, 1e-3))
    if mode == "neg_depth":
        return -depth
    raise ValueError(mode)


def main() -> None:
    scenes = [make_scene(96, 128, seed=i) for i in range(8)]
    rows = []
    for mode in ("disparity", "log", "neg_depth"):
        for align in ("none", "median", "least_squares"):
            metrics = []
            for s in scenes:
                gt = s["depth"]
                pred = relative_depth_from_metric(gt, mode)
                # add small noise to mimic network
                pred = pred + np.random.default_rng(0).normal(0, 0.01 * np.std(pred), pred.shape)
                m = compute_metrics(pred, gt, max_depth=20.0, align=align)
                metrics.append(m.as_dict())
            avg = {k: float(np.nanmean([m[k] for m in metrics])) for k in metrics[0]}
            rows.append({"repr": mode, "align": align, **avg})
            print(f"repr={mode:10s} align={align:14s} abs_rel={avg['abs_rel']:.4f} d1={avg['delta1']:.3f}")

    # Optional MiDaS small via torch hub
    midas_status = {"attempted": False}
    try:
        import torch

        midas_status["attempted"] = True
        # Use a tiny random CNN as stand-in if hub fails; try hub first
        try:
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            model.eval()
            midas_status["loaded"] = "MiDaS_small"
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            transform = transforms.small_transform
            mlist = []
            for s in scenes[:3]:
                img = (s["rgb"] * 255).astype(np.uint8)
                # midas expects RGB HWC
                sample = transform(img)
                if isinstance(sample, dict):
                    inp = sample["image"]
                else:
                    inp = sample
                if inp.dim() == 3:
                    inp = inp.unsqueeze(0)
                with torch.no_grad():
                    pred = model(inp)
                    if pred.dim() == 3:
                        pred = pred.unsqueeze(1)
                    pred = torch.nn.functional.interpolate(
                        pred,
                        size=s["depth"].shape,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].cpu().numpy()
                # MiDaS: larger = closer often (inverse-ish); align with LS
                m = compute_metrics(pred, s["depth"], max_depth=20.0, align="least_squares")
                mlist.append(m.as_dict())
            midas_status["metrics"] = {k: float(np.mean([x[k] for x in mlist])) for k in mlist[0]}
        except Exception as e:
            midas_status["loaded"] = None
            midas_status["error"] = str(e)[:500]
    except Exception as e:
        midas_status["error"] = str(e)

    analysis = {
        "rows": rows,
        "midas": midas_status,
        "interpretation": {
            "relative_repr": "disparity/log/neg need least_squares (or median for pure scale) before AbsRel is meaningful.",
            "protocol": "Foundation models reporting zero-shot relative depth must declare alignment; comparing raw AbsRel is invalid.",
            "dav2_note": "Depth Anything V2 relative weights follow same affine-invariant eval; metric fine-tunes drop alignment.",
        },
    }
    (OUT / "results.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    # acceptance: LS on disparity beats none
    none_ar = next(r for r in rows if r["repr"] == "disparity" and r["align"] == "none")["abs_rel"]
    ls_ar = next(r for r in rows if r["repr"] == "disparity" and r["align"] == "least_squares")["abs_rel"]
    assert ls_ar < none_ar * 0.2
    print("P4 acceptance: PASSED")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
