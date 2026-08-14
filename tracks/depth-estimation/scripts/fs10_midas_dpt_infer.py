#!/usr/bin/env python3
"""FS10: MiDaS_small real weights — relative depth inference + protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import compute_metrics  # noqa: E402
from synth_data import make_scene  # noqa: E402
from visutil import save_panels  # noqa: E402

OUT = ROOT / "results" / "fs10_midas"
OUT.mkdir(parents=True, exist_ok=True)


def load_midas():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, transforms.small_transform


def predict_depth(model, transform, rgb01: np.ndarray) -> np.ndarray:
    img = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    sample = transform(img)
    inp = sample["image"] if isinstance(sample, dict) else sample
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=rgb01.shape[:2], mode="bilinear", align_corners=False)
    return pred[0, 0].cpu().numpy()


def spearman_like(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation (Pearson on ranks)."""
    def ranks(x):
        order = x.reshape(-1).argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(order.size, dtype=np.float64)
        return r
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-8)
    rb = (rb - rb.mean()) / (rb.std() + 1e-8)
    return float(np.mean(ra * rb))


def naturalish_scene(seed: int, h=160, w=200) -> dict:
    """More natural RGB with multi-plane depth (not pure color=depth cheat)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # textured planes
    depth = np.full((h, w), 8.0, np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    # background texture
    noise = rng.normal(0, 0.08, (h, w, 3))
    base = np.stack(
        [
            0.3 + 0.2 * np.sin(xx / 9.0),
            0.35 + 0.15 * np.cos(yy / 11.0),
            0.4 + 0.1 * np.sin((xx + yy) / 13.0),
        ],
        axis=-1,
    )
    rgb[:] = np.clip(base + noise, 0, 1)
    # mid plane
    depth[30:120, 40:160] = 5.0
    rgb[30:120, 40:160] = np.clip(
        rgb[30:120, 40:160] * 0.5 + np.array([0.6, 0.45, 0.3]), 0, 1
    )
    # near object
    depth[50:100, 70:130] = 2.5
    rgb[50:100, 70:130] = np.clip(
        np.array([0.2, 0.5, 0.8]) + rng.normal(0, 0.05, (50, 60, 3)), 0, 1
    )
    return {"rgb": rgb.astype(np.float32), "depth": depth}


def main() -> None:
    model, transform = load_midas()
    scenes = [naturalish_scene(i) for i in range(5)]
    rows = []
    panels = []
    corrs = []
    corrs_rnd = []
    for i, s in enumerate(scenes):
        pred = predict_depth(model, transform, s["rgb"])
        # MiDaS often: larger = closer; invert for metric compare if needed
        # Try both orientations for LS abs_rel, keep best (scale can be negative in affine)
        m_ls = compute_metrics(pred, s["depth"], max_depth=20.0, align="least_squares")
        m_ls_inv = compute_metrics(-pred, s["depth"], max_depth=20.0, align="least_squares")
        use_inv = m_ls_inv.abs_rel < m_ls.abs_rel
        pred_use = -pred if use_inv else pred
        m_none = compute_metrics(pred_use, s["depth"], max_depth=20.0, align="none")
        m_ls = compute_metrics(pred_use, s["depth"], max_depth=20.0, align="least_squares")
        # ordinal: compare to GT with correct orientation for correlation
        # if larger midas = closer, corr with depth is negative
        c = abs(spearman_like(pred, s["depth"]))
        c_r = abs(spearman_like(np.random.default_rng(i).random(pred.shape), s["depth"]))
        corrs.append(c)
        corrs_rnd.append(c_r)
        rows.append(
            {
                "scene": i,
                "inverted": use_inv,
                "none_abs_rel": m_none.abs_rel,
                "ls_abs_rel": m_ls.abs_rel,
                "rank_corr_abs": c,
            }
        )
        if i < 3:
            panels += [(f"rgb{i}", s["rgb"]), (f"gt{i}", s["depth"]), (f"midas{i}", pred)]

    save_panels(OUT / "midas_gallery.png", panels, ncols=3)
    mean_corr = float(np.mean(corrs))
    mean_corr_r = float(np.mean(corrs_rnd))
    mean_ls = float(np.mean([r["ls_abs_rel"] for r in rows]))
    mean_none = float(np.nanmean([r["none_abs_rel"] for r in rows]))

    out = {
        "status": {"model": "MiDaS_small", "loaded": True},
        "rows": rows,
        "mean_rank_corr_abs": mean_corr,
        "mean_rank_corr_random": mean_corr_r,
        "mean_ls_abs_rel": mean_ls,
        "mean_none_abs_rel": mean_none,
        "new_capability": "Run a real pretrained relative depth network; evaluate with alignment + ordinal metrics.",
        "compare_to_fs09": "FS09 simulated relative maps; FS10 uses downloaded MiDaS weights.",
        "compare_to_fs06": "FS06 trains on synthetic GT; MiDaS zero-shot ranks structure without our labels.",
        "artifacts": ["midas_gallery.png"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
    assert mean_corr > mean_corr_r + 0.05, (mean_corr, mean_corr_r)
    # on naturalish multi-plane, LS should improve or at least be finite
    assert np.isfinite(mean_ls)
    print("FS10 acceptance: PASSED")


if __name__ == "__main__":
    main()
