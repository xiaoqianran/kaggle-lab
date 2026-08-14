#!/usr/bin/env python3
"""Run full Object Detection From-Scratch chain FS00–FS15 sequentially."""
from __future__ import annotations

import json
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

CHAIN = [
    ("FS00", "fs00_protocol_viz.py"),
    ("FS01", "fs01_sliding_window.py"),
    ("FS02", "fs02_image_pyramid.py"),
    ("FS03", "fs03_gradient_features.py"),
    ("FS04", "fs04_proposals.py"),
    ("FS05", "fs05_rcnn_crops.py"),
    ("FS06", "fs06_fast_rcnn_share.py"),
    ("FS07", "fs07_two_stage_viz.py"),
    ("FS08", "fs08_dense_yolo_viz.py"),
    ("FS09", "fs09_focal_loss.py"),
    ("FS10", "fs10_fpn_multiscale.py"),
    ("FS11", "fs11_anchor_vs_free.py"),
    ("FS12", "fs12_detr_viz.py"),
    ("FS13", "fs13_recipe_viz.py"),
    ("FS14", "fs14_domain_viz.py"),
    ("FS15", "fs15_hypothesis_viz.py"),
]


def main() -> None:
    summary = []
    for name, script in CHAIN:
        path = SCRIPTS / script
        print("\n" + "=" * 60)
        print(f"RUNNING {name}: {script}")
        print("=" * 60)
        try:
            runpy.run_path(str(path), run_name="__main__")
            summary.append({"step": name, "script": script, "status": "ok"})
        except Exception as e:
            traceback.print_exc()
            summary.append({"step": name, "script": script, "status": "FAIL", "error": str(e)})
            # debug: re-raise to force fix
            raise
    out = ROOT / "results" / "fs_chain_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nALL FS00–FS15 COMPLETE")
    print("summary →", out)


if __name__ == "__main__":
    main()
