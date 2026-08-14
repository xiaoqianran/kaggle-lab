#!/usr/bin/env python3
"""Run full From-Scratch chain FS00–FS14 (skips optional heavy if env blocks)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("FS00", "scripts/fs00_metrics_vis.py"),
    ("FS01", "scripts/fs01_pointcloud_vis.py"),
    ("FS02", "scripts/fs02_block_match_stereo.py"),
    ("FS03", "scripts/fs03_cost_volume.py"),
    ("FS04", "scripts/fs04_tiny_regressor.py"),
    ("FS05", "scripts/fs05_multiscale_silog.py"),
    ("FS06", "scripts/fs06_skip_ablation.py"),
    ("FS07", "scripts/run_p3_photometric.py"),
    ("FS08", "scripts/fs08_automask_minreproj.py"),
    ("FS09", "scripts/run_p4_foundation_protocol.py"),
    ("FS10", "scripts/fs10_midas_dpt_infer.py"),
    ("FS11", "scripts/fs11_depth_anything_student.py"),
    ("FS12", "scripts/fs12_metric_tape_test.py"),
    ("FS13", "scripts/fs13_generative_depth_mini.py"),
    ("FS14", "scripts/fs14_research_closed_loop.py"),
]


def main() -> None:
    results = []
    for name, rel in STEPS:
        script = ROOT / rel
        print("\n" + "=" * 60 + f"\n{name} :: {rel}\n" + "=" * 60, flush=True)
        r = subprocess.run([PY, str(script)], cwd=str(ROOT))
        results.append((name, r.returncode))
        if r.returncode != 0:
            print(f"FAILED {name} code={r.returncode}", flush=True)
            # continue only if non-critical? user said don't skip — stop on failure
            break
    print("\nSUMMARY:")
    for name, code in results:
        print(f"  {name}: {'OK' if code == 0 else 'FAIL'}")
    if any(c != 0 for _, c in results):
        sys.exit(1)
    print("ALL FS STEPS PASSED")


if __name__ == "__main__":
    main()
