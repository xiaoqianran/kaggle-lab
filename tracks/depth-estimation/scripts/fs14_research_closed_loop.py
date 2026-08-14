#!/usr/bin/env python3
"""FS14: research closed loop — claim card + controlled ablation table artifact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "fs14_research"
OUT.mkdir(parents=True, exist_ok=True)

# reuse prior JSON results as "experimental evidence base"
RESULT_DIRS = [
    "fs00_metrics",
    "fs02_block_match",
    "fs04_tiny_regressor",
    "fs06_skip_ablation",
    "fs08_automask",
    "p4_foundation",
    "p5_metric",
    "p7_hypothesis",
    "fs12_metric_tape",
]


def main() -> None:
    evidence = {}
    for name in RESULT_DIRS:
        p = ROOT / "results" / name / "results.json"
        if p.exists():
            evidence[name] = json.loads(p.read_text())

    # synthesize a research-style ablation table from available numbers
    table = []
    if "fs04_tiny_regressor" in evidence:
        e = evidence["fs04_tiny_regressor"]
        table.append(
            {
                "method": "FS04 tiny monocular",
                "abs_rel_in_domain": e.get("metrics_easy_holdout", {}).get("abs_rel"),
                "abs_rel_shift": e.get("metrics_hard_domain_shift", {}).get("abs_rel"),
            }
        )
    if "fs06_skip_ablation" in evidence:
        e = evidence["fs06_skip_ablation"]
        table.append({"method": "UNet skips", "abs_rel": e.get("unet_abs_rel"), "boundary": e.get("unet_boundary_absrel")})
        table.append({"method": "No skips", "abs_rel": e.get("noskip_abs_rel"), "boundary": e.get("noskip_boundary_absrel")})
    if "p7_hypothesis" in evidence:
        e = evidence["p7_hypothesis"]
        table.append(
            {
                "method": "H1 edge vs capacity",
                "supported": e.get("supported"),
                "edge_gain": e.get("edge_overall_gain"),
                "cap_gain": e.get("cap_overall_gain"),
            }
        )

    claim_card = {
        "paper_or_idea": "Local: Edge-aware loss beats modest capacity under fixed budget",
        "claim": "Boundary-weighted training reduces AbsRel more than width 8→12 at fixed steps.",
        "evidence": evidence.get("p7_hypothesis", {}),
        "supports_claim": evidence.get("p7_hypothesis", {}).get("supported"),
        "threats_to_validity": [
            "Synthetic easy data",
            "CPU short budget",
            "No NYU/KITTI confirmation",
        ],
        "next_experiment": "Repeat H1 on NYU subset with fixed FLOPs and edge IoU metrics.",
    }

    # write human-readable markdown report
    md = [
        "# FS14 Research Closed Loop Report",
        "",
        "## Claim card",
        f"- **Claim:** {claim_card['claim']}",
        f"- **Supported (this repo run):** {claim_card['supports_claim']}",
        f"- **Threats:** {', '.join(claim_card['threats_to_validity'])}",
        f"- **Next:** {claim_card['next_experiment']}",
        "",
        "## Ablation table (from chain evidence)",
        "```json",
        json.dumps(table, indent=2),
        "```",
        "",
        "## Capability ladder (what we can do now)",
        "1. Evaluate depth honestly (alignment protocols).",
        "2. Recover depth from stereo matching and see texture failure.",
        "3. Train monocular nets; measure domain shift.",
        "4. Use photometric self-supervision signals.",
        "5. Distill relative teachers; demand metric tape tests.",
        "6. Write claim cards with threats to validity.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    out = {
        "claim_card": claim_card,
        "ablation_table": table,
        "evidence_keys": list(evidence.keys()),
        "new_capability": "Turn experiments into a paper-style claim + ablation + next-step.",
        "compare_to_all": "Closes the from-scratch ladder with research hygiene.",
        "artifacts": ["REPORT.md"],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:2000])
    assert "p7_hypothesis" in evidence or table
    print("FS14 acceptance: PASSED")


if __name__ == "__main__":
    main()
