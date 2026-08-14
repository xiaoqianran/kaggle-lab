#!/usr/bin/env python3
"""010-quota-dashboard — GPU/TPU 额度 + 本地 AI Models 用量账本。

Usage:
  python main.py 010 show
  python main.py 010 gpu
  python main.py 010 usage
  python main.py 010 usage --json
  python main.py 010 all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import ensure_kaggle_token  # noqa: E402
from common.usage import USAGE_LOG, load_usage_rows, summarize_usage  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
DAILY_BUDGET = 10.0
MONTHLY_BUDGET = 100.0


def _kaggle_out(args: list[str]) -> str:
    ensure_kaggle_token()
    env = os.environ.copy()
    r = subprocess.run(
        ["kaggle", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise SystemExit(f"kaggle {' '.join(args)} failed ({r.returncode}): {err[:500]}")
    return r.stdout


def cmd_gpu(_: argparse.Namespace) -> int:
    print("=== Accelerator quota (GPU / TPU) ===")
    print(_kaggle_out(["quota"]))
    return 0


def cmd_config(_: argparse.Namespace) -> int:
    print("=== kaggle config ===")
    print(_kaggle_out(["config", "view"]))
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    rows = load_usage_rows()
    summary = summarize_usage(rows)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print("=== Local AI Models usage log ===")
    print(f"log:    {USAGE_LOG}")
    print(f"calls:  {summary['calls']}")
    print(f"tokens: prompt={summary['prompt_tokens']}  completion={summary['completion_tokens']}")
    print(f"cost≈   ${summary['total_cost_usd']:.6f}")
    print(f"budget: daily ${DAILY_BUDGET:.0f} / monthly ${MONTHLY_BUDGET:.0f}  (account-level; log is this machine only)")
    if summary["total_cost_usd"] >= DAILY_BUDGET * 0.5:
        print("warn:   local logged cost ≥ 50% of daily budget — check kaggle.com usage UI")
    if summary["by_experiment"]:
        print("\nby experiment:")
        for k, v in sorted(summary["by_experiment"].items(), key=lambda x: -x[1]):
            print(f"  {k:28} ${v:.6f}")
    if summary["by_model"]:
        print("\nby model:")
        for k, v in sorted(summary["by_model"].items(), key=lambda x: -x[1]):
            print(f"  {k:48} ${v:.6f}")
    if args.tail and rows:
        print(f"\nlast {min(args.tail, len(rows))} calls:")
        for r in rows[-args.tail :]:
            c = r.get("cost_usd")
            c_s = f"${c:.6f}" if c is not None else "$?"
            print(
                f"  {r.get('ts')}  {r.get('experiment')}  {r.get('model')}  "
                f"p={r.get('prompt_tokens')} c={r.get('completion_tokens')}  {c_s}"
            )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "usage-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    cmd_config(args)
    print()
    cmd_gpu(args)
    print()
    # default tail for show
    if not hasattr(args, "tail") or args.tail is None:
        args.tail = 5
    if not hasattr(args, "json"):
        args.json = False
    cmd_usage(args)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    return cmd_show(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="010-quota-dashboard")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show", help="config + GPU/TPU + local AI usage")
    p.add_argument("--tail", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("all", help="同 show")
    p.add_argument("--tail", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("gpu", help="仅 kaggle quota")
    p.set_defaults(func=cmd_gpu)

    p = sub.add_parser("config", help="kaggle config view")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("usage", help="本地 logs/usage.jsonl 汇总")
    p.add_argument("--tail", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_usage)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
