#!/usr/bin/env python3
"""014-camel-workforce-bench — 检索→辩论→写 Benchmark task→(可选)005 push。

Workforce-style multi-agent pipeline on Kaggle Model Proxy.

Usage:
  # 全流程到本地 validate（推荐）
  python main.py 014 run
  python main.py 014 run --theme "instruction following" --query "llm"

  # 分步
  python main.py 014 scout
  python main.py 014 debate
  python main.py 014 author
  python main.py 014 assemble
  python main.py 014 validate-local

  # 上传到 Kaggle Benchmarks（需确认）
  python main.py 014 publish --push --i-accept
  python main.py 014 publish --push --run-remote --i-accept

  # 实验性：原生 CAMEL Workforce PIPELINE 探针
  python main.py 014 workforce-probe

  python main.py 014 show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.camel_proxy import DEFAULT_CAMEL_MODEL  # noqa: E402
from common.proxy import refresh_token  # noqa: E402

import pipeline as P  # noqa: E402


def cmd_auth(_: argparse.Namespace) -> int:
    print(refresh_token())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    log = P.run_pipeline(
        theme=args.theme,
        query=args.query,
        n_comp=args.n,
        model=args.model,
        debate_rounds=args.debate_rounds,
        refresh=args.refresh,
        max_tokens=args.max_tokens,
        local_validate=not args.skip_local_validate,
        push=args.push,
        run_remote=args.run_remote,
        run_model=args.run_model,
        i_accept=args.i_accept,
    )
    print(json.dumps({"slug": log.get("stages", {}).get("author", {}), "elapsed": log.get("elapsed_sec")}, ensure_ascii=False))
    return 0


def cmd_scout(args: argparse.Namespace) -> int:
    P.stage_scout(
        theme=args.theme,
        model=args.model,
        query=args.query,
        n=args.n,
        refresh=args.refresh,
        max_tokens=args.max_tokens,
    )
    return 0


def cmd_debate(args: argparse.Namespace) -> int:
    if not P.BRIEF_OUT.is_file():
        raise SystemExit("先跑 scout，或 full run")
    brief = P.BRIEF_OUT.read_text(encoding="utf-8")
    P.stage_debate(
        brief=brief,
        model=args.model,
        rounds=args.debate_rounds,
        refresh=args.refresh,
        max_tokens=args.max_tokens,
    )
    return 0


def cmd_author(args: argparse.Namespace) -> int:
    if not P.BRIEF_OUT.is_file() or not P.DEBATE_OUT.is_file():
        raise SystemExit("需要 scout + debate 产物")
    brief = P.BRIEF_OUT.read_text(encoding="utf-8")
    # take last advocate section as final if present
    debate = P.DEBATE_OUT.read_text(encoding="utf-8")
    P.stage_author(
        brief=brief,
        debate_final=debate,
        model=args.model,
        refresh=args.refresh,
        max_tokens=args.max_tokens,
        theme=args.theme,
    )
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    if not P.SPEC_OUT.is_file():
        raise SystemExit("需要 author 产物 spec.json")
    data = json.loads(P.SPEC_OUT.read_text(encoding="utf-8"))
    cases = [P.Case(**c) for c in data["cases"]]
    spec = P.Spec(
        slug=data["slug"],
        title=data.get("title") or data["slug"],
        rationale=data.get("rationale") or "",
        cases=cases,
        source_theme=data.get("source_theme") or "",
        model=data.get("model") or "",
        created_at=data.get("created_at") or "",
    )
    path = P.assemble_task_py(spec)
    probs = P.static_validate(path)
    if probs:
        raise SystemExit("static problems: " + "; ".join(probs))
    print("static OK:", path)
    return 0


def cmd_validate_local(_: argparse.Namespace) -> int:
    if not P.TASK_OUT.is_file():
        raise SystemExit("无 task.py，先 assemble")
    code = P.local_kbench_validate(P.TASK_OUT)
    return code


def cmd_publish(args: argparse.Namespace) -> int:
    if not args.i_accept:
        raise SystemExit("需要 --i-accept 才能 push/run 到 Kaggle Benchmarks")
    if not P.TASK_OUT.is_file() or not P.SPEC_OUT.is_file():
        raise SystemExit("需要 assemble 产物")
    spec = json.loads(P.SPEC_OUT.read_text(encoding="utf-8"))
    out = P.publish(
        slug=spec["slug"],
        task_path=P.TASK_OUT,
        do_push=args.push,
        do_run=args.run_remote,
        run_model=args.run_model,
        wait=True,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("pushed") or not args.push else 1


def cmd_workforce_probe(args: argparse.Namespace) -> int:
    P.run_workforce_probe(
        model=args.model, refresh=args.refresh, max_tokens=args.max_tokens
    )
    return 0


def cmd_show(_: argparse.Namespace) -> int:
    for p in [
        P.BRIEF_OUT,
        P.DEBATE_OUT,
        P.SPEC_OUT,
        P.TASK_OUT,
        P.LOG_OUT,
        P.ART / "workforce-probe.json",
    ]:
        print(f"{'OK' if p.is_file() else '--'}  {p}")
    if P.SPEC_OUT.is_file():
        print("\n--- spec ---")
        print(P.SPEC_OUT.read_text(encoding="utf-8")[:2000])
    if P.LOG_OUT.is_file():
        print("\n--- pipeline log ---")
        print(P.LOG_OUT.read_text(encoding="utf-8")[:2000])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="014-camel-workforce-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, *, debate=False):
        sp.add_argument("-m", "--model", default=DEFAULT_CAMEL_MODEL)
        sp.add_argument("--refresh", action="store_true")
        sp.add_argument("--max-tokens", type=int, default=1400)
        sp.add_argument("--theme", default="multi-agent instruction following")
        sp.add_argument("--query", default="llm agent")
        sp.add_argument("--n", type=int, default=4, help="competitions to fetch")
        if debate:
            sp.add_argument("--debate-rounds", type=int, default=2)

    sub.add_parser("auth").set_defaults(func=cmd_auth)

    pr = sub.add_parser("run", help="scout→debate→author→assemble→[validate]→[publish]")
    add_common(pr, debate=True)
    pr.add_argument("--skip-local-validate", action="store_true")
    pr.add_argument("--push", action="store_true", help="push task to Kaggle")
    pr.add_argument("--run-remote", action="store_true", help="kaggle b t run after push")
    pr.add_argument("--run-model", default="gemini-3.5-flash")
    pr.add_argument("--i-accept", action="store_true")
    pr.set_defaults(func=cmd_run)

    for name, fn, debate in [
        ("scout", cmd_scout, False),
        ("debate", cmd_debate, True),
        ("author", cmd_author, False),
    ]:
        sp = sub.add_parser(name)
        add_common(sp, debate=debate)
        if name == "debate":
            sp.set_defaults(func=fn)
        else:
            sp.set_defaults(func=fn)

    # fix debate default
    # re-get debate parser defaults - already set in loop for debate

    sub.add_parser("assemble").set_defaults(func=cmd_assemble)
    sub.add_parser("validate-local").set_defaults(func=cmd_validate_local)

    pp = sub.add_parser("publish")
    pp.add_argument("--push", action="store_true", default=True)
    pp.add_argument("--run-remote", action="store_true")
    pp.add_argument("--run-model", default="gemini-3.5-flash")
    pp.add_argument("--i-accept", action="store_true")
    pp.set_defaults(func=cmd_publish)

    pw = sub.add_parser("workforce-probe", help="原生 CAMEL Workforce 探针")
    pw.add_argument("-m", "--model", default=DEFAULT_CAMEL_MODEL)
    pw.add_argument("--refresh", action="store_true")
    pw.add_argument("--max-tokens", type=int, default=2048)
    pw.set_defaults(func=cmd_workforce_probe)

    sub.add_parser("show").set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    # defaults for debate_rounds when missing
    if not hasattr(args, "debate_rounds"):
        args.debate_rounds = 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
