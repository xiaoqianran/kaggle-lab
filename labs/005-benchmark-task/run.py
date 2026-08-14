#!/usr/bin/env python3
"""005-benchmark-task — Kaggle Benchmarks 最小闭环。

分步命令（推荐按顺序）:
  python main.py 005 auth
  python main.py 005 validate          # 本地 python task.py
  python main.py 005 push              # 上传任务
  python main.py 005 run -m gemini-3.5-flash
  python main.py 005 status
  python main.py 005 download
  python main.py 005 list
  python main.py 005 models

默认不自动串联 push→run（避免意外烧额度）；显式:
  python main.py 005 pipeline --i-accept -m gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import ENV_FILE, ensure_kaggle_token, refresh_token  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK_FILE = HERE / "task.py"
TASK_SLUG = "kaggle-lab-smoke-math"
ART = HERE / "artifacts"
DEFAULT_MODEL = "gemini-3.5-flash"  # CLI bare slug


def _env() -> dict[str, str]:
    ensure_kaggle_token()
    return os.environ.copy()


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd or HERE), env=_env())
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def cmd_auth(_: argparse.Namespace) -> int:
    path = refresh_token(ENV_FILE)
    # also write a .env next to task for local `python task.py` / dotenv loaders
    local_env = HERE / ".env"
    if ENV_FILE.is_file():
        shutil.copyfile(ENV_FILE, local_env)
        # append defaults useful for kbench if missing
        text = local_env.read_text(encoding="utf-8")
        extras = []
        if "LLM_DEFAULT=" not in text:
            extras.append(f"LLM_DEFAULT={DEFAULT_MODEL}")
        if "LLM_DEFAULT_EVAL=" not in text:
            extras.append(f"LLM_DEFAULT_EVAL={DEFAULT_MODEL}")
        if extras:
            with local_env.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(extras) + "\n")
    print(f"Model Proxy 凭证 → {path}")
    print(f"本地副本         → {local_env}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if not ENV_FILE.exists() and not (HERE / ".env").exists():
        print("无凭证，先 auth…", file=sys.stderr)
        cmd_auth(args)
    # Prefer root .env.model-proxy loaded by dotenv if present
    if ENV_FILE.is_file() and not (HERE / ".env").is_file():
        shutil.copyfile(ENV_FILE, HERE / ".env")
    print(f"=== validate {TASK_FILE.name} ===")
    # load env into process for kaggle_benchmarks
    env_path = HERE / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return _run([sys.executable, str(TASK_FILE)], cwd=HERE)


def cmd_push(args: argparse.Namespace) -> int:
    ensure_kaggle_token()
    cmd = [
        "kaggle",
        "b",
        "t",
        "push",
        args.slug,
        "-f",
        str(TASK_FILE),
    ]
    if args.wait:
        cmd.append("--wait")
    return _run(cmd, cwd=HERE)


def cmd_run(args: argparse.Namespace) -> int:
    ensure_kaggle_token()
    cmd = ["kaggle", "b", "t", "run", args.slug]
    for m in args.model:
        cmd.extend(["-m", m])
    if args.wait:
        cmd.append("--wait")
    return _run(cmd, cwd=HERE)


def cmd_status(args: argparse.Namespace) -> int:
    ensure_kaggle_token()
    cmd = ["kaggle", "b", "t", "status", args.slug]
    for m in args.model or []:
        cmd.extend(["-m", m])
    return _run(cmd, cwd=HERE)


def cmd_download(args: argparse.Namespace) -> int:
    ensure_kaggle_token()
    out = Path(args.out or ART / "results")
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "b", "t", "download", args.slug, "-o", str(out)]
    for m in args.model or []:
        cmd.extend(["-m", m])
    if args.force:
        cmd.append("-f")
    if args.source:
        cmd.append("-s")
    return _run(cmd, cwd=HERE)


def cmd_log(args: argparse.Namespace) -> int:
    ensure_kaggle_token()
    cmd = ["kaggle", "b", "t", "log", args.slug]
    for m in args.model or []:
        cmd.extend(["-m", m])
    return _run(cmd, cwd=HERE)


def cmd_list(_: argparse.Namespace) -> int:
    ensure_kaggle_token()
    return _run(["kaggle", "b", "t", "list"], cwd=HERE)


def cmd_models(_: argparse.Namespace) -> int:
    ensure_kaggle_token()
    return _run(["kaggle", "b", "t", "models"], cwd=HERE)


def cmd_pipeline(args: argparse.Namespace) -> int:
    if not args.i_accept:
        raise SystemExit(
            "pipeline 会 auth → validate → push --wait → run --wait → download，\n"
            "消耗 AI Models 额度。确认后:\n"
            "  python main.py 005 pipeline --i-accept -m gemini-3.5-flash"
        )
    cmd_auth(args)
    cmd_validate(args)
    args.wait = True
    args.slug = args.slug or TASK_SLUG
    cmd_push(args)
    if not args.model:
        args.model = [DEFAULT_MODEL]
    cmd_run(args)
    cmd_status(args)
    args.out = str(ART / "results")
    args.force = False
    args.source = False
    cmd_download(args)
    print("=== pipeline done ===")
    print(f"results → {ART / 'results'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="005-benchmark-task")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("auth", help="刷新 Model Proxy 凭证")
    p.set_defaults(func=cmd_auth)

    p = sub.add_parser("validate", help="本地运行 task.py")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("push", help="上传 task 到 Kaggle")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument("--wait", action="store_true", default=True)
    p.add_argument("--no-wait", action="store_false", dest="wait")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("run", help="对模型跑 task")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument(
        "-m",
        "--model",
        action="append",
        default=None,
        help="可重复；默认 gemini-3.5-flash",
    )
    p.add_argument("--wait", action="store_true", default=True)
    p.add_argument("--no-wait", action="store_false", dest="wait")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="任务/运行状态")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument("-m", "--model", action="append", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("download", help="下载结果")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("-m", "--model", action="append", default=None)
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-s", "--source", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("log", help="运行日志")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument("-m", "--model", action="append", default=None)
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("list", help="列出我的 tasks")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("models", help="可用模型")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("pipeline", help="auth→validate→push→run→download（需确认）")
    p.add_argument("--i-accept", action="store_true")
    p.add_argument("--slug", default=TASK_SLUG)
    p.add_argument("-m", "--model", action="append", default=None)
    p.set_defaults(func=cmd_pipeline)

    args = parser.parse_args(argv)
    if getattr(args, "model", None) is None and args.cmd == "run":
        args.model = [DEFAULT_MODEL]
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
