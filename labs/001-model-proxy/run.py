#!/usr/bin/env python3
"""001-model-proxy — 刷新凭证并调用 Kaggle AI Models。

Usage:
  python run.py auth
  python run.py chat "你好"
  python run.py chat -m openai/gpt-5.4-nano-2026-03-17 "1+1=?"
  python main.py 001 auth
  python main.py 001 chat "hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import (  # noqa: E402
    DEFAULT_MODEL,
    ENV_FILE,
    chat,
    refresh_token,
)


def cmd_auth() -> int:
    path = refresh_token()
    print(f"Model Proxy 凭证已写入: {path}")
    print("提示: 临时 key 约 1 小时过期，过期后重新 auth。")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if args.refresh or not ENV_FILE.exists():
        refresh_token()
    text = chat(
        args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        refresh=False,
    )
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="001-model-proxy: Kaggle Model Proxy 调用"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_auth = sub.add_parser("auth", help="刷新 MODEL_PROXY 临时凭证")
    p_auth.set_defaults(func=lambda a: cmd_auth())

    p_chat = sub.add_parser("chat", help="一次对话补全")
    p_chat.add_argument("prompt", help="用户提示词")
    p_chat.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p_chat.add_argument("--max-tokens", type=int, default=256)
    p_chat.add_argument(
        "--refresh",
        action="store_true",
        help="调用前先刷新凭证",
    )
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
