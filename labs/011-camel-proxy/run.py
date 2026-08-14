#!/usr/bin/env python3
"""011-camel-proxy — CAMEL ChatAgent + Kaggle Model Proxy 桥接冒烟。

Usage:
  python main.py 011 smoke
  python main.py 011 chat "用三句话说明 multi-agent 是什么"
  python main.py 011 chat -m google/gemini-3.5-flash "..."
  python main.py 011 tools "算 19*3，再查一下现在 UTC 大概描述"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.camel_proxy import (  # noqa: E402
    DEFAULT_CAMEL_MODEL,
    extract_text,
    make_chat_agent,
)
from common.proxy import refresh_token  # noqa: E402
from common.usage import log_usage  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXP = "011-camel-proxy"


def _log_camel_usage(resp, model: str, cmd: str) -> None:
    info = getattr(resp, "info", None) or {}
    usage = info.get("usage") if isinstance(info, dict) else None
    prompt_t = completion_t = cost = None
    if usage is not None:
        prompt_t = getattr(usage, "prompt_tokens", None) or (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        completion_t = getattr(usage, "completion_tokens", None) or (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        # camel may nest token counts differently
        if prompt_t is None and isinstance(usage, dict):
            prompt_t = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_t = usage.get("completion_tokens") or usage.get("output_tokens")
    log_usage(
        experiment=EXP,
        model=model,
        prompt_tokens=prompt_t,
        completion_tokens=completion_t,
        cost_usd=cost,
        extra={"cmd": cmd, "info_keys": list(info) if isinstance(info, dict) else []},
    )


def cmd_auth(_: argparse.Namespace) -> int:
    path = refresh_token()
    print(f"Model Proxy 凭证 → {path}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    print(f"=== 011 smoke / model={args.model} ===")
    agent = make_chat_agent(
        "You only output a single token. Never explain.",
        model_type=args.model,
        max_tokens=32,
        temperature=0,
        refresh=args.refresh,
    )
    resp = agent.step("Output exactly this string and nothing else: PONG")
    text = extract_text(resp).strip()
    print(repr(text))
    # accept PONG with optional punctuation/whitespace
    ok = bool(re.search(r"\bPONG\b", text, re.I))
    _log_camel_usage(resp, args.model, "smoke")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "smoke.json").write_text(
        json.dumps({"ok": ok, "text": text, "model": args.model}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not ok:
        raise SystemExit("smoke failed: expected PONG in response")
    print("smoke OK")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    agent = make_chat_agent(
        args.system,
        model_type=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        refresh=args.refresh,
        output_language=args.lang,
    )
    print(f"=== 011 chat / model={args.model} ===")
    resp = agent.step(args.prompt)
    text = extract_text(resp)
    print(text)
    _log_camel_usage(resp, args.model, "chat")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "last-chat.json").write_text(
        json.dumps(
            {"prompt": args.prompt, "reply": text, "model": args.model},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _local_tools():
    from camel.toolkits import FunctionTool

    def now_utc() -> str:
        """Return current UTC time as ISO string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def calc(expression: str) -> str:
        """Evaluate a simple arithmetic expression like 19*3 or (2+3)/5."""
        expr = expression.strip()
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", expr):
            return json.dumps({"error": "only simple arithmetic allowed"})
        try:
            return json.dumps(
                {"expression": expr, "result": eval(expr, {"__builtins__": {}})}  # noqa: S307
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [FunctionTool(now_utc), FunctionTool(calc)]


def cmd_tools(args: argparse.Namespace) -> int:
    tools = _local_tools()
    agent = make_chat_agent(
        "You are a careful tool-using agent. Use tools for time and math. "
        "Give a short final answer after tools.",
        model_type=args.model,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=0.1,
        refresh=args.refresh,
    )
    print(f"=== 011 tools / model={args.model} ===")
    resp = agent.step(args.prompt)
    text = extract_text(resp)
    print(text)
    info = getattr(resp, "info", None) or {}
    if isinstance(info, dict) and info.get("tool_calls"):
        print(f"\n[tool_calls] {len(info['tool_calls'])} recorded")
    _log_camel_usage(resp, args.model, "tools")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "last-tools.json").write_text(
        json.dumps(
            {"prompt": args.prompt, "reply": text, "model": args.model},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="011-camel-proxy")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_model(sp):
        sp.add_argument("-m", "--model", default=DEFAULT_CAMEL_MODEL)
        sp.add_argument("--refresh", action="store_true")
        sp.add_argument("--max-tokens", type=int, default=1024)
        sp.add_argument("--temperature", type=float, default=0.2)

    sub.add_parser("auth", help="刷新 Model Proxy").set_defaults(func=cmd_auth)

    ps = sub.add_parser("smoke", help="连通性: 期望 PONG")
    add_model(ps)
    ps.set_defaults(func=cmd_smoke)

    pc = sub.add_parser("chat", help="单轮 CAMEL ChatAgent")
    add_model(pc)
    pc.add_argument("prompt")
    pc.add_argument(
        "--system",
        default="You are a sharp research assistant on Kaggle / ML agents.",
    )
    pc.add_argument("--lang", default=None, help="e.g. Chinese")
    pc.set_defaults(func=cmd_chat)

    pt = sub.add_parser("tools", help="ChatAgent + 本地 calc/now 工具")
    add_model(pt)
    pt.add_argument(
        "prompt",
        nargs="?",
        default="现在 UTC 几点？再算 19*3，简短回答。",
    )
    pt.set_defaults(func=cmd_tools)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
