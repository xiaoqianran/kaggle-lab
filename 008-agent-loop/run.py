#!/usr/bin/env python3
"""008-agent-loop — 多轮 ReAct：Model Proxy + 本地 tools + 可选 MCP。

Usage:
  python main.py 008 run "东京天气和 2+3 是多少？"
  python main.py 008 run --with-mcp "找一个热门 LLM 竞赛" --refresh
  python main.py 008 run -m openai/gpt-5.4-nano-2026-03-17 --max-rounds 4 "..."
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.proxy import DEFAULT_MODEL, make_client, print_usage  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXP = "008-agent-loop"


def tool_get_weather(city: str, unit: str = "celsius") -> str:
    table = {
        "tokyo": (24, "partly cloudy"),
        "东京": (24, "partly cloudy"),
        "shanghai": (28, "humid"),
        "上海": (28, "humid"),
        "san francisco": (18, "foggy"),
    }
    key = city.strip().lower()
    temp, cond = table.get(key, (22, "sunny"))
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return json.dumps(
        {"city": city, "unit": unit, "temp": temp, "condition": cond},
        ensure_ascii=False,
    )


def tool_calc(expression: str) -> str:
    expr = expression.strip()
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expr):
        return json.dumps({"error": "only simple arithmetic allowed"})
    try:
        val = eval(expr, {"__builtins__": {}}, {"math": math})  # noqa: S307
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"expression": expr, "result": val})


def tool_now() -> str:
    return json.dumps(
        {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local_hint": "container UTC unless TZ set",
        }
    )


def tool_search_competitions(query: str, n: int = 3) -> str:
    import importlib.util

    path = ROOT / "007-mcp-harness" / "run.py"
    spec = importlib.util.spec_from_file_location("mcp007", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        r = mod.mcp_request(
            "tools/call",
            {
                "name": "search_competitions",
                "arguments": {"request": {"search": query, "pageSize": int(n)}},
            },
            req_id=40,
        )
        data = mod._unwrap_text(r)
        return json.dumps(data, ensure_ascii=False)[:4000]
    except SystemExit as e:
        return json.dumps({"error": str(e)})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气（演示数据）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算简单算术表达式，如 2+3 或 7*6",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now_utc",
            "description": "返回当前 UTC 时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

MCP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_competitions",
        "description": "在 Kaggle 搜索竞赛（真实 MCP）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n": {"type": "integer", "description": "返回条数，默认 3"},
            },
            "required": ["query"],
        },
    },
}


def _dispatch(name: str, args_obj: dict[str, Any], *, with_mcp: bool) -> str:
    if name == "get_weather":
        return tool_get_weather(**args_obj)
    if name == "calc":
        return tool_calc(**args_obj)
    if name == "now_utc":
        return tool_now()
    if name == "search_competitions" and with_mcp:
        return tool_search_competitions(
            args_obj.get("query", ""), int(args_obj.get("n") or 3)
        )
    return json.dumps({"error": f"unknown or disabled tool: {name}"})


def _create_with_retry(client, **kwargs):
    last_exc = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            s = str(exc).lower()
            if "429" in s or "rate_limit" in s or "heavy load" in s:
                wait = 2 ** attempt
                print(f"[retry] rate limit, sleep {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            if "401" in s or "403" in s:
                print("凭证可能过期，刷新后重试…", file=sys.stderr)
                client = make_client(refresh=True)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def cmd_run(args: argparse.Namespace) -> int:
    client = make_client(refresh=args.refresh)
    model = args.model
    tools = list(TOOL_SCHEMAS)
    if args.with_mcp:
        tools.append(MCP_SCHEMA)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a careful agent. Use tools when helpful. "
                "Prefer calc for math. Prefer get_weather for weather. "
                "When search_competitions is available, use it for Kaggle comps. "
                "After tools return, give a concise final answer to the user."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]

    print(f"=== 008-agent-loop / model={model} / mcp={args.with_mcp} ===")
    transcript: list[dict[str, Any]] = []
    final = ""

    for round_i in range(1, args.max_rounds + 1):
        print(f"\n--- round {round_i} ---")
        resp = _create_with_retry(
            client,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=args.max_tokens,
        )
        print_usage(resp.usage, experiment=EXP, model=model)
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        print(f"finish_reason={finish}")

        if not msg.tool_calls:
            final = msg.content or ""
            print(final)
            transcript.append({"round": round_i, "final": final})
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args_obj = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args_obj = {}
            print(f"tool → {tc.function.name}({args_obj})")
            result = _dispatch(tc.function.name, args_obj, with_mcp=args.with_mcp)
            print(f"result ← {result[:300]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
            transcript.append(
                {
                    "round": round_i,
                    "tool": tc.function.name,
                    "args": args_obj,
                    "result": result,
                }
            )
    else:
        print("(max rounds reached without final text)", file=sys.stderr)

    ART.mkdir(parents=True, exist_ok=True)
    out = {
        "model": model,
        "prompt": args.prompt,
        "with_mcp": args.with_mcp,
        "final": final,
        "transcript": transcript,
    }
    path = ART / "last-run.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\ntranscript → {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="008-agent-loop")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="多轮 tool-using agent")
    p.add_argument("prompt", help="用户问题")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--with-mcp",
        action="store_true",
        help="启用 search_competitions（真 MCP，需 KAGGLE_API_TOKEN）",
    )
    p.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
