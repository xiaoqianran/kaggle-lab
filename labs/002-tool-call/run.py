#!/usr/bin/env python3
"""002-tool-call — Model Proxy function calling 演示。

Usage:
  python run.py run
  python run.py run -m openai/gpt-5.4-nano-2026-03-17
  python main.py 002 run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import DEFAULT_MODEL, make_client, print_usage  # noqa: E402

EXP = "002-tool-call"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str, unit: str = "celsius") -> str:
    return json.dumps(
        {"city": city, "unit": unit, "temp": 22, "condition": "sunny"},
        ensure_ascii=False,
    )


def cmd_run(args: argparse.Namespace) -> int:
    client = make_client(refresh=args.refresh)
    model = args.model
    messages = [
        {
            "role": "user",
            "content": args.prompt,
        }
    ]

    print(f"=== 002-tool-call / model={model} ===")
    r1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=args.max_tokens,
    )
    m1 = r1.choices[0].message
    print(f"round1 finish_reason={r1.choices[0].finish_reason}")
    print_usage(r1.usage, experiment=EXP, model=model)

    if not m1.tool_calls:
        print("模型未发起 tool_calls，直接回复:")
        print(m1.content or "")
        return 0

    messages.append(
        {
            "role": "assistant",
            "content": m1.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m1.tool_calls
            ],
        }
    )

    for tc in m1.tool_calls:
        args_obj = json.loads(tc.function.arguments)
        print(f"tool → {tc.function.name}({args_obj})")
        if tc.function.name == "get_weather":
            result = get_weather(**args_obj)
        else:
            result = json.dumps({"error": f"unknown tool {tc.function.name}"})
        print(f"result ← {result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )

    r2 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        max_tokens=args.max_tokens,
    )
    print(f"round2 finish_reason={r2.choices[0].finish_reason}")
    print_usage(r2.usage, experiment=EXP, model=model)
    print(r2.choices[0].message.content or "")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="002-tool-call")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="跑一遍 weather 工具调用")
    p.add_argument(
        "-m",
        "--model",
        default="openai/gpt-5.4-nano-2026-03-17",
        help=f"默认 nano；也可用 {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--prompt",
        default="东京天气怎么样？请用工具查询（摄氏度）。",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
