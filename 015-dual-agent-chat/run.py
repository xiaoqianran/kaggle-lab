#!/usr/bin/env python3
"""015-dual-agent-chat — 双智能体自动多轮对谈（Kaggle Model Proxy）。

不依赖 CAMEL RolePlaying：A/B 各自 system + 共享历史，轮流调用 Proxy。
默认模型走「便宜好用」三选一（flash-lite / 3.1-lite / gpt-nano）。

Usage:
  python main.py 015 models
  python main.py 015 presets
  python main.py 015 run
  python main.py 015 run --preset debate --rounds 3
  python main.py 015 run -m google/gemini-3.1-flash-lite-preview --rounds 2
  python main.py 015 run --topic "AI 会不会取代程序员？" --a-name 乐观派 --b-name 悲观派
  python main.py 015 smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from common.proxy import refresh_token  # noqa: E402

from dual_chat import (  # noqa: E402
    Agent,
    result_to_dict,
    result_to_markdown,
    run_dual_chat,
)
from presets import DEFAULT_MODEL, MODELS, PRESETS  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXP = "015-dual-agent-chat"


def cmd_models(_: argparse.Namespace) -> int:
    print("=== 015 推荐模型（Kaggle Model Proxy）===")
    for m in MODELS:
        mark = " ← default" if m["id"] == DEFAULT_MODEL else ""
        print(f"  [{m['tag']}] {m['id']}")
        print(f"           {m['label']} — {m['blurb']}{mark}")
    return 0


def cmd_presets(_: argparse.Namespace) -> int:
    print("=== 015 预设 ===")
    for key, p in PRESETS.items():
        print(f"  {key:12s} {p['label']:8s}  {p['topic']}")
        print(f"               A={p['a']['name']} / B={p['b']['name']}")
    return 0


def cmd_auth(_: argparse.Namespace) -> int:
    path = refresh_token()
    print(f"Model Proxy 凭证 → {path}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """1 轮极短对谈，验证链路。"""
    print(f"=== 015 smoke / model={args.model} ===")
    res = run_dual_chat(
        topic="用一句话互相问好，然后各说一个关于 Kaggle 的冷知识。",
        agent_a=Agent("a", "甲", "简洁、幽默"),
        agent_b=Agent("b", "乙", "简洁、爱抬杠"),
        rounds=1,
        model=args.model,
        max_tokens=min(args.max_tokens, 256),
        temperature=0.7,
        refresh_first=args.refresh,
        on_turn=lambda t: print(f"\n[{t.agent_name}]\n{t.content}\n"),
    )
    ok = len(res.transcript) == 2 and all(len(t.content) >= 4 for t in res.transcript)
    print("smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _resolve_agents(args: argparse.Namespace) -> tuple[str, Agent, Agent]:
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(
                f"unknown preset {args.preset!r}; choose from {', '.join(PRESETS)}"
            )
        p = PRESETS[args.preset]
        topic = args.topic or p["topic"]
        a = Agent("a", args.a_name or p["a"]["name"], args.a_persona or p["a"]["persona"])
        b = Agent("b", args.b_name or p["b"]["name"], args.b_persona or p["b"]["persona"])
        return topic, a, b

    topic = args.topic or PRESETS["debate"]["topic"]
    a = Agent(
        "a",
        args.a_name or "构建者",
        args.a_persona
        or "乐观的产品负责人，相信流程与异步协作能放大创造力，喜欢给可执行方案",
    )
    b = Agent(
        "b",
        args.b_name or "怀疑者",
        args.b_persona
        or "犀利的组织行为学者，擅长拆穿空话，强调面对面偶然碰撞与文化密度",
    )
    return topic, a, b


def cmd_run(args: argparse.Namespace) -> int:
    topic, agent_a, agent_b = _resolve_agents(args)
    print("=== 015-dual-agent-chat ===")
    print(f"model:  {args.model}")
    print(f"rounds: {args.rounds}  (each = A then B)")
    print(f"topic:  {topic}")
    print(f"A:      {agent_a.name} — {agent_a.persona[:48]}…")
    print(f"B:      {agent_b.name} — {agent_b.persona[:48]}…")
    print()

    def on_turn(t) -> None:
        print(f"-------- R{t.round} · {t.agent_name} --------")
        print(t.content)
        print()

    res = run_dual_chat(
        topic=topic,
        agent_a=agent_a,
        agent_b=agent_b,
        rounds=args.rounds,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        refresh_first=args.refresh,
        on_turn=on_turn,
    )

    ART.mkdir(parents=True, exist_ok=True)
    payload = result_to_dict(res)
    payload["preset"] = args.preset
    json_path = ART / "last-dual-chat.json"
    md_path = ART / "last-dual-chat.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(result_to_markdown(res), encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    print(f"turns: {len(res.transcript)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="015-dual-agent-chat: 双智能体自动对谈 (Kaggle Model Proxy)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_auth = sub.add_parser("auth", help="刷新 Model Proxy 临时凭证")
    p_auth.set_defaults(func=cmd_auth)

    p_models = sub.add_parser("models", help="列出推荐模型")
    p_models.set_defaults(func=cmd_models)

    p_presets = sub.add_parser("presets", help="列出预设话题/人设")
    p_presets.set_defaults(func=cmd_presets)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-m",
            "--model",
            default=DEFAULT_MODEL,
            help=f"Model Proxy model id (default: {DEFAULT_MODEL})",
        )
        p.add_argument("--max-tokens", type=int, default=640)
        p.add_argument("--temperature", type=float, default=0.9)
        p.add_argument(
            "--refresh",
            action="store_true",
            help="开跑前刷新 Model Proxy 凭证",
        )

    p_smoke = sub.add_parser("smoke", help="1 轮冒烟")
    add_model_args(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke)

    p_run = sub.add_parser("run", help="自动多轮对谈")
    add_model_args(p_run)
    p_run.add_argument("--rounds", type=int, default=4, help="轮数（每轮 A+B 各一次）")
    p_run.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="debate",
        help="预设：debate/socratic/brainstorm/comedy",
    )
    p_run.add_argument("--topic", default=None, help="覆盖话题")
    p_run.add_argument("--a-name", default=None)
    p_run.add_argument("--b-name", default=None)
    p_run.add_argument("--a-persona", default=None)
    p_run.add_argument("--b-persona", default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
