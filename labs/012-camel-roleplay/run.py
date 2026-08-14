#!/usr/bin/env python3
"""012-camel-roleplay — CAMEL RolePlaying 双智能体协作（烧 Model Proxy 额度做有意义产出）。

默认任务：为 Kaggle 玩家写一份「本周可动手的 Agent/LLM 竞赛简报」。

Usage:
  python main.py 012 run
  python main.py 012 run --rounds 4
  python main.py 012 run --task "设计一个用 Model Proxy 评测 tool-calling 的最小 benchmark"
  python main.py 012 run -m google/gemini-3-flash-preview --no-task-specify
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

from common.camel_proxy import DEFAULT_CAMEL_MODEL, extract_text, make_camel_model  # noqa: E402
from common.usage import log_usage  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXP = "012-camel-roleplay"

DEFAULT_TASK = (
    "为一名已有 Kaggle Model Proxy（日$10 AI Models）和 30h GPU 的研究者，"
    "设计本周可执行的 multi-agent 实验计划。"
    "要求：1) 3 个具体实验（含输入/输出/成功标准）；"
    "2) 如何省额度（模型选择、轮数、缓存）；"
    "3) 一份 1 天时间表。"
    "最终由助手给出可直接照做的 Markdown 简报。"
)


def cmd_run(args: argparse.Namespace) -> int:
    from camel.societies import RolePlaying
    from camel.utils import print_text_animated

    model = make_camel_model(
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        refresh=args.refresh,
    )

    task = args.task
    print("=== 012-camel-roleplay ===")
    print(f"model:   {args.model}")
    print(f"rounds:  {args.rounds}")
    print(f"task:    {task[:120]}…")

    # Assistant = researcher/writer; User = project lead who drives the task
    session = RolePlaying(
        assistant_role_name="Kaggle Agent Researcher",
        user_role_name="Lab Lead",
        task_prompt=task,
        with_task_specify=not args.no_task_specify,
        with_task_planner=args.with_planner,
        with_critic_in_the_loop=False,
        model=model,
        output_language=args.lang or "Chinese",
    )

    print("\n--- specified task ---")
    print(session.task_prompt)

    n = 0
    transcript: list[dict] = []
    final_assistant = ""

    # Kickoff
    input_msg = session.init_chat()
    while n < args.rounds:
        n += 1
        print(f"\n======== round {n}/{args.rounds} ========")
        assistant_response, user_response = session.step(input_msg)

        user_text = extract_text(user_response)
        asst_text = extract_text(assistant_response)
        print(f"\n[Lab Lead]\n{user_text}\n")
        print(f"[Researcher]\n{asst_text}\n")

        transcript.append(
            {
                "round": n,
                "user": user_text,
                "assistant": asst_text,
                "user_terminated": getattr(user_response, "terminated", False),
                "assistant_terminated": getattr(assistant_response, "terminated", False),
            }
        )
        final_assistant = asst_text

        if getattr(assistant_response, "terminated", False) or getattr(
            user_response, "terminated", False
        ):
            print("(terminated by agent)")
            break
        # CAMEL convention: continue with assistant message as next input
        if not assistant_response.msgs:
            break
        input_msg = assistant_response.msg

    ART.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "rounds": n,
        "task_original": task,
        "task_specified": session.task_prompt,
        "transcript": transcript,
        "final_assistant": final_assistant,
    }
    path = ART / "last-roleplay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = ART / "last-brief.md"
    md_path.write_text(
        f"# RolePlay brief\n\n## Task\n\n{session.task_prompt}\n\n"
        f"## Final assistant output\n\n{final_assistant}\n",
        encoding="utf-8",
    )
    log_usage(
        experiment=EXP,
        model=args.model,
        extra={"cmd": "run", "rounds": n, "chars": len(final_assistant)},
    )
    print(f"\nsaved → {path}")
    print(f"brief → {md_path}")
    # silence unused import if animation not used
    _ = print_text_animated
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="012-camel-roleplay")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="双角色协作若干轮")
    pr.add_argument("-m", "--model", default=DEFAULT_CAMEL_MODEL)
    pr.add_argument("--task", default=DEFAULT_TASK)
    pr.add_argument("--rounds", type=int, default=3, help="对话轮数（每轮双方各一次）")
    pr.add_argument("--max-tokens", type=int, default=1200)
    pr.add_argument("--temperature", type=float, default=0.3)
    pr.add_argument("--refresh", action="store_true")
    pr.add_argument("--lang", default="Chinese")
    pr.add_argument(
        "--no-task-specify",
        action="store_true",
        help="跳过 TaskSpecifyAgent（少 1 次调用、省额度）",
    )
    pr.add_argument("--with-planner", action="store_true", help="额外 TaskPlanner（更费）")
    pr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
