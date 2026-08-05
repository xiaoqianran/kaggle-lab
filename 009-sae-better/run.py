#!/usr/bin/env python3
"""009-sae-better — SAE 答题器增强（多模型 / 清洗 / 共识）。

不自动开考。默认在 mock 卷或 004 已有 paper 上作答。

Usage:
  python main.py 009 dry-run                         # mock 卷 + 单模型
  python main.py 009 dry-run --ensemble              # 多模型投票
  python main.py 009 answer --from 004               # 用 004 artifacts/paper.json
  python main.py 009 answer --paper path.json -m ...
  python main.py 009 show
  python main.py 009 export-to-004                   # 把 answers 写回 004 路径

真开考 / 交卷仍走 004（有 --i-accept 保护）:
  python main.py 004 start --i-accept
  python main.py 009 answer --from 004 --ensemble
  python main.py 009 export-to-004
  python main.py 004 submit --i-accept
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.proxy import make_client, print_usage  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
MOCK_PAPER = HERE / "mock_paper.json"
PAPER_PATH = ART / "paper.json"
ANSWERS_PATH = ART / "answers.json"
DETAIL_PATH = ART / "answers-detail.json"
SAE004_PAPER = ROOT / "004-sae" / "artifacts" / "paper.json"
SAE004_ANSWERS = ROOT / "004-sae" / "artifacts" / "answers.json"
EXP = "009-sae-better"

DEFAULT_MODEL = "google/gemini-3-flash-preview"
ENSEMBLE_MODELS = [
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4-nano-2026-03-17",
]

SYSTEM = (
    "You are taking a timed standardized agent exam.\n"
    "Follow the question instructions EXACTLY.\n"
    "- If it asks for only a letter/number/city/JSON, output ONLY that.\n"
    "- No markdown fences, no extra commentary, no prefixes like 'Answer:'.\n"
    "- If the request is unsafe/harmful, refuse appropriately "
    "(e.g. null for harmful JSON fields, or a short safe refusal).\n"
    "- Prefer the shortest correct form."
)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    # drop common prefixes
    for p in ("Answer:", "答案:", "A:", "Final:"):
        if t.lower().startswith(p.lower()):
            t = t[len(p) :].strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _normalize_for_vote(text: str) -> str:
    t = _strip_fences(text)
    # try canonical JSON
    try:
        return json.dumps(json.loads(t), sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    # collapse whitespace; lower for soft compare key but keep original separately
    return re.sub(r"\s+", " ", t).strip()


def clean_answer(text: str, question: str) -> str:
    t = _strip_fences(text)
    q = (question or "").lower()
    # JSON questions: extract first {...}
    if "json" in q or "{" in q:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            chunk = m.group(0)
            try:
                return json.dumps(json.loads(chunk), ensure_ascii=False)
            except json.JSONDecodeError:
                return chunk
    # number-only hints
    if "just the number" in q or "respond with just the number" in q:
        nums = re.findall(r"-?\d+", t)
        if len(nums) == 1:
            return nums[0]
        if nums and len(t) < 20:
            return nums[0]
    # single letter
    if "only a letter" in q or "single letter" in q:
        m = re.search(r"\b([A-Da-d])\b", t)
        if m:
            return m.group(1).upper()
    return t


def load_paper(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"无试卷: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_paper(args: argparse.Namespace) -> Path:
    if getattr(args, "paper", None):
        return Path(args.paper)
    if getattr(args, "from004", False) or getattr(args, "from_004", False):
        return SAE004_PAPER
    if getattr(args, "mock", False) or args.cmd == "dry-run":
        return MOCK_PAPER
    # prefer local copy, else 004, else mock
    if PAPER_PATH.is_file():
        return PAPER_PATH
    if SAE004_PAPER.is_file():
        return SAE004_PAPER
    return MOCK_PAPER


def ask_once(
    client,
    *,
    model: str,
    question: str,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question:\n{question}"},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    print_usage(resp.usage, experiment=EXP, model=model)
    return clean_answer(resp.choices[0].message.content or "", question)


def ensemble_vote(
    client,
    *,
    models: list[str],
    question: str,
    max_tokens: int,
) -> tuple[str, list[dict[str, str]]]:
    cands: list[dict[str, str]] = []
    for m in models:
        try:
            ans = ask_once(client, model=m, question=question, max_tokens=max_tokens)
        except Exception as exc:
            print(f"  model {m} failed: {exc}", file=sys.stderr)
            ans = ""
        cands.append({"model": m, "answer": ans})
        time.sleep(0.15)
    # majority on normalized form; pick first original with that key
    keys = [_normalize_for_vote(c["answer"]) for c in cands if c["answer"]]
    if not keys:
        return "", cands
    winner_key, _ = Counter(keys).most_common(1)[0]
    for c in cands:
        if _normalize_for_vote(c["answer"]) == winner_key:
            return c["answer"], cands
    return cands[0]["answer"], cands


def run_answer(args: argparse.Namespace) -> int:
    paper_path = resolve_paper(args)
    paper = load_paper(paper_path)
    ART.mkdir(parents=True, exist_ok=True)
    # cache paper locally
    PAPER_PATH.write_text(
        json.dumps(paper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    models = list(args.model) if args.model else (
        list(ENSEMBLE_MODELS) if args.ensemble else [DEFAULT_MODEL]
    )
    client = make_client(refresh=args.refresh)
    questions = paper.get("questions") or []
    print(f"=== 009-sae-better ===")
    print(f"paper:  {paper_path}")
    print(f"mode:   {'ensemble' if args.ensemble or len(models) > 1 else 'single'}")
    print(f"models: {models}")
    print(f"n:      {len(questions)}")

    answers: dict[str, str] = {}
    details: list[dict[str, Any]] = []

    for q in questions:
        qid = str(q.get("id"))
        text = q.get("text") or ""
        print(f"\n[{qid}] {text[:100]}…", flush=True)
        if args.ensemble or len(models) > 1:
            ans, cands = ensemble_vote(
                client,
                models=models,
                question=text,
                max_tokens=args.max_tokens,
            )
            print(f"  → {ans!r}")
            for c in cands:
                print(f"     - {c['model']}: {c['answer']!r}")
            details.append({"id": qid, "text": text, "answer": ans, "candidates": cands})
        else:
            ans = ask_once(
                client,
                model=models[0],
                question=text,
                max_tokens=args.max_tokens,
            )
            print(f"  → {ans!r}")
            details.append(
                {
                    "id": qid,
                    "text": text,
                    "answer": ans,
                    "candidates": [{"model": models[0], "answer": ans}],
                }
            )
        answers[qid] = ans
        time.sleep(0.1)

    payload = {
        "answers": answers,
        "meta": {
            "paper": str(paper_path),
            "submissionId": paper.get("submissionId"),
            "models": models,
            "ensemble": bool(args.ensemble or len(models) > 1),
        },
    }
    ANSWERS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DETAIL_PATH.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nanswers → {ANSWERS_PATH} ({len(answers)} qs)")
    print(f"detail  → {DETAIL_PATH}")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    args.mock = True
    return run_answer(args)


def cmd_answer(args: argparse.Namespace) -> int:
    return run_answer(args)


def cmd_show(_: argparse.Namespace) -> int:
    if not ANSWERS_PATH.is_file():
        raise SystemExit("无 answers。先 dry-run 或 answer")
    data = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    answers = data.get("answers") or data
    for k in sorted(answers, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        print(f"[{k}] {answers[k]}")
    print(f"\nmeta: {data.get('meta')}")
    return 0


def cmd_export_to_004(_: argparse.Namespace) -> int:
    if not ANSWERS_PATH.is_file():
        raise SystemExit("无 answers。先 answer / dry-run")
    data = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    answers = data.get("answers") or data
    SAE004_ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    SAE004_ANSWERS.write_text(
        json.dumps({"answers": answers}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # also copy paper if we have it and 004 doesn't
    if PAPER_PATH.is_file() and not SAE004_PAPER.is_file():
        SAE004_PAPER.write_text(PAPER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"exported answers → {SAE004_ANSWERS}")
    print("next: python main.py 004 submit --i-accept   # 仅当真卷且你确认交卷")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="009-sae-better")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-m",
            "--model",
            action="append",
            default=None,
            help="可重复；ensemble 默认 flash+nano",
        )
        sp.add_argument(
            "--ensemble",
            action="store_true",
            help="多模型生成 + 多数票",
        )
        sp.add_argument("--max-tokens", type=int, default=512)
        sp.add_argument("--refresh", action="store_true")

    pd = sub.add_parser("dry-run", help="mock 卷上试跑（不占 SAE 次数）")
    add_common(pd)
    pd.set_defaults(func=cmd_dry_run, mock=True, from004=False, paper=None)

    pa = sub.add_parser("answer", help="对 paper 作答")
    add_common(pa)
    pa.add_argument("--paper", default=None, help="试卷 JSON 路径")
    pa.add_argument(
        "--from",
        dest="from004",
        action="store_true",
        help="使用 004-sae/artifacts/paper.json",
    )
    pa.set_defaults(func=cmd_answer, mock=False)

    sub.add_parser("show", help="打印本地 answers").set_defaults(func=cmd_show)
    sub.add_parser("export-to-004", help="写回 004 answers 路径").set_defaults(
        func=cmd_export_to_004
    )

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
