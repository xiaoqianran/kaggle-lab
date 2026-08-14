#!/usr/bin/env python3
"""006-judge-llm — 多模型作答 + 规则/Judge LLM 打分对照。

Usage:
  python main.py 006 run
  python main.py 006 run -m google/gemini-3-flash-preview -m openai/gpt-5.4-nano-2026-03-17
  python main.py 006 run --judge-model google/gemini-3.5-flash --refresh
  python main.py 006 show
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import make_client, print_usage  # noqa: E402

HERE = Path(__file__).resolve().parent
BANK = HERE / "bank.json"
ART = HERE / "artifacts"
EXP = "006-judge-llm"
DEFAULT_MODELS = [
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4-nano-2026-03-17",
]
DEFAULT_JUDGE = "google/gemini-3-flash-preview"


def load_bank(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or BANK
    return json.loads(p.read_text(encoding="utf-8"))


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


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


def rule_score(item: dict[str, Any], answer: str) -> tuple[float, str]:
    gold = item.get("gold") or ""
    kind = item.get("kind") or "exact"
    ans = _strip_fences(answer)

    if kind == "exact":
        ok = ans == gold or (bool(ans) and ans.split()[0] == gold)
        if not ok:
            digits = re.findall(r"-?\d+", ans)
            ok = gold in digits
        return (1.0 if ok else 0.0, f"exact gold={gold!r} got={ans!r}")

    if kind == "contains":
        ok = gold.lower() in ans.lower()
        return (1.0 if ok else 0.0, f"contains {gold!r}")

    if kind == "json_equal":
        try:
            a = json.loads(ans)
            b = json.loads(gold)
            ok = a == b
            return (1.0 if ok else 0.0, f"json equal={ok}")
        except json.JSONDecodeError as e:
            return (0.0, f"json parse fail: {e}")

    if kind == "judge":
        return (-1.0, "needs_judge")

    return (0.0, f"unknown kind {kind}")


def judge_score(
    client,
    *,
    judge_model: str,
    item: dict[str, Any],
    answer: str,
    max_tokens: int = 128,
) -> tuple[float, str]:
    prompt = (
        "You are a fair exam grader. Score 1 if the candidate answer is "
        "basically correct and on-topic for the question; score 0 only if "
        "wrong, empty, or refuses without cause.\n"
        "Do NOT require matching the reference wording.\n"
        "Return ONLY JSON: {\"score\": 0 or 1, \"reason\": \"short\"}\n\n"
        f"Question:\n{item.get('prompt')}\n\n"
        f"Reference (approximate, optional):\n{item.get('gold')}\n\n"
        f"Candidate answer:\n{_strip_fences(answer)}\n"
    )
    resp = _create_with_retry(
        client,
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    print_usage(resp.usage, experiment=EXP, model=judge_model)
    raw = _strip_fences(resp.choices[0].message.content or "")
    try:
        obj = json.loads(raw)
        score = float(obj.get("score", 0))
        reason = str(obj.get("reason", ""))
        if score not in (0.0, 1.0):
            score = 1.0 if score >= 0.5 else 0.0
        return score, reason or raw[:120]
    except json.JSONDecodeError:
        m = re.search(r'"score"\s*:\s*([01])', raw)
        if m:
            return float(m.group(1)), raw[:120]
        return 0.0, f"judge parse fail: {raw[:120]}"


def generate_answer(
    client,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    resp = _create_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    print_usage(resp.usage, experiment=EXP, model=model)
    return resp.choices[0].message.content or ""


def cmd_run(args: argparse.Namespace) -> int:
    models = args.model or list(DEFAULT_MODELS)
    bank = load_bank()
    client = make_client(refresh=args.refresh)

    print("=== 006-judge-llm ===")
    print(f"models: {models}")
    print(f"judge:  {args.judge_model} (for kind=judge items)")
    print(f"items:  {len(bank)}")

    rows: list[dict[str, Any]] = []
    per_model: dict[str, list[float]] = {m: [] for m in models}

    for item in bank:
        iid = item["id"]
        print(f"\n--- {iid} ({item.get('kind')}) ---")
        print(f"Q: {item['prompt'][:80]}…")
        for model in models:
            try:
                ans = generate_answer(
                    client,
                    model=model,
                    prompt=item["prompt"],
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                print(f"  [{model}] ERROR {exc}", file=sys.stderr)
                rows.append(
                    {
                        "id": iid,
                        "model": model,
                        "answer": "",
                        "score": 0.0,
                        "reason": f"error: {exc}",
                        "kind": item.get("kind"),
                    }
                )
                per_model[model].append(0.0)
                time.sleep(0.3)
                continue

            score, reason = rule_score(item, ans)
            if score < 0:
                try:
                    score, reason = judge_score(
                        client,
                        judge_model=args.judge_model,
                        item=item,
                        answer=ans,
                        max_tokens=args.judge_max_tokens,
                    )
                except Exception as exc:
                    score, reason = 0.0, f"judge error: {exc}"

            print(f"  [{model}] score={score}  ans={_strip_fences(ans)[:100]!r}")
            rows.append(
                {
                    "id": iid,
                    "model": model,
                    "answer": ans,
                    "score": score,
                    "reason": reason,
                    "kind": item.get("kind"),
                    "gold": item.get("gold"),
                }
            )
            per_model[model].append(float(score))
            time.sleep(0.25)

    print("\n=== leaderboard (mean score) ===")
    board = []
    for m, scores in per_model.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        board.append((mean, m, scores))
    board.sort(reverse=True)
    for mean, m, scores in board:
        print(f"  {mean:.3f}  {m}  {scores}")

    ART.mkdir(parents=True, exist_ok=True)
    out = {
        "models": models,
        "judge_model": args.judge_model,
        "bank": str(BANK),
        "rows": rows,
        "leaderboard": [
            {"model": m, "mean": mean, "scores": scores} for mean, m, scores in board
        ],
    }
    path = ART / "last-run.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["model,id,score,kind"]
    for r in rows:
        lines.append(f"{r['model']},{r['id']},{r['score']},{r['kind']}")
    (ART / "last-run.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved → {path}")
    return 0


def cmd_show(_: argparse.Namespace) -> int:
    path = ART / "last-run.json"
    if not path.is_file():
        raise SystemExit("无结果。先: python main.py 006 run")
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data.get("leaderboard"), ensure_ascii=False, indent=2))
    print(f"\nfull: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="006-judge-llm")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="多模型作答 + 打分")
    pr.add_argument(
        "-m",
        "--model",
        action="append",
        default=None,
        help="可重复；默认 flash + nano",
    )
    pr.add_argument("--judge-model", default=DEFAULT_JUDGE)
    pr.add_argument("--max-tokens", type=int, default=256)
    pr.add_argument("--judge-max-tokens", type=int, default=128)
    pr.add_argument("--refresh", action="store_true")
    pr.set_defaults(func=cmd_run)

    sub.add_parser("show", help="显示上次 leaderboard").set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
