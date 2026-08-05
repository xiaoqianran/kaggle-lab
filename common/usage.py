"""Append-only AI Models usage log under logs/usage.jsonl."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
USAGE_LOG = LOG_DIR / "usage.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_usage(
    *,
    experiment: str,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "ts": _now_iso(),
        "epoch": time.time(),
        "experiment": experiment,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
    }
    if extra:
        row["extra"] = extra
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return USAGE_LOG


def log_from_openai_usage(
    usage: Any,
    *,
    experiment: str,
    model: str,
    extra: dict[str, Any] | None = None,
) -> float | None:
    """Parse OpenAI-style usage (+ optional Kaggle nanodollar cost). Returns cost_usd."""
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    cost = getattr(usage, "cost", None) or {}
    cost_usd: float | None = None
    if isinstance(cost, dict):
        total_nano = (cost.get("input_tokens_cost_nanodollars") or 0) + (
            cost.get("output_tokens_cost_nanodollars") or 0
        )
        cost_usd = float(total_nano) / 1e9
    log_usage(
        experiment=experiment,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost_usd,
        extra=extra,
    )
    return cost_usd


def load_usage_rows(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or USAGE_LOG
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_usage(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = load_usage_rows() if rows is None else rows
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    by_exp: dict[str, float] = {}
    by_model: dict[str, float] = {}
    for r in rows:
        c = r.get("cost_usd")
        if c is None:
            c = 0.0
        total_cost += float(c)
        total_prompt += int(r.get("prompt_tokens") or 0)
        total_completion += int(r.get("completion_tokens") or 0)
        exp = str(r.get("experiment") or "unknown")
        model = str(r.get("model") or "unknown")
        by_exp[exp] = by_exp.get(exp, 0.0) + float(c)
        by_model[model] = by_model.get(model, 0.0) + float(c)
    return {
        "calls": len(rows),
        "total_cost_usd": total_cost,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "by_experiment": by_exp,
        "by_model": by_model,
        "log_path": str(USAGE_LOG),
    }
