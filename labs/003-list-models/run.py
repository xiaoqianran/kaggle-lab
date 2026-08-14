#!/usr/bin/env python3
"""003-list-models — 列出 / 导出 Kaggle Benchmark AI Models。

Usage:
  python run.py list
  python run.py dump
  python main.py 003 list
  python main.py 003 dump
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import ensure_kaggle_token  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# vendor 惯例前缀（HTTP model 字段）；CLI 用短 slug
VENDOR_PREFIX = {
    "gemini": "google",
    "gemma": "google",
    "claude": "anthropic",
    "gpt": "openai",
    "deepseek": "deepseek-ai",
    "qwen": "qwen",
    "grok": "xai",
    "glm": "",
}


def guess_vendor(slug: str) -> str:
    for key, vendor in VENDOR_PREFIX.items():
        if slug.startswith(key) or slug.startswith(key.replace("-", "")):
            return vendor or "other"
    if slug.startswith("gpt-oss"):
        return "openai"
    return "other"


def guess_http_model(slug: str) -> str:
    vendor = guess_vendor(slug)
    if vendor in ("", "other"):
        return slug
    if vendor == "deepseek-ai":
        return f"deepseek-ai/{slug}"
    return f"{vendor}/{slug}"


def fetch_models() -> list[tuple[str, str]]:
    """Parse `kaggle b t models` → [(slug, display_name), ...]."""
    ensure_kaggle_token()
    out = subprocess.check_output(
        ["kaggle", "b", "t", "models"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.rstrip()
        if not line or line.startswith("Slug") or set(line) <= {"-", " "}:
            continue
        # slug is first token; display name is rest (often multi-space separated)
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
        if len(parts) == 2:
            rows.append((parts[0].strip(), parts[1].strip()))
        elif line.strip():
            rows.append((line.split()[0], line.strip()))
    return rows


def cmd_list(_: argparse.Namespace) -> int:
    rows = fetch_models()
    print(f"共 {len(rows)} 个模型\n")
    print(f"{'cli_slug':<40} {'display_name':<36} http_model_guess")
    print("-" * 110)
    for slug, name in rows:
        print(f"{slug:<40} {name:<36} {guess_http_model(slug)}")
    return 0


def cmd_dump(_: argparse.Namespace) -> int:
    rows = fetch_models()
    today = date.today().isoformat()
    models = []
    for slug, name in rows:
        vendor = guess_vendor(slug)
        models.append(
            {
                "cli_slug": slug,
                "display_name": name,
                "vendor": vendor,
                "http_model_guess": guess_http_model(slug),
            }
        )

    # JSON
    json_path = OUT_DIR / "kaggle_ai_models.json"
    payload = {
        "source": "kaggle b t models",
        "fetched_at": today,
        "count": len(models),
        "models": models,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # CSV
    csv_path = OUT_DIR / "kaggle_ai_models.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cli_slug", "display_name", "vendor", "http_model_guess"],
        )
        w.writeheader()
        w.writerows(models)

    # TXT
    txt_path = OUT_DIR / "kaggle_ai_models.txt"
    lines = [
        f"# Kaggle AI Models（{today}，共 {len(models)} 个）",
        "# 来源: kaggle b t models",
        "# 字段: cli_slug | display_name | http_model_guess",
        "",
    ]
    for m in models:
        lines.append(
            f"{m['cli_slug']:<40} | {m['display_name']:<36} | {m['http_model_guess']}"
        )
    lines.append("")
    lines.append("## 纯 slug")
    for m in models:
        lines.append(m["cli_slug"])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 同步一份到 data/，根目录不再堆模型表
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for p in (json_path, csv_path, txt_path):
        (data_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"已写入 {json_path}")
    print(f"已写入 {csv_path}")
    print(f"已写入 {txt_path}")
    print(f"并同步到 {data_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="003-list-models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="终端打印模型表")
    p_list.set_defaults(func=cmd_list)
    p_dump = sub.add_parser("dump", help="导出 txt/csv/json")
    p_dump.set_defaults(func=cmd_dump)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
