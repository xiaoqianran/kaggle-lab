#!/usr/bin/env python3
"""Rebuild the Kaggle notebook from my_agent.py. Do not edit the ipynb by hand."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = (HERE / "my_agent.py").read_text()
OUT = HERE / "arc-prize-2026-arc-agi-3-starter.ipynb"

MD0 = """# ARC Prize 2026 — ARC-AGI-3 Starter

竞赛绑定 **不要改**：`arc-prize-2026-arc-agi-3`。
提交骨架仍是官方 starter：离线装 `arc-agi` → 写 `MyAgent` → rerun 走 gateway → commit 写 dummy `submission.parquet`。

方案按官方题目写，不抄公开高分本、不特化 `ls20`/`ft09`/`vc33`。详见同目录 `DESIGN.md`。

## 官方硬约束（侧栏题目）

- 无说明书的新游戏；64x64，颜色 0-15；ACTION1-4 语义上/下/左/右，ACTION6 点击且不给可点区域
- 计分 RHAE：`(人步数 / AI步数)^2`，人 10 步你 100 步只剩 0.01；没打完所有关有分数上限
- 竞赛模式：关卡 RESET 可以，整局重开不行；所有游戏都计分
- Swarm 多游戏并行；Gemma 只能当稀缺顾问

## 这份 notebook 的策略

`my_agent.py`：**在线世界模型 + RHAE 预算**

1. 开局探测：平移 → nav，点击改世界 → point
2. 建图：墙 / 角色 / 死点 / 过关格；走上去踩到色块就 ACTION5
3. 本关步数过多：关卡 RESET 一次按图重打；再超预算则放弃本游戏
4. 卡住才问挂上的 `google/gemma-4/transformers/gemma-4-31b-it`

机器：`NvidiaRtxPro6000`。无网。
"""

PIP = """!pip install --no-index --find-links \\
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
    arc-agi python-dotenv"""

WRITE = "%%writefile /tmp/my_agent.py\n" + AGENT

RERUN = r"""import os

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Wait for the gateway sidecar to be ready.
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
          --retry-max-time 600 http://gateway:8001/api/games

    # Copy the framework into a writable location.
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
           /kaggle/working/ARC-AGI-3-Agents

    # Drop our agent in as a framework template.
    !cp /tmp/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    # Register MyAgent in the framework's agent registry. We rewrite
    # __init__.py because the upstream version eagerly imports
    # templates with deps we don't ship (langgraph, smolagents, etc.).
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("from typing import Type\nfrom dotenv import load_dotenv\nfrom .agent import Agent, Playback\nfrom .swarm import Swarm\nfrom .templates.random_agent import Random\nfrom .templates.my_agent import MyAgent\n\nload_dotenv()\n\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n    'random': Random,\n    'myagent': MyAgent,\n}\n")

    # Point the framework at the gateway sidecar.
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write('SCHEME=http\nHOST=gateway\nPORT=8001\nARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\nOPERATION_MODE=online\nENVIRONMENTS_DIR=\nRECORDINGS_DIR=/kaggle/working/server_recording\n')

    # Run it. The gateway records every action and emits submission.parquet.
    !cd /kaggle/working/ARC-AGI-3-Agents && \
        MPLBACKEND=agg \
        python main.py --agent myagent
"""

COMMIT = r"""import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Save-and-run-all (commit) mode: emit a dummy submission so the
    # commit succeeds. The real submission.parquet is produced by the
    # gateway during competition rerun.
    import pandas as pd
    from pathlib import Path
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)

    cands = [
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1",
        "/kaggle/input/google/gemma-4/transformers/gemma-4-31b-it/1",
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    ]
    found = next((p for p in cands if Path(p).exists()), None)
    print("MODEL_PATH", found)
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
              "gpus", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print("gpu", i, torch.cuda.get_device_name(i))
    except Exception as e:
        print("torch check failed", type(e).__name__, e)
    print("agent file", Path("/tmp/my_agent.py").exists())
    submission.head()
"""


def cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
        },
        "kaggle": {
            "accelerator": "nvidiaRtxPro6000",
            "isInternetEnabled": False,
            "isGpuEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "cells": [
        {k: v for k, v in cell("markdown", MD0).items() if k != "outputs" and k != "execution_count"},
        cell("code", PIP),
        cell("code", WRITE),
        cell("code", RERUN),
        cell("code", COMMIT),
    ],
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, "bytes", OUT.stat().st_size)
