#!/usr/bin/env python3
"""从 my_agent.py 重建 Kaggle notebook。不要手改 ipynb。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = (HERE / "my_agent.py").read_text()
OUT = HERE / "arc-prize-2026-arc-agi-3-starter.ipynb"

MD0 = """# ARC Prize 2026 — ARC-AGI-3

竞赛绑定 **不要改**：`arc-prize-2026-arc-agi-3`。
提交骨架仍是官方 starter（离线 wheels → 写 MyAgent → rerun 走 gateway → commit 写 dummy parquet）。

## 有没有参考别人的公开 notebook？

有。上一版只按官方文档推；这一版把公开本搜过再合成。详见仓库里的 `SURVEY.md`。

**用了别人反复验证过的：** 抹 HUD 再哈希、先走后点、没变就标死、迷路就关卡 RESET、步数必须有上限（平方分）、认角色后软偏向出口。

**没搬：** duck/TAAF 套件（要额外数据集）；不按 `ls20` 写死；不每步问大模型；不读游戏源码做离线 BFS。

## 这一版在干什么

1. 画面当图上的点。先在全图试 ACTION1-5，再点小稀有色块
2. 本关太久不过 → 放弃，别把平方分摊没
3. 图穷了才问挂上的 Gemma-4-31B

机器：`NvidiaRtxPro6000`。无网。
"""

PIP = """# 【步骤】1/4 无网安装竞赛自带的 arc-agi 轮子（不要 pip 上网）
!pip install --no-index --find-links \\
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
    arc-agi python-dotenv"""

WRITE = "%%writefile /tmp/my_agent.py\n" + AGENT

RERUN = r"""import os

# 【步骤】3/4 只有点了 Submit to Competition，Kaggle 才会设这个变量并真正打 hidden 游戏。
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # 等竞赛网关起来（sidecar 叫 gateway:8001）。
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
          --retry-max-time 600 http://gateway:8001/api/games

    # 官方框架在只读 input 里，拷到 working 才能改。
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
           /kaggle/working/ARC-AGI-3-Agents

    # 把我们的 MyAgent 塞进模板目录。
    !cp /tmp/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    # 重写 __init__.py：上游会 import langgraph 等我们没装的库。
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("from typing import Type\nfrom dotenv import load_dotenv\nfrom .agent import Agent, Playback\nfrom .swarm import Swarm\nfrom .templates.random_agent import Random\nfrom .templates.my_agent import MyAgent\n\nload_dotenv()\n\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n    'random': Random,\n    'myagent': MyAgent,\n}\n")

    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write('SCHEME=http\nHOST=gateway\nPORT=8001\nARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\nOPERATION_MODE=online\nENVIRONMENTS_DIR=\nRECORDINGS_DIR=/kaggle/working/server_recording\n')

    # 网关会记下每一步，最后吐出真正的 submission.parquet。
    !cd /kaggle/working/ARC-AGI-3-Agents && \
        MPLBACKEND=agg \
        python main.py --agent myagent
"""

COMMIT = r"""import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # 【步骤】4/4 Save and Run All 不会打游戏，只需要一份合法 parquet 让提交按钮出现。
    # 不要在这一格点 Submit to Competition。
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
