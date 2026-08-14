#!/usr/bin/env python3
"""从 my_agent.py 重建 Kaggle notebook。不要手改 ipynb。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = (HERE / "my_agent.py").read_text()
OUT = HERE / "arc-prize-2026-arc-agi-3-starter.ipynb"

MD0 = """# ARC Prize 2026 — ARC-AGI-3

竞赛绑定 **不要改**：`arc-prize-2026-arc-agi-3`。无网。机器：`NvidiaRtxPro6000`。

## 先看清楚：这个 notebook 有两种跑法

| 你点的按钮 | 大概耗时 | 有没有玩游戏 | 榜分 |
|---|---|---|---|
| Save and Run All | 十几秒 | **没有**。只装包、写 agent、写一份假 parquet | 不变 |
| Submit to Competition | 数小时 | **有**。网关把 hidden 游戏一帧帧塞给 MyAgent | 才会变 |

下面四格代码：1 装轮子 → 2 写出智能体 → 3 **只有评分重跑才打游戏** → 4 假提交（让提交按钮出现）。
Save and Run All 会跳过第 3 格里的 `main.py`，所以十几秒 COMPLETE 不代表得了分。
"""

MD_PLAY = """# 我们的方案：怎么玩游戏（对着官方赛题设计）

官方报告考四件事：探索、建模、自己定目标、规划执行。
至少 6 关。第 1 关是教程。后面关要组合前面学会的机制。
整局分按关卡号加权；只打完前几关会被封顶（5 关打完 3 关最多 40%）。

公开哈希图每关清空地图，等于把教程关白打了。我们不这么干。

## 一句话

教程关把「怎么赢」写进技能纸；后面关用技能纸定目标，A* 走近再按。

## 每一步

```mermaid
flowchart TD
    A[网关送来一帧] --> B[抹掉分数条]
    B --> C{技能纸已经知道赢的颜色或角色了吗}
    C -->|知道且走得到| D[A星向目标走一步]
    D --> E{这一步会踩上目标吗}
    E -->|会| F[下一步立刻 ACTION5]
    E -->|还没有| G[先走这一步]
    C -->|出口还在但走不到| K[先点小开关 大色块当墙]
    C -->|还不知道| H[便宜实验: 先按再走 认键位]
    H --> I[把会走/会点/会按 写进技能纸]
    I --> A
    F --> A
    G --> A
    K --> A
```

## 计分决定预算

教程关权重最小：把机制学到手就过。后面关权重大、要组合，多给步数。
没打完最后几关，前面再快也封顶。所以不要为了省第 1 关的步数而放弃整局。
整局最多 8000 步：6 关预算加起来已经超过 1500，旧上限会把后面关掐死。

Gemma-4 只在图穷了才问。官方：模型内部思考不计步。不按游戏名写死。
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
    print("这是 Save and Run All：没有打任何一局游戏。")
    print("真打 hidden 集请点 Submit to Competition，会跑数小时，才会改榜。")
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
        {k: v for k, v in cell("markdown", MD_PLAY).items() if k != "outputs" and k != "execution_count"},
        cell("code", PIP),
        cell("code", WRITE),
        cell("code", RERUN),
        cell("code", COMMIT),
    ],
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")) + "\n")
print("wrote", OUT, "bytes", OUT.stat().st_size)
