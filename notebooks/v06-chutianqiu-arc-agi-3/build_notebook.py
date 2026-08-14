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

MD_PLAY = """# 我们的方案：怎么玩游戏

比赛不给说明书。网关每次只给你一张 64×64、16 色的画面，你还一个动作：

- ACTION1–4：默认上/下/左/右（个别游戏会改键位，我们现场认）
- ACTION5：交互（踩到旗子上按一下、开门、捡东西）
- ACTION6：点击某个格子（没有准星提示）
- ACTION7：撤销
- RESET：没开局或死了只能这个，否则服务器 400。竞赛模式只允许关卡重置

计分是平方效率：`(人类步数 / AI步数)^2`，上限 1.15。人 10 步你 100 步只剩 0.01。
所以方案的核心不是「能通就行」，而是 **少走冤枉路**。

## 一句话

把每个画面当成地图上的一间房。进门先试「按一下」，再走路，最后才点。
试过没反应的招作废。迷路了沿原路回去。地图穷了才问 Gemma。

## 每一步怎么选动作

```mermaid
flowchart TD
    A[网关送来一帧 64x64] --> B[抹掉顶底长条分数条]
    B --> C[给画面做指纹 当作一间房]
    C --> D{这间房还有没试过的按一下或走路吗}
    D -->| 有| E[先 ACTION5 再朝像出口的方向走]
    D -->| 没有| F{别的房间还有没走过的路吗}
    F -->| 有| G[沿已知边走回那间房]
    F -->| 没有| H{还能点小色块吗}
    H -->| 有| I[点最小最稀有的色块]
    H -->| 没有| J[关卡 RESET 或问 Gemma 或放弃]
    E --> K[把动作交给网关]
    G --> K
    I --> K
    J --> K
    K --> L{回来的画面如何}
    L -->| 没变| M[这招在这间房标死]
    L -->| 变了| N[地图上连一条边]
    L -->| 过关| O[按平方效率记分]
    M --> A
    N --> A
```

## 两个典型关，我们会怎么玩

**走路关（小人要走到旗子上再按）。**
1. 走进任何新格子，先 ACTION5：万一已经站在旗子上，立刻就能过，不会「踩到了又走开」。
2. 看哪个小色块跟着上下左右移动 → 认出角色和键位，跨关记住。
3. 之后大约七成步数朝「最像按钮/旗子」的小色块走；只走还没踩过的格子，避免撞墙来回抖。
4. 某方向走了画面完全不变 → 当墙，这间房不再试这个方向。

**点击关（走路没用，要点对某个小色块）。**
1. ACTION1–5 全是画面不变 → 全部标死。
2. 剩下才点：小、颜色少的块更像按钮。最像的前 4 个按顺序点，后面打乱（点选顺序写死会把点选关困死）。
3. 每间房最多留 16 个点击候选，避免把时间耗在地板上。

## 为什么不每步问 Gemma-4

公开榜上纯大模型要么要额外 vLLM 包，要么把 9 小时问光。
我们挂的是 transformers 4-bit：后台慢慢加载，**图穷了最多问 8 次**。模型挂不上也能交卷。

不按游戏名写死路线。hidden 集不是公开的 ls20/vc33。
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
