# ARC Prize 2026 — ARC-AGI-3（公开本调研后重写）

上一版只按官方文档自己推，**没对着别人的 kernel 改**。这一版先搜公开 notebook，再合成：

- 能自己跑、反复出现的有效做法：抹 HUD 哈希图、先走后点、自环标死、软导航、迷路关卡 RESET、有限步数
- 公开榜头部的 duck/TAAF **不搬**（要额外数据包）
- Gemma-4 只在图穷了才问

打开 notebook 前两格看方案。核心：**教程关收割技能纸，后面关 A* 追赢过的颜色。**

Save and Run All 不打游戏。真打要点 Submit to Competition。

单测：`python3 -m unittest test_world_agent.py`（含官方封顶公式、跨关记颜色）。

- 竞赛来源不改：`arc-prize-2026-arc-agi-3`
- 无网；机器：`NvidiaRtxPro6000`
- 模型：`gemma-4-31b-it`（卡住才问）

单测：`python3 -m unittest test_world_agent.py`  
改 agent 后：`python3 build_notebook.py`

Kaggle：https://www.kaggle.com/code/chutianqiu/arc-prize-2026-arc-agi-3-starter
