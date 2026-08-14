# 公开 notebook 调研（2026-08-14）

后来对照官方技术报告重写了打法：教程关收割技能纸、后面关 A* 追颜色、按加权封顶给预算。公开本仍是调研，主策略以官方四支柱为准。详见 [DESIGN.md](DESIGN.md)。
**线上结果和最终代码差异见 [HANDOFF.md](HANDOFF.md)**（2026-08-14 交卷 0.17）。SURVEY 里「每个新画面先 ACTION5」已被最终代码改掉：只在还没认出角色时先按。

上一版只按官方文档自己推，**没有**对着别人的 kernel 改。这次按你的要求，用 Kaggle API 把比赛里能拉到的公开本搜下来，对照官方 RHAE 规则逐条看：留什么、丢掉什么、为什么。

本地摘录在 `/tmp/arc-public/`（不进仓库）。

## 拉下来对照过的公开本

| 公开本 | 可见分 / 定位 | 核心做法 | 我们的结论 |
|---|---|---|---|
| inversion Just Explore | 官方样本 | 额外 dataset + 探索论文；哈希、自环标死 | **用思路**，不绑它的 dataset |
| inversion / imaad / dynamo Stochastic Goose | 官方样本及变体 | 在线 CNN 预测「哪一步会改画面」；`MAX_ACTIONS = inf` | CNN 不跟 31B 抢显存；无限步和平方分对着干，**不用** |
| pscamillo graph explorer | 图探索代表 | 抹 HUD 再哈希；先 ACTION1-5 再点；自环标死；BFS 回有库存的点；关卡 RESET；**有限** 1500 步；认角色后软偏向出口 | **主策略骨架**。不搬它对 `ls20`/`vc33` 的调参注释 |
| Ash (ashvinsingh) FORGE | 图 + ChangeNet | 同上图；另外训 CNN。写明 hidden 集读不到游戏源码，所以删了离线 BFS | 图要，**CNN 不要**；离线读源码 BFS 在 Kaggle 计分里是空的 |
| nihilisticneuralnet Persistent BFS 0.46 | 约 0.46 | 图 + 8h50 全局时间账 | 时间账要；复杂 PER/MCTS **太重** |
| vyanktesh hybrid BFS+CNN | 混合 | 第一层居然去 `import` 游戏源码做离线 BFS；第二层才是图 | **第一层在 hidden 集不可用**（Ash 已删过）。只借鉴「先简单动作、再点、再 BFS」 |
| Murad Gemma-4-31B 0.86 | 里程碑期第 3 | 额外 vLLM wheelhouse + 逐步问视觉模型；`MAX_ACTIONS=200` | 我们没挂 vLLM 包；逐步问会把 9 小时吃光。**只当顾问** |
| ko0kip Gemma-4 reflection | 纯 LLM | 同上，还要 openai 客户端打本地 vLLM | 同上 |
| gregkamradt GPT-OSS-120B | 官方 LLM 样本 | 原文大意：这卡上不要每步都问 | 同意 |
| Tufa Duck June-30 / Sandwich / Hydra / BlackCat | 公开榜头部（约 1.2） | 别人的 harness + 大模型 + **额外 input 数据集** | **整套不搬**。少挂一个 dataset 就交不了；超时风险也高 |
| jeroencottaar simplified | 教学 | 说明官方仓库很重，最小交卷只要 MyAgent + gateway | 交卷管道我们本来就用官方 starter |

## 别人反复验证、和官方规则也对得上的

1. **计分是平方效率** `(人步/AI步)^2`，上限 1.15。人 10 步你 100 步 = 0.01。Goose 写成无限步，等于跟计分作对。pscamillo 写死 `MAX_ACTIONS = 1500`。
2. **哈希必须抹掉顶/底长条 HUD**。计步条每步都变，不抹就永远是「新画面」。短标记（钥匙/分数点）要留下。
3. **试过且画面没变 → 这个动作在这个点上标死**。Just Explore 论文这点能多过关。
4. **先在全图找没试过的走路/交互，再点**。当前格点完所有点击再走路，导航关会玩死。pscamillo / Ash 都是：本点没简单动作了，就沿已知边走到「还有简单动作」的点。我们额外规定：每个新画面先 ACTION5，避免走进目标格立刻走开。
5. **点哪里**：小、颜色少的色块像按钮。但**完全固定的点击顺序**会把点选关困死（pscamillo 写 vc33 从 10/10 掉到 0/10）。折中：最像按钮的前几个固定，后面打乱。
6. **认角色之后，软偏向「像出口的小色块」**（pscamillo 默认开）。必须是软的：全贪心会撞墙来回抖。只走还没踩过的格子。
7. **闪烁装饰像素**：会把哈希打爆。pscamillo 的动态掩码默认关（他们测过会掉分）。我们只在「同一格反复闪、且不是角色、且面积不大」时才遮。
8. **键位跨关记住**，地图每关清空。
9. **迷路就关卡 RESET**（竞赛模式允许关卡重置，不允许整局重开），RESET 次数要有上限（他们用 3）。
10. **大模型不要每步问**。Murad 要 vLLM 额外包才能到 0.86；我们挂的是 transformers 4-bit，失败也必须能交卷。
11. **`GameAction.from_name` 每次新建**，别改枚举上的旧坐标（公开本 C1 坑）。
12. **用 `levels_completed` 判断过关**，别用 `score`（Goose 注释里改过）。

## 明确不抄

- 不按 `ls20` / `vc33` / `ft09` 写死（Kaggle 计分是 hidden 集）。
- 不绑 Tufa / TAAF / duck / Sandwich 的第三方 dataset。
- 不读游戏源码做离线 BFS（hidden 集没有源码）。
- 不在 RTX 6000 上同时跑 31B 和再训一个 CNN。
- 不把 `MAX_ACTIONS` 设成无限。

## 合成后的主策略

**抹 HUD 的画面哈希图 + 先走后点 + 自环标死 + 软导航 + 迷路 RESET + 平方分预算 + 图穷了才问 Gemma。**
