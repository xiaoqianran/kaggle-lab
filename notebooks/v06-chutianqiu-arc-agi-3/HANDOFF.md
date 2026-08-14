# 换环境必读（2026-08-14）

下一任 agent / 下一场对话：**先读这个文件**，再动代码。
方案细节见 [DESIGN.md](DESIGN.md)，公开本对照见 [SURVEY.md](SURVEY.md)。
源码以 `my_agent.py` 为准；`arc-prize-2026-arc-agi-3-starter.ipynb` 必须用 `python3 build_notebook.py` 重建，不要手改 ipynb。

---

## 30 秒

- 仓库：`xiaoqianran/kaggle-lab`，本目录 `notebooks/v06-chutianqiu-arc-agi-3/`
- GitHub 用户：`xiaoqianran`。Kaggle 用户：`chutianqiu`。**密钥不入库**。
- 竞赛 slug：**不要改**：`arc-prize-2026-arc-agi-3`
- Kernel：https://www.kaggle.com/code/chutianqiu/arc-prize-2026-arc-agi-3-starter
- **2026-08-14 已 Submit to Competition，已出分。**
  - 提交号 `55511330`
  - kernel 版本 **v10**（`scriptVersionId=342407362`）
  - 公开榜 **0.17**（上一笔 2026-07-24 的 `54953206` 是 **0.06**）
  - 公开榜 **1503 / 2310**（2026-08-14 19:51 UTC 拉榜）；第一名 **2.70**（队名 cstl）
  - 队 id `16593868`
  - 同一天公开榜上 **63 队卡在整整 0.17**，像同一档「只过教程关」的封顶分，不是随机噪声
- 当天额度已用完：`numToday=1`，`numAllowedNow=0`。Kaggle 这赛每天大约 1 次正式提交。**不要在 8 月 14 日再交。**
- **2026-08-14 已 `kernels push` 出 v11**（compose-click）。Save and Run All COMPLETE，日志有 `MODEL_PATH` 和 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，并写明「没有打任何一局游戏」。**明天用 `-v 11` Submit**，不要交这份假 parquet。
- Save and Run All 十几秒 COMPLETE **不是分数**。只有 Submit to Competition 才会打 hidden 集、改榜。

---

## 线上成绩（已核实，不是猜测）

| 提交号 | 时间 UTC | 说明 | 状态 | publicScore |
|---|---|---|---|---|
| 54953206 | 2026-07-24 14:22:04 | v1 legal-action explorer | COMPLETE | 0.06 |
| **55511330** | **2026-08-14 18:09:39** | **skill-hunt v10：教程技能纸 + A-star hunt，MAX_ACTIONS=8000** | **COMPLETE** | **0.17** |

提交命令（已经跑过，不要当天再交）：

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -f submission.parquet \
  -k chutianqiu/arc-prize-2026-arc-agi-3-starter \
  -v 10 \
  -m "skill-hunt v10: tutorial skill sheet + A-star hunt, MAX_ACTIONS=8000"
```

- 这是 **code competition**：必须带 `-k` 和 `-v`。**禁止**把 Save and Run All 那份 890 字节假 parquet 当文件直接交上去。
- 交上去之后 API 先 `PENDING`（`totalBytes=0`，`errorDescription` 空）。这是正常的：Kaggle 再开 sidecar 打 hidden 集。
- 这次 18:09 交，约 18:59 变成 COMPLETE、0.17。大约 50 分钟，**不是 9 小时**。含义见下面「为什么是 0.17」。
- 查分：

```bash
kaggle competitions submissions -c arc-prize-2026-arc-agi-3
```

Python：

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest
api = KaggleApi(); api.authenticate()
print(api.competition_get_submission_limits("arc-prize-2026-arc-agi-3"))
with api.build_kaggle_client() as kaggle:
    req = ApiGetSubmissionRequest(); req.ref = 55511330
    print(kaggle.competitions.competition_api_client.get_submission(req))
```

下载 scored parquet 试过 `ApiDownloadSubmissionRequest`，GCS 404。没有逐局明细。别在这上面浪费时间。

Kernel 的 `kaggle kernels status` / `kaggle kernels logs` 永远是 **Save and Run All 那次**（COMPLETE、十几秒、打印「没有打任何一局游戏」）。**看不到**竞赛 sidecar 日志。别被这个 COMPLETE 骗了。

---

## 这个竞赛到底在考什么（官方，不是公开本）

交互式 64x64，颜色 0 到 15。**没有说明书。** 公开集 `ls20` / `ft09` / `vc33` 和 hidden 集机制刻意不重叠。**禁止按游戏名写死。**

动作：

- ACTION1 到 ACTION4：默认上 / 下 / 左 / 右，但要用平移实验认键，不要假设永远对
- ACTION5：交互（踩上按一下）
- ACTION6：点击，要带坐标
- ACTION7：undo
- 状态 `NOT_PLAYED` 或 `GAME_OVER`：**只能 RESET**，否则服务器 400
- 竞赛模式允许 **关卡 RESET**，不允许整局重开。我们最多 3 次 stranded-reset

官方技术报告四支柱：**探索、建模、自己定目标、规划执行。**

关卡：至少 6 关。第 1 关是教程，故意简单，用来教会核心交互。后面关要 **组合前面学会的机制**。

计分（代码在 `rhae_level_score` / `rhae_game_score`）：

1. 单关：`(人步 / AI步)^2`，上限 1.15。人 10 步你 100 步，单关只剩 0.01。
2. 整局：关卡号加权平均。5 关权重 1+2+3+4+5=15。
3. **没从第 1 关起连续打完，会封顶。** 5 关只打完前 3 关，封顶 6/15=0.40。前 3 关再快也抬不上去。
4. 竞赛总分是各 hidden 游戏整局分的聚合（我们交上去看到的是一个 publicScore）。

所以：教程关可以稍慢，**后面关必须打到**。把 `MAX_ACTIONS` 写成 1500 会把第 5、6 关掐死（6 关预算加起来已经超过 1500）。当前整局上限 **8000**。

时间账：`GLOBAL_TIME_LIMIT_S = 8*60*60`，预留 `GLOBAL_RESERVE_S = 20*60`。单关 `level_action_budget = 160 + 50*idx`，封顶 700；步数到 700 也会放弃本游戏。

硬件 / 规则硬约束：

- notebook-only，**无网**（`enable_internet: false`）
- 机器：`NvidiaRtxPro6000`
- 模型只挂：`google/gemma-4/transformers/gemma-4-31b-it/1`
- **不要改 `competition_sources`**
- 不绑 Tufa / TAAF / duck / Sandwich 等第三方 dataset（少挂一个就交不了）
- 不读游戏源码做离线 BFS（hidden 集没有源码）
- 不用 vLLM 额外 wheelhouse（我们没挂那个 dataset）
- Gemma 用 transformers 4-bit，失败也必须能交卷
- Swarm = 多线程并行打不同游戏；LLM 有 `_LLM_LOCK`
- 模型内部思考不计步，但逐步问 31B 会把墙钟吃光。最多 `LLM_MAX_CALLS = 8`，`LLM_MAX_NEW_TOKENS = 48`，图穷了才问

---

## 两种运行（最容易搞错，已经坑过一次）

| 你点的 | 耗时 | 环境变量 | 打不打 hidden | 榜分 |
|---|---|---|---|---|
| Save and Run All / `kaggle kernels push` | 十几秒 | 没有 `KAGGLE_IS_COMPETITION_RERUN` | 不打。只装轮子、写 agent、写假 parquet | 不变 |
| Submit to Competition | 数十分钟到数小时 | 有 `KAGGLE_IS_COMPETITION_RERUN`，网关 `gateway:8001` | 打 | 才会变 |

Notebook 四格代码（`build_notebook.py`）：

1. 无网 pip：`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels` 装 `arc-agi` + `python-dotenv`
2. `%%writefile /tmp/my_agent.py` 整份智能体
3. **仅当竞赛重跑**：等 gateway、拷官方 `ARC-AGI-3-Agents`、塞 MyAgent、重写 `agents/__init__.py`（避开 langgraph 等没装的库）、`python main.py --agent myagent`
4. **仅当不是竞赛重跑**：写假 `submission.parquet`（让提交按钮出现），打印 GPU / MODEL_PATH，明确说「没有打任何一局游戏」

假 parquet 列：`row_id, game_id, end_of_game, score`。Save and Run All 产物大约 890 字节。竞赛重跑应由官方 `main.py` 写出真正的 parquet。

v10 Save and Run All 日志要点（sidecar 不是这个）：

- `MODEL_PATH /kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1` 找到了
- GPU：`NVIDIA RTX PRO 6000 Blackwell Server Edition`
- `这是 Save and Run All：没有打任何一局游戏。`

---

## 当前智能体在干什么（和 SURVEY 不完全一样，以代码为准）

类：

- `SkillSheet`：跨关技能纸。`genre`、`goal_colors`（多个）、`useful_click_colors`、`win_kinds`
- `GraphMemory`：抹 HUD 后哈希图；自环标死；未试动作库存；闪烁掩码
- `MyAgent`：每步决策

每步大致顺序（`choose_action`）：

1. `NOT_PLAYED` / `GAME_OVER` → 只 RESET
2. `levels_completed` 变了 → 记下赢的颜色和动作种类；**地图清空，键位保留**
3. 若上一步是 ACTION1-4 且画面平移 → 认角色颜色/大小，更新 `dir_map`，记 `move`
4. 抹 HUD、哈希、上一步没变就标死，走路撞墙就把墙格子放进 `blocked`
5. 点到了东西 → `note_effect("click")`
6. `pick_hunt_candidates` + `plan_hunt`：只追小色块；优先赢过的颜色；A* 走不到就换下一个候选
7. 放弃条件：全局时间没了，或硬顶（后面关高于 700）且没有可达猎点
8. `pending_interact`：A* 下一步就会踩上目标时，先走再立刻 ACTION5
9. **还没认出角色**：新画面可以先 ACTION5。认出来之后 **不要每个新格子都按**，否则走路关步数翻倍
10. 已有角色和可达猎点：A* 走近；已在目标格则 ACTION5。纯点选只在教程关禁止 A*
11. A* 走不到且已过教程关：`compose-click` 先点小开关，点开了清 `blocked`
12. genre 是 `click`：先点后走（`kinds = ("c", "s")`），否则先走后点
13. ABAB 振荡：把正在走的边当已探索
14. 图穷：关卡 RESET（最多 3 次） → 一次 undo → Gemma → leftover 随机

关键常量（`my_agent.py`）：

- `MAX_ACTIONS = 8000`
- `REPLAY_RESETS = 3`
- `MAX_CLICKS = 16`，`CLICK_KEEP = 4`
- `NAV_BIAS = 0.7`
- `LLM_MAX_CALLS = 8`
- 新游戏 `simple_order = [5] + 打乱的方向`，但真正每格先按只发生在角色未知时

过关判断用 **`levels_completed`**，不用 `score`。
`GameAction.from_name` **每次新建**，别改枚举上的旧坐标。

单测：`python3 -m unittest test_world_agent.py`（20 个，含组合关开门）。仓库没装 `arcengine`，测试会先塞假模块。

---

## 为什么公开榜是 0.17，不是 2.x

这是交接时最重要的判断，**没有逐局日志，是推断，但要带着走**：

1. 0.17 相对 0.06 是真提升（大约 2.8 倍），说明 skill-hunt + 8000 步上限比旧 explorer 有效。
2. 公开榜 0.14 到 0.20 挤了三百多队。这像「只打完前一两关就被加权封顶」的分数带，不像「完整通关多局」。
3. 粗算：6 关权重和 21。只连续打完第 1 关，封顶 1/21 约 0.048；打完前 2 关，封顶 3/21 约 0.14。若干 hidden 游戏平均到 0.17，很像 **教程关能过、组合关过不去**。
4. 竞赛重跑大约 50 分钟就 COMPLETE，而代码里墙钟上限是 8 小时。说明 **不是算力跑满**，更像游戏局很快结束（过关很少、或 700 步放弃、或 hidden 局数不多且每局早停）。
5. 第一名 2.70。要上 1.0 以上，必须在多局里打完后面的加权关，而不能只卷教程关效率。
6. 没有 sidecar 日志，下一轮不要幻想「再 Save and Run All 一次就能看到 hidden 回放」。要进步只能：改 agent → 本地假网格单测 → 推 kernel → **第二天**再 Submit。

下一轮主攻（按优先级）：

1. **组合关：A* 走不到就先点开关。** 已写进代码（`plan_hunt` + `compose-click`）。技能纸现在记多个 `goal_colors`、点过会变的颜色、赢的动作种类。单测 `test_compose_click_opens_door` 覆盖「墙挡住出口必须先点」。
2. **后面关不要 700 步一刀切。** `should_abandon(..., has_plan=)`：有可达猎点时后面关硬顶放到 700+80*关号。
3. **点选关 vs 走路关。** 纯点选只在教程关禁止 A*。后面关即使教程是点选，也允许走路组合。
4. **不要为了省第 1 关步数而放弃整局。** 封顶比平方效率更狠。
5. Gemma 继续当顾问，不要每步问。不要挂 vLLM / 第三方 dataset。
6. 不要把 `MAX_ACTIONS` 改回 1500 或改成无限。
7. **8 月 14 日额度已用完。** 改完先 `kernels push`，**第二天**再 `competitions submit`。

---

## 明确禁止（已经讨论过、不要再犯）

- 改 `competition_sources`
- 开互联网
- 按 `ls20` / `vc33` / `ft09` 或任何 `game_id` 写死
- 绑 Tufa / duck / Sandwich / TAAF 额外数据集
- 读 `/kaggle/input` 里不存在的游戏源码做离线 BFS
- 同一张卡上 31B + 再训 CNN
- `MAX_ACTIONS = inf`
- 把假 parquet 当正式提交文件
- 当天额度用完后再 `kaggle competitions submit`（会失败或浪费）
- 只 push kernel 不点 Submit，就以为上榜了
- 手改 ipynb
- 把 Kaggle token / `KGAT_...` 写进仓库
- 用 `cursor[bot]` 的 `git push`（对这个仓库 **403**）

---

## 怎么改、测、推 kernel、再交

工作目录：`notebooks/v06-chutianqiu-arc-agi-3/`

```bash
python3 -m unittest test_world_agent.py
python3 build_notebook.py
```

推 kernel（这只是 Save and Run All，**不是交卷**）：

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
cd notebooks/v06-chutianqiu-arc-agi-3
kaggle kernels push
kaggle kernels status chutianqiu/arc-prize-2026-arc-agi-3-starter
```

等 COMPLETE 且日志里有 `MODEL_PATH` 和 GPU 之后，**确认当天还有额度**再：

```bash
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -f submission.parquet \
  -k chutianqiu/arc-prize-2026-arc-agi-3-starter \
  -v NEW_VERSION \
  -m "写清楚改了什么"
```

`kernel-metadata.json` 必须保持：

- `id`: `chutianqiu/arc-prize-2026-arc-agi-3-starter`
- `enable_internet`: false
- `enable_gpu`: true
- `machine_shape`: `NvidiaRtxPro6000`
- `competition_sources`: 只有 `arc-prize-2026-arc-agi-3`
- `model_sources`: `google/gemma-4/transformers/gemma-4-31b-it/1`

---

## 这个仓库的 GitHub 写入方式（血泪）

- 云端 agent 用 `cursor[bot]` 做 `git push` → **403**。不要再试。
- 能写进去的路径：GitHub MCP，身份是 **xiaoqianran**。工具：`create_branch`、`push_files`、`create_or_update_file`、`create_pull_request`、`merge_pull_request`。
- **`push_files` 的文件内容里不能出现英文小于号字符。** 比较请用 `_le` / `_lt`，或中文「小于 / 不超过」。本目录 `my_agent.py` 已经这样写了。
- 分支名惯例：`cursor/短英文-8a91`（全小写）。
- 用户说过「先合并」时，用 squash merge 进 `main`。换环境的人默认看 `main`，不要只把交接留在 PR 分支。

已合并过的本目录 PR：

| PR | 内容 | main 上的提交 |
|---|---|---|
| 2 | 早期 RHAE world-model starter | 已合 |
| 3 | 公开 notebook 调研 + 图探索 | `93041fb` |
| 4 | skill-hunt 智能体 | squash `d030503` |
| 5 | 最终抛光：8000 步、收紧 hunt | squash **`5f0133e`**（交接文档写入前的代码 HEAD） |

`main` 上本目录关键文件（代码）：

- `my_agent.py` — 智能体
- `build_notebook.py` — 重建 notebook
- `test_world_agent.py` — 17 个单测
- `kernel-metadata.json`
- `arc-prize-2026-arc-agi-3-starter.ipynb` — 生成物

---

## 账号与密钥（写位置，不写值）

- Kaggle：`chutianqiu`
- Token：环境变量 `KAGGLE_API_TOKEN`，或 `~/.kaggle/access_token`（`KGAT_...`）
- `export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"`
- **永远不要 commit token。** 根 README 也写了密钥不入库。

GitHub：`xiaoqianran/kaggle-lab`。本竞赛代码不在 colab-lab。云端 workspace 若是别的仓库，**改这里。**

---

## 公开本调研（结论指针，细节在 SURVEY.md）

留：抹 HUD 哈希、自环标死、先走后点（点选关反过来）、迷路关卡 RESET、平方分预算、键位跨关、Gemma 不当每步策略。

丢：Goose 无限步、Tufa 第三方 dataset、离线读源码 BFS、vLLM 额外包、按公开游戏名调参、31B+CNN 同卡。

注意：SURVEY 里有一句「每个新画面先 ACTION5」。**最终代码已经改掉。** 认角色之后不再每格都按。以 `my_agent.py` 和本文件为准。

---

## 下一任 checklist

1. 读本文件 + DESIGN.md + `my_agent.py` 的 `choose_action`
2. 跑 `python3 -m unittest test_world_agent.py`
3. 查当天额度，**不要在 0.17 这笔还在同一天时再交**
4. 若要改策略：先加单测（组合关、错误 goal_color、点选误判），再改 `my_agent.py`，再 `build_notebook.py`
5. 用 GitHub MCP 推到本仓库（记住 push_files 不能带英文小于号）
6. `kaggle kernels push`，等 Save and Run All COMPLETE
7. `kaggle competitions submit ... -k ... -v NEW_VERSION`
8. 轮询 submissions，直到 PENDING 变成 COMPLETE 或 ERROR；kernel logs 不是 sidecar
9. 把新提交号、分数、版本追加到本文件这张表

写交接的人当时还在想、但没做成的事：

- 没有 hidden 回放，不知道具体哪几局、打到第几关
- 技能纸太薄，组合关大概率过不去，这是 0.17 最可疑的原因
- 50 分钟跑完像早停，值得在下一版加更清楚的 `print`（level-up / abandon / stranded-reset），即使看不到 sidecar 日志，万一以后能拉 output 也有用
- 不要为了「看起来更 LLM」去每步问 Gemma-4-31B
