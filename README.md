# kaggle-lab

Kaggle 实验台：编号 + 主题目录做最小可跑实验（风格对齐 [LightningAI-Lab](https://github.com/xiaoqianran/LightningAI-Lab) / cuda-lab）。

账号 API：`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`（`KGAT_…`）。

---

## SAE 禁考（必读）

> **SAE** = *Standardized Agent Exam* = Kaggle 给 AI Agent 的标准化考试（16 题 / 30 分钟 / 每身份最多 3 次）。  
> **不是** 竞赛，**不是** GPU notebook，**不是** Model Proxy。

| 策略 | 说明 |
|------|------|
| **默认** | **不开考、不注册、不交卷** |
| **用户原话** | 「千万别考」 |
| **Agent 约束** | 禁止 `004 register` / `004 start` / `004 submit` |
| **允许** | 读文档、改代码、**`009 dry-run`**（假卷，不占次数） |

完整说明：**[docs/SAE.md](docs/SAE.md)**  
相关：`004-sae/`（正式客户端，默认勿跑 start）、`009-sae-better/`（只跑 dry-run）。

---

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / …
common/                 # 共享：Model Proxy、usage 账本
001-model-proxy/        # 刷新凭证 + chat
002-tool-call/          # function calling 演示
003-list-models/        # 列出 / 导出 AI Models
004-sae/                # SAE 客户端 ⚠️ 默认禁止 start/submit
005-benchmark-task/     # Benchmarks：task.py + push/run/download
006-judge-llm/          # 多模型作答 + 规则/Judge 排行榜
007-mcp-harness/        # 官方 MCP 薄客户端
008-agent-loop/         # 多轮 ReAct（本地 tools + 可选 MCP）
009-sae-better/         # SAE 增强答题 → 只用 dry-run
010-quota-dashboard/    # GPU/TPU + 本地 AI $ 账本
docs/
  SAE.md                # SAE 是什么 + 禁考清单
  sae-standardized-agent-exam.md
  mcp-and-skills.md …
notebooks/              # Mini-Instruct T4 学习 notebook
kaggle_ai_models.*      # 模型清单（003 dump）
```

命名约定：`NNN-topic`。`python main.py 001 …` 等短号在唯一时可解析到对应目录。

## 环境

```bash
source .venv/bin/activate          # 仓库根虚拟环境（Python ≥3.11）
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
# 依赖：kaggle>=2.2.3、openai、kaggle-benchmarks（005）
```

## 用法速查

```bash
# 001–003 基础
python main.py 001 auth && python main.py 001 chat "hi"
python main.py 002 run
python main.py 003 list

# 004 SAE — 默认不要 register/start/submit（见 docs/SAE.md）

# 005 Benchmarks
python main.py 005 validate
python main.py 005 push && python main.py 005 run -m gemini-3.5-flash

# 006 多模型 + Judge
python main.py 006 run
python main.py 006 show

# 007 MCP
python main.py 007 tools
python main.py 007 competitions --q llm --n 5

# 008 Agent loop
python main.py 008 run "东京天气？再算 17*3"
python main.py 008 run --with-mcp "找一个热门 LLM 竞赛"

# 009 SAE — 仅 dry-run（假卷，不占考试次数）
python main.py 009 dry-run
python main.py 009 dry-run --ensemble

# 010 额度
python main.py 010 show
```

## 实验一览

| 目录 | 作用 | 主要额度 | 备注 |
|------|------|----------|------|
| `001-model-proxy` | auth / chat | AI $ | |
| `002-tool-call` | weather tool 两轮 | AI $ | |
| `003-list-models` | list / dump 模型表 | 无 | |
| `004-sae` | SAE 正式客户端 | SAE 次数 | **默认禁 start** |
| `005-benchmark-task` | task → push → run → download | AI $ | |
| `006-judge-llm` | 多模型 + 规则/Judge 排行榜 | AI $ | |
| `007-mcp-harness` | MCP tools/list + call | 平台 API | |
| `008-agent-loop` | 多轮 tools / MCP | AI $ | |
| `009-sae-better` | SAE 清洗 + ensemble | AI $ | **只用 dry-run** |
| `010-quota-dashboard` | GPU/TPU + 本地 usage | 无 | |

## 认证与额度

| 项 | 说明 |
|----|------|
| 账号 API | `~/.kaggle/access_token` 或 `KAGGLE_API_TOKEN` |
| Model Proxy | `kaggle b auth` → `.env.model-proxy`（约 1h 过期） |
| AI Models | Daily **$10** / Monthly **$100** |
| GPU/TPU | `kaggle quota` / `python main.py 010 gpu` |
| 本地 AI 账本 | `logs/usage.jsonl` |
| SAE 身份 | `~/.kaggle-agent-*`（**与账号 token 不同**；默认不创建） |

## MCP 与 Skills

| 路径 | 内容 |
|------|------|
| [docs/mcp-and-skills.md](docs/mcp-and-skills.md) | MCP + Skills 总览 |
| [docs/mcp-tools.md](docs/mcp-tools.md) | 官方 MCP 工具清单 |
| [docs/SAE.md](docs/SAE.md) | **SAE 说明 + 禁考** |
| [docs/skills/](docs/skills/) | 官方 kaggle-skills 镜像 |
| `.grok/config.toml` | MCP Bearer 配置 |

**MCP ≠ Model Proxy ≠ SAE**：平台操作 / 调大模型 / Agent 考试，三套东西。

## 约定

- 每个实验：`run.py` + `README.md`
- 共享逻辑：`common/`
- 密钥 / artifacts / logs / vendor 不入库
- **不自动开 SAE 考**

## 仓库

- 远程：https://github.com/xiaoqianran/kaggle-lab
