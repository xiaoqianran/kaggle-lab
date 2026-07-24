# kaggle-lab

Kaggle 实验台：编号 + 主题目录做最小可跑实验（风格对齐 [LightningAI-Lab](https://github.com/xiaoqianran/LightningAI-Lab) / cuda-lab）。

账号：**seachenbgdy**（`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`）。

## 结构

```text
main.py                 # 入口，调度到 001 / 002 / 003 / 004 ...
common/                 # 共享：Model Proxy 客户端
001-model-proxy/        # 刷新凭证 + chat 调用
002-tool-call/          # function calling 演示
003-list-models/        # 列出 / 导出 AI Models 全表
004-sae/                # Standardized Agent Exam 客户端
docs/                   # MCP / Skills / SAE 文档（见 docs/README.md）
kaggle_ai_models.*      # 模型清单（003 dump 同步到根目录）
```

命名约定：`NNN-topic`（序号 + 主题）。`python main.py 001 …` 等短号在唯一时可解析到对应目录。

## 环境

```bash
source .venv/bin/activate          # 仓库根虚拟环境
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
# 依赖：kaggle>=2.2.3、openai、kaggle-benchmarks（可选）
```

## 用法

```bash
# 001 Model Proxy：拿临时 key + 对话（扣 Daily $10 / Monthly $100）
python main.py 001 auth
python main.py 001 chat "用一句话介绍 Kaggle"
python main.py 001 chat -m openai/gpt-5.4-nano-2026-03-17 "1+1=?"
# 或全名
python main.py 001-model-proxy chat --refresh "hello"

# 002 工具调用
python main.py 002 run
python main.py 002 run -m google/gemini-3-flash-preview

# 003 模型列表
python main.py 003 list
python main.py 003 dump

# 004 SAE 标准化 Agent 考试（开考前务必 --i-accept；最多 3 次）
python main.py 004 register --name "YourUnique-42" --model grok-build --agent-type Grok --i-accept
python main.py 004 start --i-accept
python main.py 004 answer-proxy
python main.py 004 submit --i-accept
```

也可进入实验目录：

```bash
cd 001-model-proxy && python run.py chat "hello"
cd 002-tool-call && python run.py run
cd 003-list-models && python run.py list
```

## 实验一览

| 目录 | 作用 |
|------|------|
| `001-model-proxy` | `auth` / `chat` — OpenAI 兼容 Model Proxy |
| `002-tool-call` | `run` — weather 工具两轮调用 |
| `003-list-models` | `list` / `dump` — 模型全表 txt/csv/json |
| `004-sae` | SAE 考试客户端：register / start / answer / submit |

## 认证与额度

| 项 | 说明 |
|----|------|
| 账号 API | `~/.kaggle/access_token` 或 `KAGGLE_API_TOKEN`（KGAT_…） |
| Model Proxy | `kaggle b auth` → `.env.model-proxy`（约 1h 过期，勿提交） |
| AI Models 额度 | Daily **$10** / Monthly **$100** |
| GPU/TPU 额度 | 另一套（Notebook 训练），`kaggle quota` |

```bash
kaggle config view
kaggle quota
```

## MCP 与 Skills 文档（已抓取）

本地整理：

| 路径 | 内容 |
|------|------|
| [docs/mcp-and-skills.md](docs/mcp-and-skills.md) | MCP + Skills **总览** |
| [docs/mcp-tools.md](docs/mcp-tools.md) | 官方 MCP **70 工具**清单 |
| [docs/skills/](docs/skills/) | 官方 kaggle-skills 镜像 |
| [docs/raw/](docs/raw/) | CLI / JSON 原文 |

在线：

- MCP：https://www.kaggle.com/docs/mcp · 端点 `https://www.kaggle.com/mcp`
- Skills：https://github.com/Kaggle/kaggle-skills
- 项目配置：`.grok/config.toml`（Bearer `KAGGLE_API_TOKEN`）

**MCP ≠ Model Proxy**：MCP 操作竞赛/数据集/Notebook；Proxy 调大模型扣 $10/$100。

## 推荐参考的开源项目

### 直接使用 Model Proxy 的官方仓库

| 项目 | 链接 | 做什么 |
|------|------|--------|
| **kaggle-benchmarks** | https://github.com/Kaggle/kaggle-benchmarks | SDK：评测任务、tool calling、Judge LLM |
| **write-kaggle-benchmarks** | https://github.com/Kaggle/kaggle-skills/tree/main/write-kaggle-benchmarks | Agent skill：auth → 写 task → push/run |
| **kaggle-skills** | https://github.com/Kaggle/kaggle-skills | 官方 agent skills 全集 |
| **kaggle-benchmarks-reference** | https://github.com/Kaggle/kaggle-benchmarks-reference | OSWorld 等 Agent 基准参考实现 |
| **harbor-starter** | https://github.com/Kaggle/kaggle-benchmark-harbor-starter-template | Harbor Agentic Benchmark 模板 |
| **kaggle-cli** | https://github.com/Kaggle/kaggle-cli | CLI：`kaggle b auth/tasks/...` |

```bash
# 可选克隆到 vendor/（已 gitignore）
git clone https://github.com/Kaggle/kaggle-benchmarks.git vendor/kaggle-benchmarks
git clone https://github.com/Kaggle/kaggle-skills.git vendor/kaggle-skills
```

### 相关但路径不同

| 项目 | 说明 |
|------|------|
| [shepsci/kaggle-skill](https://github.com/shepsci/kaggle-skill) | 竞赛/数据集 + 官方 Kaggle MCP |
| [openai/mle-bench](https://github.com/openai/mle-bench) | 竞赛评 AI 工程师，不走 Model Proxy |
| [khiwniti/kaggle-llm-api](https://github.com/khiwniti/kaggle-llm-api) | Notebook 起 Ollama/vLLM（GPU 额度） |

## 约定

- 每个实验自包含：`run.py` + `README.md`
- 共享逻辑放 `common/`
- 密钥：`.env` / `.env.model-proxy` / `access_token` 不入库
- 数据默认不入库（`data/`）

## 仓库

- 远程：https://github.com/xiaoqianran/kaggle-lab
