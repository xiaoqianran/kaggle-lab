# kaggle-lab

Kaggle 实验与工具工作区。已用 API Token 登录账号 **seachenbgdy**。

## 认证状态

| 方式 | 位置 | 状态 |
|------|------|------|
| 新版 API Token 文件 | `~/.kaggle/access_token` | 已配置 |
| 环境变量 | `KAGGLE_API_TOKEN` | 推荐在 shell 中 export |
| 认证方式 | `ACCESS_TOKEN` | 已验证可用 |

```bash
# 验证登录
export PATH="/workspace/2/kaggle-lab/.venv/bin:$PATH"
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle config view
kaggle competitions list --page-size 5
```

## 本地 CLI

本目录已创建虚拟环境并安装 `kaggle>=2.2.3`：

```bash
source /workspace/2/kaggle-lab/.venv/bin/activate
kaggle --help
```

常用命令：

```bash
kaggle competitions list
kaggle competitions download -c titanic -p ./data
kaggle datasets list -s "keyword"
kaggle kernels list -m   # 自己的 notebooks
```

## MCP（Model Context Protocol）

### 官方 MCP（推荐）

- 文档：https://www.kaggle.com/docs/mcp  
- 端点：`https://www.kaggle.com/mcp`  
- 协议：Streamable HTTP  
- 鉴权：`Authorization: Bearer <KAGGLE_API_TOKEN>`  
- 能力：约 70 个工具（竞赛 / 数据集 / Notebook / 模型 / 论坛 / Writeup / Benchmark 等）

Grok 配置示例（`~/.grok/config.toml`）：

```toml
[mcp_servers.kaggle]
url = "https://www.kaggle.com/mcp"
enabled = true

[mcp_servers.kaggle.headers]
Authorization = "Bearer ${KAGGLE_API_TOKEN}"
```

或 CLI：

```bash
grok mcp add --transport http kaggle https://www.kaggle.com/mcp \
  --header "Authorization: Bearer $KAGGLE_API_TOKEN"
```

本项目也提供了 `.grok/config.toml`（项目级 MCP），重启会话后生效。

### 社区 MCP

| 项目 | 说明 |
|------|------|
| [arrismo/kaggle-mcp](https://github.com/arrismo/kaggle-mcp) | 数据集搜索 / 下载 / EDA prompt |
| [Dishant27/kaggle-MCP](https://github.com/Dishant27/kaggle-MCP) | 竞赛交互 |
| [54yyyu/kaggle-mcp](https://github.com/54yyyu/kaggle-mcp) | 竞赛 / 数据集 / Kernel |

优先使用 **官方 MCP**。

## Skills

| 来源 | 安装 | 说明 |
|------|------|------|
| [shepsci/kaggle-skill](https://github.com/shepsci/kaggle-skill) | `npx skills add shepsci/kaggle-skill` | 非官方一体化 skill：凭证、竞赛、数据集、notebook、论坛、benchmark、徽章；并捆绑官方 MCP |
| 本环境内置 skills | 无 kaggle 专用 skill | 当前 Grok skills 目录下未发现 kaggle skill |

如需安装 skill 到本机：

```bash
# skills.sh 通用安装
npx skills add shepsci/kaggle-skill

# 或手动克隆
git clone https://github.com/shepsci/kaggle-skill.git /workspace/2/kaggle-lab/vendor/kaggle-skill
```

## AI Models / Model Proxy（Daily $10 · Monthly $100）

账号里的 **Daily AI Models / Monthly AI Models** 额度走 **Kaggle Model Proxy**，不是官方 MCP 那 70 个工具。

| 项 | 值 |
|----|-----|
| 拿临时凭证 | `kaggle b auth -y --env-file .env.model-proxy`（约 1 小时过期） |
| OpenAI 兼容 base | `{MODEL_PROXY_URL}/openapi`（当前多为 `https://mp-staging.kaggle.net/models/openapi`） |
| 列模型 | `kaggle b t models` |
| 本仓调用脚本 | `python call_ai_model.py --refresh -m google/gemini-3-flash-preview "你好"` |
| 模型全表 | `kaggle_ai_models.txt` / `.csv` / `.json`（便于筛选） |

```bash
source .venv/bin/activate
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle b auth -y --env-file .env.model-proxy
python call_ai_model.py --refresh -m openai/gpt-5.4-nano-2026-03-17 "Say hi"
```

支持标准 OpenAI **function / tool calling**（模型返回 `tool_calls`，由本地代码执行工具）。

## 推荐参考的开源项目

以下项目与 **Model Proxy / Community Benchmarks / Agent 评测** 相关，适合对照实现。  
生态仍偏「评测与基准」，而不是通用生产聊天网关。

### 直接使用 Model Proxy 的官方仓库（优先）

| 项目 | 链接 | 做什么 |
|------|------|--------|
| **kaggle-benchmarks** | https://github.com/Kaggle/kaggle-benchmarks | 核心 SDK：`@kbench.task`、`llm.prompt`、断言、tool calling、Judge LLM、多模态；本地/Notebook 调 Model Proxy 跑评测 |
| **write-kaggle-benchmarks**（skill） | https://github.com/Kaggle/kaggle-skills/tree/main/write-kaggle-benchmarks | 给 AI agent 用的技能：`kaggle b auth/init` → 写 task → `push/run` → 查状态与结果 |
| **kaggle-skills**（全集） | https://github.com/Kaggle/kaggle-skills | Kaggle 官方 agent skills 仓库（含上面的 write-kaggle-benchmarks） |
| **kaggle-benchmarks-reference** | https://github.com/Kaggle/kaggle-benchmarks-reference | 参考实现：OSWorld、The Agent Company 等 **Agent 基准** 适配到 Kaggle |
| **kaggle-benchmark-harbor-starter-template** | https://github.com/Kaggle/kaggle-benchmark-harbor-starter-template | Harbor 框架的 Agentic Benchmark 起步模板 |
| **kaggle-cli** | https://github.com/Kaggle/kaggle-cli | 官方 CLI：`kaggle b auth/init/tasks/...` 是本地拿 Model Proxy 与跑任务的入口 |

文档与示例：

- Cookbook：https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md  
- Quick Start：https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md  
- 官方演示视频：https://www.youtube.com/watch?v=c7B8vyehyUA（本地写 eval 并多模型跑）  
- Community Benchmarks 公告：https://www.kaggle.com/product-announcements/667898  

本地最小闭环（与官方项目一致）：

```bash
git clone https://github.com/Kaggle/kaggle-benchmarks.git vendor/kaggle-benchmarks
git clone https://github.com/Kaggle/kaggle-skills.git vendor/kaggle-skills
git clone https://github.com/Kaggle/kaggle-benchmarks-reference.git vendor/kaggle-benchmarks-reference
git clone https://github.com/Kaggle/kaggle-benchmark-harbor-starter-template.git vendor/kaggle-benchmark-harbor-starter

# 在本 lab 或任意目录
kaggle b init -y
# 编辑 example_task.py 或自建 @kbench.task
kaggle b t push my-task -f example_task.py --wait
kaggle b t run  my-task -m gemini-3-flash-preview --wait
```

### 相关但路径不同的项目

| 项目 | 链接 | 说明 |
|------|------|------|
| [shepsci/kaggle-skill](https://github.com/shepsci/kaggle-skill) | 同上 Skills 节 | 竞赛/数据集/Notebook + **官方 Kaggle MCP**；Benchmark 只是其中一块 |
| [openai/mle-bench](https://github.com/openai/mle-bench) | https://github.com/openai/mle-bench | 用 Kaggle **竞赛** 评 AI 工程师；**不走** Model Proxy 聊天 API |
| [khiwniti/kaggle-llm-api](https://github.com/khiwniti/kaggle-llm-api) | https://github.com/khiwniti/kaggle-llm-api | 在 Notebook 里跑 Ollama/vLLM 再暴露 OpenAI 兼容端点；用 **GPU 额度**，不是 AI Models $ |

### 用法对照（这些项目在做什么）

```
                    ┌─────────────────────────────┐
                    │  Kaggle Model Proxy         │
                    │  (Daily $10 / Monthly $100) │
                    └─────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
 kaggle-benchmarks          write-kaggle-benchmarks    harbor / OSWorld
 自定义评测任务               Agent 自动写任务并跑         长程 Agent 基准
 工具调用 / 多模态            多模型对比 + 成本/延迟        环境交互
 Judge LLM 打分               push/run 到 Kaggle           Docker 沙箱
```

主流不是「当生产 Chat 后端」，而是：自定义评测、多模型对比、Agent/tool-use 基准、本地写 task 云端扣额度跑。

## 目录结构

```
kaggle-lab/
  .venv/                      # Python 虚拟环境 + kaggle / kaggle-benchmarks
  .grok/                      # 项目级 Grok MCP 配置
  data/                       # 数据集下载目录（默认）
  notebooks/                  # 本地 notebook 草稿
  call_ai_model.py            # Model Proxy 简易调用脚本
  kaggle_ai_models.txt        # 模型全表（人读）
  kaggle_ai_models.csv        # 模型全表（表格筛选）
  kaggle_ai_models.json       # 模型全表（程序读）
  .env.model-proxy            # Model Proxy 临时凭证（勿提交）
  README.md
  vendor/                     # 可选：clone 上述参考仓库
```

## 安全提示

- 不要把 API Token / `MODEL_PROXY_API_KEY` 提交到 git。
- Token 若曾出现在聊天记录中，建议在 https://www.kaggle.com/settings/api 轮换。
- 优先使用 `~/.kaggle/access_token` 或环境变量，不要硬编码进代码。
- `.env` / `.env.model-proxy` 已在 `.gitignore` 思路中排除敏感文件时请确认生效。
