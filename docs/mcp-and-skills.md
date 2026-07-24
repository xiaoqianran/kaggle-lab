# Kaggle MCP 与 Skills 抓取总览

> 整理时间：2026-07-24  
> 用途：004+ 复杂实验前的官方能力地图

---

## 1. 三条能力线（别混）

| 能力线 | 是什么 | 鉴权 | 本仓库入口 |
|--------|--------|------|------------|
| **官方 MCP** | `https://www.kaggle.com/mcp` 约 70 工具：竞赛/数据集/Notebook/论坛/Hackathon… | Bearer `KAGGLE_API_TOKEN` | `.grok/config.toml`；见 [mcp-tools.md](./mcp-tools.md) |
| **Model Proxy** | 调 Gemini/Claude/GPT 等，扣 Daily $10 / Monthly $100 | `kaggle b auth` → `MODEL_PROXY_*` | `python main.py 001` |
| **Agent Skills** | `SKILL.md` 指令包，教 agent 怎么用 CLI/MCP/SDK | 装到 agent 即可 | [skills/](./skills/) |

```
Agent (Grok / Claude Code / …)
 ├── MCP  ──→  kaggle.com/mcp          # 平台操作
 ├── Skill ─→  write-kaggle-benchmarks  # 工作流剧本
 └── CLI  ──→  kaggle b / kernels      # 本地命令
              └── Model Proxy          # LLM 推理额度
```

---

## 2. 官方 MCP

### 2.1 配置（Grok / 通用）

```toml
# .grok/config.toml 或 ~/.grok/config.toml
[mcp_servers.kaggle]
url = "https://www.kaggle.com/mcp"
enabled = true

[mcp_servers.kaggle.headers]
Authorization = "Bearer ${KAGGLE_API_TOKEN}"
```

```bash
grok mcp add --transport http kaggle https://www.kaggle.com/mcp \
  --header "Authorization: Bearer $KAGGLE_API_TOKEN"
```

JSON 客户端示例：

```json
{
  "mcpServers": {
    "kaggle": {
      "url": "https://www.kaggle.com/mcp",
      "headers": {
        "Authorization": "Bearer ${KAGGLE_API_TOKEN}"
      }
    }
  }
}
```

### 2.2 工具规模（本环境实测）

- **70 tools**（`tools/list`）
- 分类摘要见 [mcp-tools.md](./mcp-tools.md)
- 原始 JSON： [raw/mcp-tools.json](./raw/mcp-tools.json)

| 类别 | 数量 | 代表工具 |
|------|------|----------|
| competition | ~20 | `search_competitions`, `submit_to_competition`, `get_competition_leaderboard` |
| dataset | ~10 | `search_datasets`, `download_dataset`, `upload_dataset_file` |
| notebook | ~10 | `save_notebook`, `create_notebook_session`, `get_accelerator_quota` |
| model | ~10 | `list_models`, `create_model`, `download_model_variation_version` |
| forum / writeup / hackathon | ~14 | `list_hackathon_write_ups`, `get_writeup`, `list_forum_topics` |
| benchmark | ~2 | `create_benchmark_task_from_prompt`, `get_benchmark_leaderboard` |
| auth / quota / search | ~4 | `authorize`, `get_user_profile`, `get_accelerator_quota`, `search_content` |

### 2.3 文档源

| 源 | 状态 |
|----|------|
| https://www.kaggle.com/docs/mcp | 官方页（SPA，纯爬虫易失败） |
| 实时 `tools/list` | **最可靠**（本仓库已抓） |
| https://www.kaggle.com/product-announcements/635978 | 产品公告 |
| hackathon-judging skill 内 MCP 表 | 角色权限矩阵 |

### 2.4 Hackathon MCP 权限（来自官方 skill）

| 角色 | overview / tracks | get writeup | list writeups | CSV export |
|------|-------------------|-------------|---------------|------------|
| 匿名 | ✅ | ❌ | ❌ | ❌ |
| 登录用户 | ✅ | ✅ | ❌ | ❌ |
| 提交者 | ✅ | ✅ | ✅ | ❌ |
| host | ✅ | ✅ | ✅ | ✅ |

---

## 3. 官方 Skills（[Kaggle/kaggle-skills](https://github.com/Kaggle/kaggle-skills)）

本仓库镜像：`docs/skills/`

| Skill | 何时用 | 路径 |
|-------|--------|------|
| **write-kaggle-benchmarks** | 写/推/跑/下 Benchmark 任务 | [skills/write-kaggle-benchmarks/](./skills/write-kaggle-benchmarks/) |
| **hackathon-judging** | Host 用 MCP+LLM 评 Hackathon writeup | [skills/hackathon-judging/](./skills/hackathon-judging/) |
| **kaggle-standardized-agent-exam** | Agent 标准化考试 HTTP API | [skills/kaggle-standardized-agent-exam/](./skills/kaggle-standardized-agent-exam/) |
| **template-skill** | 新建 skill 模板 | [skills/template-skill/](./skills/template-skill/) |

### 3.1 write-kaggle-benchmarks（最常用）

工作流：

```text
init/auth → 写 task.py → 本地 python 校验 → push → run → status → log → download → publish
```

核心命令：

```bash
kaggle b init -y
kaggle b auth -y
kaggle b t push my-task -f task.py --wait
kaggle b t run my-task -m gemini-3.5-flash --wait
kaggle b t status my-task
kaggle b t log my-task -m gemini-3.5-flash
kaggle b t download my-task -o ./results
kaggle b t models
```

Skill 要求 agent：**分步确认，不要一条龙自动跑完**（除非用户明确要求）。

关键坑：

- 必须有 `.run(kbench.llm)`，否则 push 成功但无结果
- `MODEL_PROXY_API_KEY` 短时效，失败就 `auth`
- CLI 模型用**短 slug**；HTTP OpenAI 兼容常用 `vendor/model`
- 多模型：`-m a -m b`，不要 `-m a b`

### 3.2 安装到 agent

官方建议对话里说：

> Install the write-kaggle-benchmarks skill: https://github.com/Kaggle/kaggle-skills

本仓库可直接指向本地：

```text
docs/skills/write-kaggle-benchmarks/SKILL.md
```

### 3.3 社区相关（非官方镜像）

| 项目 | 说明 |
|------|------|
| [shepsci/kaggle-skill](https://github.com/shepsci/kaggle-skill) | 一体化竞赛/MCP skill |
| [wenmin-wu/ds-skills](https://github.com/wenmin-wu/ds-skills) | 从高票 Notebook 蒸馏的 DS 技巧 |

---

## 4. CLI Benchmarks 文档（已抓）

全文： [raw/kaggle-cli-benchmarks.md](./raw/kaggle-cli-benchmarks.md)

要点：

- `kaggle b auth` / `init` 写 `MODEL_PROXY_*`
- `LLMS_AVAILABLE` 只是本地子集；全量用 `kaggle b t models`
- push 可挂数据集 `-d owner/slug`；再 push 不带 `-d` 会卸掉数据集
- delete 服务端尚未支持

---

## 5. 与 kaggle-lab 实验的对应

| 实验 | 对应能力 |
|------|----------|
| 001-model-proxy | Model Proxy / Skills 里的 `auth`+chat |
| 002-tool-call | Proxy function calling |
| 003-list-models | `kaggle b t models` |
| **004+** | 应用 MCP 70 工具 + write-kaggle-benchmarks + 可选 hackathon-judging |

---

## 6. 刷新抓取

```bash
# Skills 镜像
git clone --depth 1 https://github.com/Kaggle/kaggle-skills.git /tmp/kaggle-skills
rsync -a --delete /tmp/kaggle-skills/write-kaggle-benchmarks docs/skills/
# …其余 skill 目录同理

# CLI 文档
curl -sL https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/benchmarks.md \
  -o docs/raw/kaggle-cli-benchmarks.md

# MCP tools（需 token）
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
# 用 POST https://www.kaggle.com/mcp method=tools/list → 更新 raw/mcp-tools.json + mcp-tools.md
```

---

## 7. 参考链接速查

- MCP docs: https://www.kaggle.com/docs/mcp  
- Skills: https://github.com/Kaggle/kaggle-skills  
- CLI benchmarks: https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md  
- SDK: https://github.com/Kaggle/kaggle-benchmarks  
- Blog local benchmarks: https://www.kaggle.com/blog/build-kaggle-benchmarks-local-dev  
