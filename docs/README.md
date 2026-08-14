# kaggle-lab / docs

本目录存放从 **Kaggle 官方 MCP / Skills / CLI** 抓取与整理的资料，供 004+ 复杂实验查阅。

## 索引

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 仓库怎么拆：labs / tracks / 目录册 / CLI |
| [mcp-and-skills.md](./mcp-and-skills.md) | **总览**：MCP 配置、Skills 清单、与 Model Proxy 的区别、本仓库怎么用 |
| [mcp-tools.md](./mcp-tools.md) | 官方 MCP **70 工具**实时清单（按类别） |
| [sae-standardized-agent-exam.md](./sae-standardized-agent-exam.md) | **SAE**：如何把 agent 接成标准化考试客户端 |
| [skills/](./skills/) | 官方 [Kaggle/kaggle-skills](https://github.com/Kaggle/kaggle-skills) 镜像副本 |
| [raw/](./raw/) | 原始抓取：CLI benchmarks 文档、MCP tools JSON、skill 原文 |

## 官方源（在线）

| 类型 | URL |
|------|-----|
| MCP 文档 | https://www.kaggle.com/docs/mcp |
| MCP 端点 | `https://www.kaggle.com/mcp` |
| MCP 公告 | https://www.kaggle.com/product-announcements/635978 |
| Skills 仓库 | https://github.com/Kaggle/kaggle-skills |
| CLI Benchmarks | https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md |
| Benchmarks SDK | https://github.com/Kaggle/kaggle-benchmarks |
| Platform docs 入口 | https://www.kaggle.com/docs |

## 抓取说明

- **MCP 工具**：用账号 KGAT 对 `tools/list` 实时拉取 → `raw/mcp-tools.json` / `mcp-tools.md`
- **Skills**：`git clone --depth 1` 官方仓库 → `skills/`
- **CLI 文档**：raw.githubusercontent.com → `raw/kaggle-cli-*.md`
- **docs-mcp-server**：`kaggle.com/docs` 为 SPA，全站 scrape 易失败；GitHub 文档更可靠

刷新 MCP 工具列表：

```bash
# 需有效 KAGGLE_API_TOKEN
python - <<'PY'
# 或复用实验脚本扩展；当前可手动: 见 common + docs 生成逻辑
print("见 docs/mcp-tools.md 生成方式 / 重新跑抓取脚本")
PY
```

更稳妥：在本机已登录状态下让 agent 重跑「抓取 MCP tools + 写 mcp-tools.md」。
