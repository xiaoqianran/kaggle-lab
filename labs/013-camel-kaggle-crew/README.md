# 013-camel-kaggle-crew

CAMEL **带工具的智能体** + 真实 Kaggle 信号（MCP 搜竞赛/数据集、`kaggle quota`）。

目标：把 **Model Proxy 额度**用在「有依据的决策」上，而不是空聊。

## 工具

| Tool | 来源 |
|------|------|
| `search_kaggle_competitions` | MCP `search_competitions` |
| `search_kaggle_datasets` | MCP `search_datasets` |
| `get_gpu_quota` | `kaggle quota` |
| `list_proxy_model_hints` | 本地便宜模型清单 |

## 用法

```bash
# 默认有意义任务：竞赛侦察 + 配额 + 行动简报
python main.py 013 scout
python main.py 013 scout --q "agent llm" --n 5

# 自由问
python main.py 013 ask "我只想用 Model Proxy 做 multi-agent，不要 GPU，给 3 天计划"
```

产物：`artifacts/last-ask.md`

## 额度

- 通常 1 次用户问题 + 若干 tool 往返（约 2–5 次 LLM 调用）
- 工具本身（MCP/quota）**不扣** AI $
- 推荐 flash：`google/gemini-3-flash-preview`
