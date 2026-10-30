# 007-mcp-harness

Kaggle 官方 MCP（`https://www.kaggle.com/mcp`）薄客户端。

- 鉴权：`Bearer $KAGGLE_API_TOKEN`（与 Model Proxy 不同）
- **不扣** AI Models $；部分工具只读平台元数据

## 用法

```bash
python run.py tools
python run.py tools --grep notebook
python run.py profile
python run.py competitions --q "llm" --n 5
python run.py datasets --q "titanic" --n 3
python run.py call get_accelerator_quota
python run.py call search_competitions --json '{"request":{"search":"arc","pageSize":3}}'
```

产物：`artifacts/tools.json`、`call-*.json` 等。

## 与 001 的区别

| | 001 Model Proxy | 007 MCP |
|--|-----------------|---------|
| 用途 | 调 Gemini/GPT 等 | 竞赛/数据集/Notebook/配额… |
| 鉴权 | `kaggle b auth` 临时 key | `KAGGLE_API_TOKEN` |
| 费用 | 日 $10 / 月 $100 | 一般免费（平台 API） |

全量工具清单：`docs/mcp-tools.md`。
