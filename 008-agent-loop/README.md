# 008-agent-loop

多轮 **ReAct / tool-calling** agent：Model Proxy + 本地 tools，可选 Kaggle MCP。

在 002 单轮 weather 之上加：

- 多轮循环（默认最多 5 round）
- `calc` / `now_utc` / `get_weather`
- `--with-mcp` 时挂上 `search_competitions`

## 用法

```bash
python main.py 008 run "东京天气？再算 17*3"
python main.py 008 run --refresh "现在 UTC 几点？"
python main.py 008 run --with-mcp "搜一个进行中的 LLM 竞赛并简述"
python main.py 008 run -m openai/gpt-5.4-nano-2026-03-17 "2+2 和上海天气"
```

产物：`artifacts/last-run.json`（含 tool transcript）。

## 费用

- 默认路径：扣 **AI Models $**（每轮 chat）
- MCP 搜索本身不扣 AI $，但模型决策/总结仍扣
