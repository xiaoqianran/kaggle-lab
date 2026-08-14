# 006-judge-llm

多模型在同一题库上作答，**规则判分 + Judge LLM** 汇总排行榜。

- 规则：`exact` / `contains` / `json_equal`（零额外费用）
- Judge：`kind=judge` 的题由独立模型打 0/1 分（扣 AI $）

## 用法

```bash
python main.py 006 run
python main.py 006 run -m google/gemini-3-flash-preview -m openai/gpt-5.4-nano-2026-03-17
python main.py 006 run --judge-model google/gemini-3.5-flash --refresh
python main.py 006 show
```

## 文件

| 路径 | 作用 |
|------|------|
| `bank.json` | 题库（可改） |
| `artifacts/last-run.json` | 全量明细 |
| `artifacts/last-run.csv` | 简表 |

## 与 005 的关系

- **005**：官方 Benchmarks 服务端跑 task
- **006**：本地快速对比模型 + 判分（适合迭代题库 / 选模型）

两者可互补：005 定正式 task，006 做本地预筛。
