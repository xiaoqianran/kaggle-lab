# 012-camel-roleplay

CAMEL **RolePlaying** 双智能体协作，走 Kaggle Model Proxy。

| 角色 | 设定 |
|------|------|
| Lab Lead (user) | 推进任务、提要求 |
| Kaggle Agent Researcher (assistant) | 给可执行方案 / 简报 |

默认产出：本周 multi-agent 实验计划（省额度策略 + 日程）。

## 用法

```bash
python main.py 012 run
python main.py 012 run --rounds 4
python main.py 012 run --no-task-specify   # 更省额度
python main.py 012 run --task "对比 flash vs nano 在 tool-calling 上的稳定性，给出评测协议"
```

产物：

- `artifacts/last-roleplay.json` — 全对话
- `artifacts/last-brief.md` — 最终简报

## 额度提示

- 每轮 ≈ 2 次 LLM 调用；`with_task_specify` 默认再 +1  
- 默认 3 轮 + specify ≈ **7 次** Proxy 调用 → 用 **flash** 级模型  
- 需要更长协作：`--rounds 5`，同时降 `--max-tokens`
