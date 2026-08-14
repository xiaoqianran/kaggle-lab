# 010-quota-dashboard

实验室运维看板：

1. **GPU / TPU** 周额度 — `kaggle quota`
2. **AI Models $** 本地账本 — `logs/usage.jsonl`（由 `kaggle_lab.proxy` 自动追加）

> 官方日 $10 / 月 $100 是**账号级**；本地 log 只统计本机跑过的 Proxy 调用。

## 用法

```bash
python main.py 010 show
python main.py 010 gpu
python main.py 010 usage
python main.py 010 usage --tail 20
python main.py 010 usage --json
```

## 相关

- Proxy 调用：`001` / `002` / `008`（会写 usage log）
- 额度网页：Kaggle 账号 settings / AI Models
