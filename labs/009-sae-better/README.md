# 009-sae-better

SAE 答题器增强版（在 004 之上）。

> ## 禁止开考（仓库策略）
>
> **用户要求：千万别考。**  
> 本目录 **只建议** `dry-run` / `show`。  
> **不要** `answer --from 004`、**不要** `export-to-004` 后交卷、**不要** 联动 `004 start/submit`。  
> SAE 说明：[`docs/SAE.md`](../../docs/SAE.md)

| 能力 | 说明 | 默认 |
|------|------|------|
| **清洗** | 去 fence / 前缀；JSON 抽取；纯数字收紧 | 可用 |
| **ensemble** | 多模型多数票 | dry-run 可用 |
| **dry-run** | `mock_paper.json` 假卷，**不占 3 次** | **推荐** |
| **export-to-004** | 写回 004 供交卷 | **禁用（除非真考）** |

## 用法（允许）

```bash
python main.py 009 dry-run
python main.py 009 dry-run --ensemble
python main.py 009 show
```

## 用法（禁止默认执行 — 真考专用）

```bash
# 以下会依赖真卷 / 占次数，本仓库策略下不要跑：
# python main.py 004 start --i-accept
# python main.py 009 answer --from 004 --ensemble
# python main.py 009 export-to-004
# python main.py 004 submit --i-accept
```

## 安全

- 默认不 `start` / `submit`
- SAE API key 只应发给 `www.kaggle.com`
- 404 → 立刻停；412 → 已达 3 次上限

## 产物

- `artifacts/answers.json` — 答案（dry-run 也是 mock）
- `artifacts/answers-detail.json` — 各模型候选
- `mock_paper.json` — 本地 5 题假试卷
