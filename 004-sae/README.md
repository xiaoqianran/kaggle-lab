# 004-sae — Kaggle Standardized Agent Exam 客户端

> ## 禁止开考（仓库策略）
>
> **用户要求：千万别考。**  
> 默认 **不要** 执行 `register` / `start` / `submit`。  
> 需要了解 SAE 或练答题：看 [`docs/SAE.md`](../docs/SAE.md)，或只跑 **`009 dry-run`**（假卷）。  
> 只有用户以后**书面明确**说「现在正式开考」并带确认时，才可使用下方带 `--i-accept` 的命令。

---

把本机 agent **接成** Kaggle SAE 标准考生（HTTP 客户端），不是自建考场。

- **SAE 是什么**：[`docs/SAE.md`](../docs/SAE.md)（必读）
- 官方说明：https://www.kaggle.com/experimental/sae  
- 详细协议：`docs/sae-standardized-agent-exam.md`  
- 官方 skill：`docs/skills/kaggle-standardized-agent-exam/SKILL.md`

## 前置（仅在用户明确要求真考时）

- 无需 Kaggle 账号即可注册 agent（SAE 独立身份）
- **开考前必须人工确认**（30 分钟计时 + 最多 3 次）
- 凭证：`~/.kaggle-agent-id` / `~/.kaggle-agent-api-key`（chmod 600）
- 与 `~/.kaggle/access_token`（平台 API）**不是同一把钥匙**

## 用法（默认不要跑 start/submit）

```bash
# ——— 禁止默认执行 ———
# python main.py 004 register --name "..." --i-accept
# python main.py 004 start --i-accept      # 领 16 题；开始计时！占 1 次
# python main.py 004 submit --i-accept     # 交卷

# 只读 / 调试（无真卷时无用）
# python main.py 004 whoami
# python main.py 004 show-paper
# python main.py 004 status
# python main.py 004 history
```

**推荐替代（不占考试次数）：**

```bash
python main.py 009 dry-run
python main.py 009 dry-run --ensemble
```

产物（若曾跑过）在 `004-sae/artifacts/`（试卷、答案草稿；**不含** api key）。

## 安全

- API key **只**发给 `https://www.kaggle.com/api/v1/*`
- 勿把 key 打进日志 / git
- 404 → 立刻停；412 → 已达 3 次上限
- Agent 不得在用户未确认时调用 start/submit
