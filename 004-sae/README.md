# 004-sae — Kaggle Standardized Agent Exam 客户端

把本机 agent **接成** Kaggle SAE 标准考生（HTTP 客户端），不是自建考场。

- 官方说明：https://www.kaggle.com/experimental/sae  
- 详细协议：`docs/sae-standardized-agent-exam.md`  
- 官方 skill：`docs/skills/kaggle-standardized-agent-exam/SKILL.md`

## 前置

- 无需 Kaggle 账号即可注册 agent（SAE 独立身份）
- **开考前必须人工确认**（30 分钟计时 + 最多 3 次）
- 凭证：`~/.kaggle-agent-id` / `~/.kaggle-agent-api-key`（chmod 600）

## 用法

```bash
# 从仓库根
python main.py 004 register --name "YourUniqueName-42" \
  --model grok-build --agent-type Grok \
  --description "kaggle-lab SAE harness"

python main.py 004 whoami
python main.py 004 start              # 领 16 题；开始计时！
python main.py 004 show-paper         # 打印当前试卷
python main.py 004 answer-stub        # 占位答案（调试用，勿当真）
python main.py 004 answer-proxy       # 用 Model Proxy 答题（扣 AI $）
python main.py 004 submit             # 交齐 16 题
python main.py 004 status
python main.py 004 history
```

产物在 `004-sae/artifacts/`（试卷、答案草稿；**不含** api key）。

## 安全

- API key **只**发给 `https://www.kaggle.com/api/v1/*`
- 勿把 key 打进日志 / git
- 404 → 立刻停；412 → 已达 3 次上限
