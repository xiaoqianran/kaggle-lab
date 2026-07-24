# Kaggle Standardized Agent Exam（SAE）说明

> 官方页：https://www.kaggle.com/experimental/sae  
> Skill：`docs/skills/kaggle-standardized-agent-exam/SKILL.md`  
> Base API：`https://www.kaggle.com/api/v1`

## 你要“serve”的是什么？

**不是**你自己搭考试服务器。  
**是**你的程序作为 **Agent 客户端**，按标准 HTTP 协议去考 Kaggle 托管的 SAE：

```
┌─────────────────────┐         HTTPS          ┌──────────────────────────┐
│  你的 Agent / harness │ ───────────────────► │  Kaggle SAE API          │
│  (本仓库 004-sae)     │  register / start /   │  www.kaggle.com/api/v1/* │
│                       │  answer / score       │  出题 · 判分 · 排行榜     │
└─────────────────────┘                       └──────────────────────────┘
```

| 角色 | 谁做 |
|------|------|
| **Exam Server** | Kaggle（出 16 题、30 分钟、最多 3 次、打分、证书、公开 profile） |
| **Agent（你）** | 注册身份 → 领卷 → 作答 → 交卷 → 看分 |

因此「serve 成标准化 agent」= **实现并运行一个遵守 SAE 协议的客户端 agent**，而不是 host 考试。

---

## 协议一览

| 步骤 | 方法 | 路径 | 鉴权 |
|------|------|------|------|
| 注册 agent | `POST` | `/agentExamAgent` | 无（公开） |
| 开考 | `POST` | `/agentExamSubmission` | Bearer 注册返回的 `apiToken` |
| 交卷 | `POST` | `/agentExamSubmission/{submissionId}` | Bearer |
| 查单次成绩 | `GET` | `/agentExamSubmission/{submissionId}` | Bearer |
| 查 agent 历史 | `GET` | `/agentExamAgent/{agentId}` | 公开 |
| 删除 agent | `DELETE` | `/agentExamAgent/{agentId}` | Bearer（需人类明确授权） |

**本地凭证（与 KGAT 账号 token 不同）：**

| 文件 | 内容 |
|------|------|
| `~/.kaggle-agent-id` | `agentId` |
| `~/.kaggle-agent-api-key` | 注册时一次性返回的 `apiToken`（`KGAT_…`） |

⚠️ **不要**把 SAE 的 key 发到非 `www.kaggle.com` 域名。  
⚠️ **不要**与 `~/.kaggle/access_token`（平台 API / MCP）混用——用途不同。

---

## 约束

| 规则 | 值 |
|------|-----|
| 题量 | 16 |
| 时限 | 30 分钟 / 次 |
| 次数 | 最多 **3** 次 / agent（超限 412） |
| 格式 | 自由文本 / 单字母 / **严格 JSON**（不包 markdown fence） |
| 安全题 | 需 **拒绝** 有害请求（如有害字段填 `null`） |
| 404 | **立刻停止**（功能未开放） |
| 401/403 | 重试一次 → 仍失败则删本地凭证重注册 |

---

## 最小 curl 流程

```bash
# 1) 注册（先征得用户同意；name 要唯一）
curl -s -X POST https://www.kaggle.com/api/v1/agentExamAgent \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "YourUniqueAgent-42",
    "model": "grok-build",
    "version": "1.0",
    "description": "kaggle-lab SAE client",
    "agentType": "Grok"
  }'
# → 保存 agentId / apiToken

# 2) 开考（开始 30 分钟倒计时）
curl -s -X POST https://www.kaggle.com/api/v1/agentExamSubmission \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SAE_KEY" \
  -d '{}'
# → submissionId + questions[16]

# 3) 一次交齐 16 题
curl -s -X POST "https://www.kaggle.com/api/v1/agentExamSubmission/$SUB_ID" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SAE_KEY" \
  -d '{"answers":{"1":"...","2":"...", ... "16":"..."}}'

# 4) 公开主页
# https://www.kaggle.com/experimental/sae/{agentId}
```

---

## 如何把「你的 agent」接成标准化考生

### 方案 A：人 + Agent 手动走 skill（官方设计）

1. 把 `docs/skills/kaggle-standardized-agent-exam/SKILL.md` 交给 agent  
2. Agent 按 skill 分步：确认 → 注册 → 开考 → 作答 → 交卷  
3. 适合 Claude Code / Grok / OpenClaw 等「读 skill 再行动」

### 方案 B：本仓库 `004-sae` 脚本客户端（推荐自动化）

```bash
python main.py 004 register --name "ZephyrMind-42" --model grok-build --agent-type Grok
python main.py 004 start          # 领卷，写到 artifacts/
python main.py 004 answer-auto    # 用 Model Proxy / 本地模型答（可换）
python main.py 004 submit         # 交卷
python main.py 004 status
python main.py 004 history
```

核心循环：

```
register → start_exam → for each question: LLM/rules 生成答案
        → submit_all_16 → print score / certificate / profile URL
```

### 方案 C：自建 HTTP harness（对外 “serve” 你的 agent）

若你希望 **别人/CI 通过 HTTP 触发考试**，你 serve 的是 **harness**，不是 Kaggle 考场：

```
POST /sae/run  →  你的服务内部调用 Kaggle SAE API
               →  用你的 LLM 答题
               →  返回 score / certificateId
```

伪结构：

```python
# 你的服务
@app.post("/sae/run")
def run_exam():
    ensure_registered()
    paper = kaggle.start_exam()          # POST agentExamSubmission
    answers = {}
    for q in paper["questions"]:
        answers[q["id"]] = your_llm.solve(q["text"])  # 严格按题面格式
    result = kaggle.submit(paper["submissionId"], answers)
    return result  # score, percentage, passed, certificateId
```

要点：

- 考试 API **只打** `https://www.kaggle.com/api/v1/*`
- LLM 可走 Kaggle Model Proxy（001）或其它 key
- 答题逻辑必须 **服从题面格式**（格式错 = 判错）
- 安全题要拒绝，不要「尽力帮忙」

---

## 答题质量要点

1. **格式优先**：只回字母 / 只回数字 / 纯 JSON 时不要加解释  
2. **JSON**：不要用 \`\`\`json 包裹；拒答字段用 `null`  
3. **一次交齐 16 题**；漏题算错  
4. **30 分钟内交卷**；超时整次作废  
5. **最多 3 次**；谨慎开考  

---

## 与 MCP / Model Proxy 关系

| 系统 | 关系 |
|------|------|
| SAE | **独立** experimental API，无需 Kaggle 登录账号即可注册 agent |
| MCP | 平台资源（竞赛/数据集）；**不用于** SAE 考试流程 |
| Model Proxy | 可选：用 AI Models 额度当答题 LLM |

---

## 本仓库状态（探测）

- 主页 `https://www.kaggle.com/experimental/sae` → **200**  
- `POST /agentExamAgent` 空 body → **400 Name is required**（说明接口 **在线**，不是 404）  
- 功能可用时按 skill 操作；若任何调用 **404** → 立即停止并告知用户  

---

## 下一步

见实验目录 `004-sae/`：`register` / `start` / `submit` / `status` / `history`。  
**开考会占用 3 次配额之一并开始计时**——运行 `start` 前请人工确认。
