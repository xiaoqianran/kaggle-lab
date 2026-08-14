# SAE 是什么？（必读）

> **本仓库策略（用户明确要求）：默认不开考、不注册、不交卷。**  
> Agent / 协作者 **禁止** 执行 `004 start`、`004 submit`、`004 register`（除非用户以后书面改口）。  
> 允许：读文档、改代码、跑 `009 dry-run`（假试卷，不占次数）。

---

## 一句话

**SAE = Standardized Agent Exam = Kaggle 给 AI Agent 出的标准化小考。**

不是竞赛、不是 GPU notebook、不是 Model Proxy 调模型额度。

| 对比 | SAE | Model Proxy (001) | 竞赛 / Notebook |
|------|-----|-------------------|-----------------|
| 目的 | 测 agent 跟不跟指令、答不答对格式 | 调 Gemini/GPT 等大模型 | 训模型 / 交预测 |
| 出题方 | Kaggle 服务器 | 你自己 prompt | 竞赛主办方 |
| 次数 | **每 agent 最多 3 次** | 按日/月 $ 额度 | 看竞赛规则 |
| 时限 | **30 分钟 / 次，16 题** | 无 | 看竞赛 |
| 鉴权 | `~/.kaggle-agent-*`（注册返回） | `KAGGLE_API_TOKEN` + 临时 Proxy key | 账号 API |

官网：https://www.kaggle.com/experimental/sae  

---

## 流程长什么样（仅作说明，不要跑）

```text
register  →  拿到 agentId + apiToken（独立身份）
start     →  领 16 题，开始 30 分钟倒计时   ← 占 1 次配额
作答      →  本地生成 16 个答案
submit    →  一次交齐，Kaggle 判分、公开 profile
```

本仓库对应代码：

| 目录 | 角色 | 本仓库是否建议运行 |
|------|------|-------------------|
| `labs/004-sae/` | 正式客户端（register / start / submit） | **默认禁止** start/submit/register |
| `labs/009-sae-better/` | 增强答题器 | **只建议** `dry-run`（mock 卷） |
| `docs/sae-standardized-agent-exam.md` | 协议细节 | 可读 |
| `docs/skills/kaggle-standardized-agent-exam/` | 官方 skill 镜像 | 可读 |

---

## 禁令清单（给 Agent / 未来的自己）

| 命令 | 状态 |
|------|------|
| `python main.py 004 register ...` | 禁止（除非用户明确要求） |
| `python main.py 004 start --i-accept` | **禁止** |
| `python main.py 004 submit --i-accept` | **禁止** |
| `python main.py 004 answer-proxy` | 禁止对着真卷跑（会浪费真题时间） |
| `python main.py 009 answer --from 004` | 禁止（依赖真卷） |
| `python main.py 009 dry-run` | **允许**（假卷，不占 3 次） |
| `python main.py 009 dry-run --ensemble` | **允许** |
| 阅读 / 改代码 / 写文档 | **允许** |

若用户只说「继续实验 / 跑一下 / 测试 SAE」→ **一律当作 dry-run**，不要 start。

---

## 为什么不随便考

1. **每 agent 只有 3 次**，考砸也占名额  
2. 开考即 **30 分钟计时**，中断也算一次  
3. 和日常练手无关；练答题逻辑用 **mock dry-run** 足够  
4. 用户已明确：**千万别考**

---

## 允许的本地练习

```bash
# 假试卷 5 题，不联网开考、不占 SAE 次数（会扣一点 Model Proxy AI $）
python main.py 009 dry-run
python main.py 009 dry-run --ensemble
python main.py 009 show
```

---

## 相关文档

- 协议：`docs/sae-standardized-agent-exam.md`
- 客户端：`labs/004-sae/README.md`
- 增强答题：`009-sae-better/README.md`
- 总览：根目录 `README.md` 中的「SAE 禁考」小节
