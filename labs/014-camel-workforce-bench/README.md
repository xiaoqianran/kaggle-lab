# 014-camel-workforce-bench

**有意义的 multi-agent 流水线**：把 Kaggle Model Proxy 额度花在  
「侦察 → 辩论 → 写出可 push 的 Benchmark task →（可选）005 发布」。

## 为什么不是裸调 Workforce？

实测 CAMEL 原生 `Workforce` + `use_structured_output_handler` 在 Model Proxy  
（尤其 Gemini flash + reasoning tokens）上 **JSON 结构化分配不稳定**、易耗尽 `max_tokens`。

因此本实验采用 **Workforce 风格、可控流水线**：

| 阶段 | 角色 | 实现 | 产出 |
|------|------|------|------|
| 1 Scout | 研究员 | CAMEL `ChatAgent` + MCP/quota 工具 | `artifacts/scout-brief.md` |
| 2 Debate | Advocate vs Critic | CAMEL `RolePlaying` | `artifacts/debate.md` |
| 3 Author | 出题人 | 只生成 **结构化 cases JSON** | `artifacts/spec.json` |
| 4 Assemble | 工程 | 模板渲染 `task.py` + AST 校验 | `artifacts/workspace/task.py` |
| 5 Validate | 裁判 | 本地 `kbench` 跑 task（可选） | 终端 pass/fail |
| 6 Publish | 发布 | `kaggle b t push/run`（需 `--i-accept`） | Kaggle Benchmarks |

另提供 `workforce-probe`：直接打原生 CAMEL Workforce（实验性，允许失败）。

## 安全设计

- **Author 不直接写 Python**：只输出 JSON cases → 模板拼装，避免非法代码 / 漏 `.run()`  
- **Publish 默认关闭**：必须 `--i-accept`  
- **SAE 无关**：不碰 004 开考  

## 用法

```bash
# 推荐：全流程到本地 validate（会烧 AI $，约十余次 LLM 调用）
python main.py 014 run
python main.py 014 run --theme "JSON instruction following" --query "llm" --debate-rounds 2

# 分步
python main.py 014 scout
python main.py 014 debate
python main.py 014 author
python main.py 014 assemble
python main.py 014 validate-local

# 发布到 Kaggle Benchmarks（确认后）
python main.py 014 publish --push --i-accept
python main.py 014 publish --push --run-remote --run-model gemini-3.5-flash --i-accept

# 原生 Workforce 探针
python main.py 014 workforce-probe

python main.py 014 show
```

## 额度粗估（flash 级）

| 阶段 | 约调用次数 |
|------|------------|
| scout（含 tools） | 2–5 |
| debate ×2 轮 | ~4 |
| author（+可能 repair） | 1–2 |
| local validate | ≈ cases 数（每次 llm.prompt） |
| **合计无 push** | **~10–15** 次 Proxy 调用 |

省额度：`--debate-rounds 1`、`--skip-local-validate`、用 `gemini-3.5-flash-lite`。

## 与 005/012/013 关系

- **013**：侦察简报  
- **012**：双角色协作  
- **014**：把两者串成 **可交付 Benchmark 产物**，并接到 **005** 的 push/run  
- 生成的 task 与手写 `labs/005-benchmark-task/task.py` 同构（`@kbench.task` + `.run(kbench.llm)`）
