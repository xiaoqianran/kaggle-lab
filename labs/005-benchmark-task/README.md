# 005-benchmark-task

Kaggle **Benchmarks** 最小闭环：写 task → 本地校验 → push → run → download。

扣 **AI Models 额度**（日 $10 / 月 $100），不是 GPU 小时。

## 用法

```bash
# 分步（推荐）
python run.py auth
python run.py validate              # 本地 python task.py
python run.py push                  # 上传 kaggle-lab-smoke-math
python run.py run -m gemini-3.5-flash
python run.py status
python run.py download

# 或一条龙（需显式确认，会烧额度）
python run.py pipeline --i-accept -m gemini-3.5-flash

python run.py list
python run.py models
```

## 文件

| 路径 | 作用 |
|------|------|
| `task.py` | `@kbench.task` + 必须 `.run(kbench.llm)` |
| `run.py` | CLI 封装 `kaggle b t …` |
| `artifacts/results/` | `download` 输出 |

## 前置

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
source .venv/bin/activate
# 需 kaggle-benchmarks（已装到 .venv，或: uv pip install -e vendor/kaggle-benchmarks）
```

## 注意

- CLI 模型用**短 slug**（`gemini-3.5-flash`），不要 `google/…`
- 多模型：`-m a -m b`，不要 `-m a b`
- 临时 key ~1h 过期 → 再 `auth`
- 官方 skill：`docs/skills/write-kaggle-benchmarks/SKILL.md`
