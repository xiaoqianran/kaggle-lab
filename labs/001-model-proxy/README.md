# 001-model-proxy

Kaggle **Model Proxy** 最小冒烟：拿临时 key + OpenAI 兼容 `chat/completions`。

扣账号 **Daily AI Models $10 / Monthly $100** 额度（不是 GPU 小时）。

## 用法

```bash
# 从仓库根目录
python main.py 001 auth
python main.py 001 chat "用一句话介绍 Kaggle"
python main.py 001 chat -m openai/gpt-5.4-nano-2026-03-17 "1+1=?"
python main.py 001 chat --refresh -m google/gemini-3-flash-preview "写一句问候"

# 或进入目录
cd 001-model-proxy && python run.py chat "hello"
```

## 前置

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
source .venv/bin/activate   # 仓库根 .venv
```

## 说明

- 凭证写到仓库根 `.env.model-proxy`（勿提交 git）
- 临时 key 约 **1 小时**过期
- 模型名 HTTP 常用 `vendor/model`，见 `003-list-models`
