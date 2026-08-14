# 011-camel-proxy

把 **[CAMEL](https://github.com/camel-ai/camel)** 接到 **Kaggle Model Proxy**（日 $10 / 月 $100 AI Models）。

这是后续 multi-agent 实验的底座。

## 依赖

```bash
source .venv/bin/activate
uv pip install 'camel-ai==0.2.90'
# camel 0.2.90 需要 FastMCP；若 import 失败：
uv pip install 'mcp==1.9.4'
```

共享封装：`kaggle_lab.camel_proxy`（`common.camel_proxy` 仍可用）  
→ `ModelPlatformType.OPENAI_COMPATIBLE_MODEL` + `{MODEL_PROXY_URL}/openapi`

## 用法

```bash
python main.py 011 auth
python main.py 011 smoke
python main.py 011 chat "用三句话说明 multi-agent"
python main.py 011 tools "UTC 时间 + 算 19*3"
python main.py 011 chat -m google/gemini-3.5-flash --lang Chinese "..."
```

## 与 001/008 的区别

| | 001 | 008 | **011** |
|--|-----|-----|---------|
| 客户端 | 裸 OpenAI SDK | 手写 ReAct | **CAMEL ChatAgent** |
| 工具 | 无 | 手写 loop | CAMEL FunctionTool |
| 后续扩展 | — | — | RolePlaying / Workforce (012+) |
