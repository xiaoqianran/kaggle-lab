# 002-tool-call

演示 Kaggle Model Proxy 的 **OpenAI function / tool calling**。

模型只负责返回 `tool_calls`；本地执行工具后再回填 `role=tool` 消息。

## 用法

```bash
python main.py 002 run
python main.py 002 run -m google/gemini-3-flash-preview
python main.py 002 run --prompt "上海天气？" --refresh
```

## 说明

- 示例工具：`get_weather(city, unit)`
- 已在 `gemini-3-flash-preview` / `gpt-5.4-nano` 上验证过
- 旗舰模型可能 503 unavailable，换便宜模型即可
