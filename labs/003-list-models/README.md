# 003-list-models

列出 / 导出 Kaggle Benchmarks 可用的 AI Models（扣额度无关，仅元数据）。

## 用法

```bash
python run.py list
python run.py dump   # 写出 kaggle_ai_models.{txt,csv,json}
```

`dump` 只写到本目录（模型表的唯一一份）。
