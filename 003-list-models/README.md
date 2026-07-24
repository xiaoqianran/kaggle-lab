# 003-list-models

列出 / 导出 Kaggle Benchmarks 可用的 AI Models（扣额度无关，仅元数据）。

## 用法

```bash
python main.py 003 list
python main.py 003 dump   # 写出 kaggle_ai_models.{txt,csv,json}
```

`dump` 会写到本目录，并同步一份到仓库根，方便用表格软件筛选。
