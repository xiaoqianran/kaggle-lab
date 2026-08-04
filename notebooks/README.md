# Mini-Instruct T4 学习路线（001–007）

按顺序读 notebook 顶部的「学习向」说明，再跑代码。

| 课 | 主题 | 学完应能回答 |
|---|---|---|
| 001 | GPU 冒烟 | 如何确认 T4 可用？训练三步？ |
| 002 | 单卡 LoRA | LoRA 训什么？如何锁单卡？ |
| 003 | 双卡 SFT 试验 | DataParallel 是什么？ |
| 004 | 数据与评测尺子 | 如何防泄漏？加权主分？ |
| 005 | 正式 SFT + 对比 | 为何同一 eval_suite？ |
| 006 | DPO 对齐 | chosen/rejected 与三方对比 |
| 007 | Merge 发布 | adapter vs merged？模型卡？ |

注释约定：
- `# 【步骤】...`：本 cell 在整条流水线中的位置
- 行尾中文：该行关键语法/训练细节


## 本目录

存放 Mini-Instruct-T4 系列 **学习向** notebook（001–007），含简体中文关键步骤注释。

| 文件 | Kaggle（在线） |
|---|---|
| `grok-001-t4-smoke-cnn.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-001-t4-smoke-cnn |
| `grok-002-single-t4-qwen-lora.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-002-single-t4-qwen-lora |
| `grok-003-dual-t4-instruct-lora.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-003-dual-t4-instruct-lora |
| `grok-004-data-eval-suite.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-004-data-eval-suite |
| `grok-005-sft-dual-t4-usable.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-005-sft-dual-t4-usable |
| `grok-006-align-dpo-orpo.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-006-align-dpo-orpo |
| `grok-007-release-merge-card.ipynb` | https://www.kaggle.com/code/seachenbgdy/grok-007-release-merge-card |

在 Kaggle 上打开对应 notebook 可选 T4 GPU 直接跑；本地需自备 CUDA + 依赖（torch / transformers / peft 等）。


## 深入层（008+）

对应「T4×2 社区深入实践」学习路径：

| 文件 | 深入层 | 学什么 |
|---|---|---|
| `grok-008-qlora-scale-bench.ipynb` | B 效率工程 | 4bit NF4 QLoRA、peak VRAM、步耗时、3B/7B 边界 |
| `grok-009-gsm8k-eval-baseline.ipynb` | D 硬评测 | GSM8K 冻结子集 + base EM |
| `grok-010-gsm8k-qlora-sft.ipynb` | 闭环 | 同尺 QLoRA SFT：base vs sft EM |

后续可写：`011` 单卡 vs 双卡吞吐；`012` GRPO/可验证 reward。
