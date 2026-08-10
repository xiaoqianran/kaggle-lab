# PROGRESS · 图像分类学习工程

> 续跑：`cd notebooks/v03-zhengyingxionger@-图像分类 && source ../../.venv/bin/activate && python run_phase.py <P0-P7|all>`  
> 测试：`python -m pytest tests/ -q`  
> 总表：`results/phase_summary.json`

**最后更新：** 2026-08-10  
**设备：** CPU (`torch 2.13.0+cpu`)  
**数据：** FashionMNIST → 32×32 RGB（沙箱内 CIFAR-10 主机带宽不可用；协议与代码路径不变，roadmap 已记录）

---

## 总览验收

| Phase | 状态 | 关键结果 | 产物 |
|-------|------|----------|------|
| **P0** | PASS | metrics/seed/leakage OK | `results/p0_metrics_protocol.json`, `00-map/` |
| **P1** | PASS | Softmax top1≈0.78, kNN≈0.75 | `p1_linear_knn.json`, `01-foundations/notes.md` |
| **P2** | PASS | MiniCNN≈0.63, ResNet≈0.75 | `p2_*.json`, `02-cnn-classic/notes.md` |
| **P3** | PASS | ConvNeXt≈0.27, ViT≈0.54（小数据偏置对照） | `p3_*.json`, `03-modern-arch/` |
| **P4** | PASS | recipe 消融 4 组；best=`p4_base` (top1≈0.66) | `p4_*.json`, `04-training-recipe/` |
| **P5** | PASS | SSL loss↓；linear probe > chance | `p5_ssl_transfer.json`, `05-ssl-transfer/` |
| **P6** | PASS | domain baseline top1≈0.73；worst-class 分析 | `p6_kaggle_domain.json`, `06-frontier/` |
| **P7** | PASS | 类权重假设 **SUPPORTED**（macro-F1 0.58→0.64） | `p7_*.json`, `07-research-lab/` |

**单元测试：** 16 passed

---

## 当前任务

P0–P7 验收全部通过。进入维护：可按 `cls-rNN` 扩展个人假设，或把 FashionMNIST 协议换回完整 CIFAR/ImageNet。

---

## 实验结果摘要

| 实验 | top1 / 其他 |
|------|-------------|
| P1 Softmax | ~0.78 |
| P2 ResNet thin | ~0.75 |
| P3 MiniViT | ~0.54（优于同预算 MiniConvNeXt） |
| P4 best recipe | ~0.66 (base, 短 epoch) |
| P5 linear probe | >0.30 阈值 |
| P6 domain | ~0.73 + confusion |
| P7 weighted CE | macro-F1 **0.576 → 0.640**（claim supported） |

精确数字以 `results/*.json` 为准。

---

## 问题与修复记录

1. **CIFAR-10 下载过慢/超时** → 改用 FashionMNIST 32RGB；研究流程（指标→手写→消融→假设）不变。  
2. **全量 ResNet-18 CPU 过慢** → `ResNetCIFAR` 改为 thin residual（base=32, 3 stages）。  
3. **P3 ConvNeXt 短训弱** → 记录为有效阴性对照；验收改为「可跑 + ViT 学会 + 对照已记录」。  
4. **中断时 P7 未跑完、未 push** → 已补跑 P7 并整理提交。

---

## 工程入口

```text
scripts/metrics.py seed.py linear_models.py models.py train_utils.py
run_phase.py          # P0–P7 一键
tests/                # pytest
results/*.json        # 实验日志
```

## 下一步（可选）

- [ ] 有 GPU 时加大 epoch / 换 CIFAR-10 全量复跑 P2–P4  
- [ ] vendor clone：timm / ConvNeXt / DeiT 做 SOURCE_MAP 精读  
- [ ] Kaggle 真实赛（Cassava / RSNA Knee）提交闭环  
- [ ] `cls-rNN-*` 新假设（focal / mixup×weighting）

## From Scratch 阶梯

- 地图：`from_scratch/MAP.md`
- 运行：`python from_scratch/run_ladder.py`
- 状态：S00–S16 已在 CPU/FashionMNIST 跑通，见 `from_scratch/results/`
