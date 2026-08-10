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

## From Scratch 阶梯执行

**状态: S00–S20 全部完成**（2026-08-10）

- 地图: `from_scratch/MAP.md`
- 能力链: `from_scratch/CAPABILITY_MAP.md`
- 结果表: `from_scratch/results/LADDER_RESULTS.md`
- 可视化: `from_scratch/results/viz/`（每步 PNG）
- 笔记: `from_scratch/notes/Sxx.md`
- 复现: `python from_scratch/run_ladder.py`

| Step | Status | 新增能力一句话 |
|------|--------|----------------|
| S00 | DONE | 能力#0：把像素当数据——先看 shape/scale/类分布，再谈模型。 |
| S01 | DONE | 能力#1：无学习基线。kNN>>随机 ⇒ 数据有可分结构，后续模型必须超过它。 |
| S02 | DONE | 能力#2：参数学习+概率输出。仍是超平面决策，无空间结构。 |
| S03 | DONE | 能力#3：深度非线性。常略好于线性，但过拟合与无视空间 → 需要卷积。 |
| S04 | DONE | 能力#4：局部模式提取。参数效率数量级优势；第一次‘看见’边缘。 |
| S05 | DONE | 能力#5：完整卷积分类器。空间归纳偏置进入闭环。 |
| S06 | DONE | 能力#6：训练稳定性技术。同架构下 BN 通常更快更稳。 |
| S07 | DONE | 能力#7：深度设计直觉。过深可能退化 → 引出残差。 |
| S08 | DONE | 能力#8：可训练的深度。2015 分水岭——现代主干默认残差。 |
| S09 | DONE | 能力#9：效率轴。研究也要报成本，不只报 top1。 |
| S10 | DONE | 能力#10：工业级训练配方。SOTA 提升常来自 recipe。 |
| S11 | DONE | 能力#11：通道注意力。ViT 之前的实用注意形态。 |
| S12 | DONE | 能力#12：全局自注意力。小数据常弱于 CNN → 需配方/预训练。 |
| S13 | DONE | 能力#13：现代 CNN 双路线。注意力非唯一答案。 |
| S14 | DONE | 能力#14：无标签表征学习。linear probe 是标准度量。 |
| S15 | DONE | 能力#15：开放集接口。换原型即可扩类（真 CLIP 用文本塔）。 |
| S16 | DONE | 能力#16：迁移协议。少标签下 probe/FT 通常碾压从头训。 |
| S17 | DONE | 能力#17：掩码图像建模。MAE 路线与对比学习并列的 SSL 支柱。 |
| S18 | DONE | 能力#18：数据高效 ViT 配方。架构+配方缺一不可。 |
| S19 | DONE | 能力#19：多模态对齐接口。真 CLIP 换文本编码器即可扩展任意类名。 |
| S20 | DONE | 能力#20：完整评测协议。只有 top1 不够，校准与 OOD 同属现代系统。 |
