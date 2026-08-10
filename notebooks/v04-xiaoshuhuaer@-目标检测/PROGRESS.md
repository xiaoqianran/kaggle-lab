# PROGRESS · 目标检测 From-Scratch 全链路

最后更新：2026-08-10  
状态：**FS00–FS15 全部跑通**（代码 + JSON + 可视化 PNG + 对比）

权威地图：[FROM_SCRATCH.md](./FROM_SCRATCH.md)

## 总览

| Step | 主题 | 状态 | 关键可观察结果 |
|------|------|------|----------------|
| FS00 | 协议 IoU/NMS/mAP | ✅ | loc_noise：AP50 高但 AP 崩 |
| FS01 | 穷举滑窗 | ✅ | ~169 窗/图；AP50≈0.34 |
| FS02 | 图像金字塔 | ✅ | 窗数×3+；朴素融合可掉点 |
| FS03 | 梯度特征 | ✅ | 暗光下 color 崩、grad 更稳 |
| FS04 | 候选区域 | ✅ | 窗 169 vs 提案 22；random recall≈0.18 |
| FS05 | R-CNN 裁剪 CNN | ✅ | AP50 **0.85 vs 滑窗 0.37** |
| FS06 | 共享 backbone | ✅ | **500:1** forwards；墙钟 **~10×** |
| FS07 | Faster/RPN 味 | ✅ | AP50 0→**0.65** |
| FS08 | 密集单阶段 | ✅ | AP50→**0.81** + conf 消融 |
| FS09 | Focal Loss | ✅ | 正样本~1%；γ 超参课 |
| FS10 | 多尺度头 | ✅ | 无融合可负结果（真 FPN 细节） |
| FS11 | 锚 vs 无锚 | ✅ | 锚 pos_frac 随 thr 变 |
| FS12 | DETR matching | ✅ | loss↓；Hungarian 可演示 |
| FS13 | 现代配方 | ✅ | best=`flip_longer` AP50**0.79** |
| FS14 | 域偏移 | ✅ | ID 0.79→OOD 0.30→FT 0.92 |
| FS15 | 研究假设 | ✅ | H1–H3 **全部 accepted** |

## 能力阶梯（你应能口述）

1. **FS00–01**：会评测；知道穷举搜索的病（慢、尺度、重复框）  
2. **FS03–05**：特征与提案如何把搜索变成可算的「候选+分类」  
3. **FS06–07**：共享计算 + 学提案（现代两阶段 DNA）  
4. **FS08–11**：单阶段、损失、多尺度、无锚  
5. **FS12–13**：集合预测与训练系统  
6. **FS14–15**：真实域与可证伪研究  

## 一键复验

```bash
source /workspace/.venv-det/bin/activate
cd notebooks/v04-xiaoshuhuaer@-目标检测
python scripts/run_fs_all.py
# 或分步：python scripts/fs00_protocol_viz.py … fs15_hypothesis_viz.py
```

产物：`results/fsXX_*/results.json` + `*.png`  
总表：`results/fs_chain_summary.json`

## 路线修正

| 修正 | 原因 |
|------|------|
| 合成色块数据 | CPU 可训可看；保证闭环 |
| FS02/10 保留负结果 | 朴素多尺度/无融合多头会掉点——历史动机课 |
| FS06 用 500 提案模拟 | 少提案时墙钟对比失真 |
| FS09 双组 Focal 超参 | 默认 γ=2 在 easy synth 可输 CE |

## 研究轨（并行）

P0–P7 仍可用 `./scripts/run_all.sh`；与 FS 共享 `scripts/` 核心库。
