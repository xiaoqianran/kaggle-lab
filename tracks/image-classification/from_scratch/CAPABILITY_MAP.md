# 能力形成地图（实验验收）

从零到前沿，每步**新增一种可感知能力**：

- **S00 pixels as tensors**: 能力#0：把像素当数据——先看 shape/scale/类分布，再谈模型。
- **S01 kNN baseline (no learning)**: 能力#1：无学习基线。kNN>>随机 ⇒ 数据有可分结构，后续模型必须超过它。
- **S02 linear softmax from scratch**: 能力#2：参数学习+概率输出。仍是超平面决策，无空间结构。
- **S03 MLP on flat pixels**: 能力#3：深度非线性。常略好于线性，但过拟合与无视空间 → 需要卷积。
- **S04 hand-written convolution + feature maps**: 能力#4：局部模式提取。参数效率数量级优势；第一次‘看见’边缘。
- **S05 shallow CNN end-to-end**: 能力#5：完整卷积分类器。空间归纳偏置进入闭环。
- **S06 BatchNorm ablation**: 能力#6：训练稳定性技术。同架构下 BN 通常更快更稳。
- **S07 3x3 stacks (VGG intuition)**: 能力#7：深度设计直觉。过深可能退化 → 引出残差。
- **S08 ResNet residual breakthrough**: 能力#8：可训练的深度。2015 分水岭——现代主干默认残差。
- **S09 depthwise separable conv**: 能力#9：效率轴。研究也要报成本，不只报 top1。
- **S10 modern training recipe**: 能力#10：工业级训练配方。SOTA 提升常来自 recipe。
- **S11 Squeeze-and-Excitation**: 能力#11：通道注意力。ViT 之前的实用注意形态。
- **S12 minimal Vision Transformer**: 能力#12：全局自注意力。小数据常弱于 CNN → 需配方/预训练。
- **S13 ConvNeXt modern CNN**: 能力#13：现代 CNN 双路线。注意力非唯一答案。
- **S14 SimCLR-style contrastive SSL**: 能力#14：无标签表征学习。linear probe 是标准度量。
- **S15 prototype cosine (CLIP interface)**: 能力#15：开放集接口。换原型即可扩类（真 CLIP 用文本塔）。
- **S16 transfer learning protocols**: 能力#16：迁移协议。少标签下 probe/FT 通常碾压从头训。
- **S17 toy MAE (masked autoencoder)**: 能力#17：掩码图像建模。MAE 路线与对比学习并列的 SSL 支柱。
- **S18 DeiT-style data-efficient ViT recipe**: 能力#18：数据高效 ViT 配方。架构+配方缺一不可。
- **S19 MiniCLIP dual-encoder (from scratch)**: 能力#19：多模态对齐接口。真 CLIP 换文本编码器即可扩展任意类名。
- **S20 modern evaluation protocol**: 能力#20：完整评测协议。只有 top1 不够，校准与 OOD 同属现代系统。

可视化目录: `from_scratch/results/viz/`
分步笔记: `from_scratch/notes/`
