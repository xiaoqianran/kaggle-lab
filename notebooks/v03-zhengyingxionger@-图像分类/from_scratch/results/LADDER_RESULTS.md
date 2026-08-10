# From-Scratch Ladder Results

| Step | Title | Metric | Δ top1 | New capability |
|------|-------|--------|--------|----------------|
| S00 | pixels as tensors | None | — | 能力#0：把像素当数据——先看 shape/scale/类分布，再谈模型。 |
| S01 | kNN baseline (no learning) | 0.7100 | — | 能力#1：无学习基线。kNN>>随机 ⇒ 数据有可分结构，后续模型必须超过它。 |
| S02 | linear softmax from scratch | 0.8063 | +0.0963 | 能力#2：参数学习+概率输出。仍是超平面决策，无空间结构。 |
| S03 | MLP on flat pixels | 0.8200 | +0.0137 | 能力#3：深度非线性。常略好于线性，但过拟合与无视空间 → 需要卷积。 |
| S04 | hand-written convolution + feature maps | 0.5188 | — | 能力#4：局部模式提取。参数效率数量级优势；第一次‘看见’边缘。 |
