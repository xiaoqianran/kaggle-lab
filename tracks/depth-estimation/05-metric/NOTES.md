# P5 · 度量深度协议

## 原理

度量模型报告 **align=none**。Median 对齐会 **隐藏** 全局尺度偏差（见 scale_bias_1.15）。  
焦距错误 → \(Z \propto f\) 线性尺度错误；3D 点 \(X=(u-c)Z/f\) 同步膨胀。

## 实验族（可替换为真权重）

oracle / scale_bias / relative_misuse / depth_pro_like / metric3d_like / unidepth_like  

## 验收

oracle AbsRel < 0.1；relative_misuse ≫ oracle ✅  

## 真实权重说明

无多 GB 权重缓存时，协议与焦距/3D 耦合实验为可执行核心；Kaggle 上可挂 `intel/midas`、`depth-anything` 替换 simulate 函数。
