# P2 · 有监督最小可训

## 原理

Encoder-decoder 预测稠密深度；**SILog** 对尺度更稳；输出需保证 \(d>0\)（本实现用 `max_depth * sigmoid`）。

## 实验设定

- 数据：easy synthetic（颜色编码深度 + 矩形物体）— 在 **CPU 短预算** 下保证可收敛。  
- 消融：SILog vs L1；base 16 vs 8。

## 结果（见 `results/p2_supervised/results.json`）

- untrained AbsRel ~0.47  
- silog_b16 AbsRel ~0.014，δ1≈1.0  
- l1_b16 略低 AbsRel；silog_b8 略差于 b16  

## 验收

训练后 AbsRel < 0.25 且显著优于 untrained ✅

## 局限

非 NYU 真值；证明 **训练环+指标+消融流程**，不宣称 SOTA。
