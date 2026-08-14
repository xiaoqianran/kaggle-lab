# P7 · 个人研究假设

## H1

在固定步数下，**边缘加权损失** 对边界 AbsRel 的改善 ≥ 小幅加宽网络；边界误差主导整体误差。

## 实验

`scripts/run_p7_hypothesis.py` → `results/p7_hypothesis/results.json`

## 结果解读

以当次运行为准（CPU 短预算噪声大）。  
- 若 `supported=true`：在该预算下边缘项相对容量项更有效或边界增益更大。  
- 若 `false`：修订 H1（加大 edge weight、更长训练、真实 NYU 边界 mask）。

## 验收

假设被 **可重复脚本检验**，指标有限，写出 interpretation ✅  
（研究验收 ≠ 假设必须为真）
