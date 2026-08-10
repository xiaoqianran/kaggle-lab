# P0 · 评测与对齐

## 原理（最小集）

- **AbsRel** \(\frac{1}{n}\sum |p-g|/g\)：相对误差，对近处敏感。
- **RMSE**：绝对米制误差，受远处 outliers 主导。
- **δ<1.25**：比例阈值内像素占比，论文表格主读数之一。
- **SI-RMSE**：对 \(\log p-\log g\) 去均值后的 RMSE → **对全局尺度免疫**。
- **Median scale**：\(s=\mathrm{med}(g)/\mathrm{med}(p)\)，Eigen monocular 协议。
- **Scale-shift (LS)**：\(s p + t \approx g\)，相对/仿射不变深度（MiDaS/DPT/DA-V2 relative）。

## 失效条件

| 错误 | 表现 |
|------|------|
| 相对深度当米制 | align=none 时 AbsRel 爆炸 |
| 只报 median 对齐的 metric 模型 | 掩盖绝对尺度 bug |
| 反深度未翻转 | LS 也无法救到可用 AbsRel |

## 交付

- 实现：`scripts/metrics.py`
- 测试：`tests/test_metrics.py` ✅
- 实验：`scripts/run_p0_experiment.py` → `results/p0_alignment/results.json`

## 消融结论

1. 纯尺度误差：median / LS 归零；SI-RMSE 本就近 0。  
2. 仿射误差：仅 LS 能归零；median 残留。  
3. 加性噪声：任何对齐都救不了。  
4. 逆深度误用：三种对齐都差。
