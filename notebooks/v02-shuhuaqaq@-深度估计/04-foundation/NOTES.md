# P4 · 相对深度 / 基础模型协议

## 原理

Foundation relative depth 输出常是 **视差/仿射不变** 量，**不能** 与米制 GT 直接算 AbsRel。  
必须：`least_squares` scale+shift（或论文声明的对齐）。

## 实验

`run_p4_foundation_protocol.py`：

| 表示 | align=none AbsRel | align=LS AbsRel |
|------|-------------------|-----------------|
| disparity | ~0.94 | ~0.07 |
| log | ~0.70 | ~0.04 |

MiDaS_small：若 torch.hub 可下载则额外评；本环境曾触发 hub zip 下载（可选）。

## 验收

disparity 上 LS AbsRel ≪ none ✅
