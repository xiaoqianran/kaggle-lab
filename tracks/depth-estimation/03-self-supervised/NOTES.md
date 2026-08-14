# P3 · 自监督光度闭环

## 原理（Monodepth2）

1. 预测目标帧深度  
2. 用位姿把源帧 warp 过来  
3. 光度：`0.85*SSIM + 0.15*L1`  
4. **min** 多源 reproj（遮挡）  
5. **automask**：丢掉“不动也更像”的像素  

## 交付

- 手写：`scripts/photometric.py`  
- 源码图：`papers/Monodepth2/SOURCE_MAP.md`  
- 实验：正确深度 vs 2×深度 光度误差比 ≈ **460×**；优化 depth 参数 loss 下降 ✅  

## 验收

`err_bad_depth > err_good_depth` 且 photometric train improved ✅
