# P1 · 几何最小集

## 原理

针孔：\(u = f_x X/Z + c_x,\ v = f_y Y/Z + c_y\)。  
反投影：\(X = (u-c_x)Z/f_x\) 等。  
立体：\(Z = f B / d\)。  
自监督核心：已知深度 + 相对位姿 → 源图采样 → 光度一致性。

## 交付

- `scripts/geometry.py`
- `tests/test_geometry.py` ✅（投影往返、视差可逆、平移引起的 u 位移）

## 与后续连接

P3 的 `warp_with_pose` 是同一几何的 torch 版；读 Monodepth2 `layers.py` 对照。
