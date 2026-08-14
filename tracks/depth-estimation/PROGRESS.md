# PROGRESS · 深度估计学习工程

最后更新：2026-08-10  
状态：**P0–P7 ✅ · From-Scratch FS00–FS14 全链 ✅**

## From Scratch 全链（权威地图：`FROM_SCRATCH.md`）

| FS | 主题 | 状态 | 入口 | 可观察产物 |
|----|------|------|------|------------|
| FS00 | 指标与对齐 | ✅ | `fs00_metrics_vis.py` | `results/fs00_metrics/*.png` |
| FS01 | 针孔↔点云 | ✅ | `fs01_pointcloud_vis.py` | `cloud_fx500.xyz` |
| FS02 | 块匹配双目 | ✅ | `fs02_block_match_stereo.py` | 弱纹理 total failure |
| FS03 | 代价体积 | ✅ | `fs03_cost_volume.py` | WTA vs smooth |
| FS04 | 最小单目回归 | ✅ | `fs04_tiny_regressor.py` | 域偏移 0.04→0.41 |
| FS05 | 多尺度+SILog | ✅ | `fs05_multiscale_silog.py` | `compare.png` |
| FS06 | U-Net skip | ✅ | `fs06_skip_ablation.py` | 边界 0.20→0.02 |
| FS07 | 光度重投影 | ✅ | `run_p3_photometric.py` | bad/good ≈460× |
| FS08 | automask/min-reproj | ✅ | `fs08_automask_minreproj.py` | `automask_demo.png` |
| FS09 | 相对深度协议 | ✅ | `run_p4_foundation_protocol.py` | LS≪none |
| FS10 | MiDaS 真权重 | ✅ | `fs10_midas_dpt_infer.py` | rank corr 0.51 vs rnd 0.00 |
| FS11 | 伪标签学生 | ✅ | `fs11_depth_anything_student.py` | teacher→student |
| FS12 | 卷尺度量测试 | ✅ | `fs12_metric_tape_test.py` | metric≈1m vs 错尺度 |
| FS13 | 生成式迷你扩散 | ✅ | `fs13_generative_depth_mini.py` | uncertainty map |
| FS14 | 研究闭环 | ✅ | `fs14_research_closed_loop.py` | `REPORT.md` |

一键复跑：

```bash
cd tracks/depth-estimation
python scripts/run_all_fs.py
```

依赖：`numpy scipy torch matplotlib timm opencv-python-headless`（venv 已装）

## 研究轨 P0–P7

| Phase | 状态 |
|-------|------|
| P0–P7 | ✅ 见历史提交 `a58bd8d` 等 |

## 能力阶梯（执行后应能口述）

1. **评测诚实**：对齐改变数字；逆深度陷阱可视化  
2. **几何**：深度误差 = 3D 缩放；焦距改变 X 跨度  
3. **经典立体**：有纹理能算出深度；无纹理匹配死亡  
4. **学习单目**：能过拟合；换域崩  
5. **Eigen/U-Net**：SILog + skip 改善  
6. **自监督**：光度解释几何；automask 抑制动态区  
7. **相对基础模型**：MiDaS 真权重 + 伪标签蒸馏  
8. **度量**：卷尺测试；错尺度/错焦距失败  
9. **生成式**：迭代+不确定性（toy）  
10. **研究**：claim 卡 + 消融表 + 威胁效度  

## 路线修正

| 项 | 说明 |
|----|------|
| FS10 验收 | 合成「颜色=深度」上 AbsRel 不可靠 → 改用 **排序相关** + naturalish 场景 |
| FS11 教师 | DA-V2 大权重未绑 → **MiDaS_small 作教师**（配方同构） |
| FS13 | 非 Marigold 全量 → **可运行 toy diffusion** 讲清能力差 |
| 数据 | CPU 友好 synthetic；真 NYU/KITTI 可在 Kaggle 热替换 |

## 当前任务

**全链验收完成。** 可选增强（非阻塞）：
- Kaggle 挂 NYU + DA-V2 / Depth Pro 真权重重跑 FS10–12  
- 替换 `data/fs_pack/` 为用户自己的照片输入  

## 问题日志

| 问题 | 修复 |
|------|------|
| MiDaS 嵌套 hub 需信任 | `trusted_list` 写入 `rwightman_gen-efficientnet-pytorch` |
| 缺 cv2/timm | pip 安装 |
| FS12 高度算错 | fronto 解析公式 + 像素高度适配画布 |
| FS10 AbsRel vs random | 改 ordinal 相关验收 |
