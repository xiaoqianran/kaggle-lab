# 06-frontier · P6 真实域与开放词汇

**出口：** 在 Kaggle 域上给出可辩护 baseline + 错误分析；开放词汇至少完成推理级对照。

## 绑定来源（≤3）

1. catalog 中进行中/经典赛之一（Wheat / Reef / RSNA Knee / 车辆等）  
2. OWL-ViT **或** Grounding DINO 择一  
3. 对应高票 kernel **仅** 作数据管线参考

## 六段清单（det-06）

- [ ] 原理：域移（光照/尺度/标注规范）  
- [ ] 实现：竞赛数据 → 训练格式  
- [ ] 源码：提交 encode / 后处理  
- [ ] Kaggle：baseline notebook  
- [ ] 消融：预训练权重或 imgsz  
- [ ] 解释：FP/FN 按场景切片  

## 纪律

LB 分数 ≠ 研究主指标；主指标仍固定协议 val。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
