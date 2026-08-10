# 05-training-recipe · P5 配方与消融

**出口：** 同数据、同协议下完成 ≥2 变量消融表；会改 ultralytics 或 mmdet 配置。

## 绑定来源（≤3）

1. Ultralytics 默认 recipe（aug、EMA、scheduler）  
2. MMDetection config 体系（研究向）  
3. 一篇 YOLO/DETR 论文的消融表（学“怎么设计消融”）

## 六段清单（det-05）

- [ ] 原理：哪些超参改的是优化景观 vs 数据分布  
- [ ] 实现：配置化实验脚本  
- [ ] 源码：trainer 中 aug 注入点  
- [ ] Kaggle：同 val 协议多 run  
- [ ] 消融：主表（强制）  
- [ ] 解释：效应量与方差，避免一次 run 下结论  

## 记录

使用蓝图 §4.4 YAML 字段写入 `results/`。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
