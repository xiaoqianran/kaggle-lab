# v03 · zhengyingxionger@ · 图像分类

本目录 = **图像分类研究学习工程**。  
权威学习蓝图：**[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)**（领域地图 · 精选资料 · 最短路径 · 工程结构）。

Kaggle 账号侧用户：`zhengyingxiong`。Notebook 命名：`cls-NN-*.ipynb`。

---

## 快速导航

| 文档 | 内容 |
|------|------|
| [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md) | **主蓝图**（先读这个） |
| [catalog.json](./catalog.json) | 竞赛/数据/模型机器索引 |
| [papers/TEMPLATE.md](./papers/TEMPLATE.md) | 论文笔记模板 |

## 阶段目录

| 目录 | Phase | 主题 |
|------|-------|------|
| `00-map/` | — | 术语、指标、概念卡 |
| `01-foundations/` | P0–P1 | 指标协议、softmax/线性分类 |
| `02-cnn-classic/` | P2 | CNN / ResNet 手写与训练 |
| `03-modern-arch/` | P3 | ConvNeXt / ViT / DeiT |
| `04-training-recipe/` | P4 | timm 配方与消融 |
| `05-ssl-transfer/` | P5 | SSL / CLIP / 迁移 |
| `06-frontier/` | P6 | 新论文、域适应、Kaggle 域 |
| `07-research-lab/` | P7 | 个人假设与实验日志 |
| `scripts/` `tests/` | 全程 | 指标、增强、单元测试 |
| `vendor/` `data/` `results/` | 全程 | 源码/数据/结果（大文件不入库） |

## Kaggle 图像分类任务速查（调研 2026-08-10）

### 竞赛（学习闭环优先）

| 竞赛 | 状态 | 要点 |
|------|------|------|
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | **进行中** (~2026-10) | 医学影像；可作 P6 真实域 |
| [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) | 近期/进行 | 检测向，分类头可借鉴 |
| [ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge) | 已结束 | 皮肤病变；强增强/校准教材 |
| [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | 已结束 | 细粒度植物病害；经典迁移课 |
| [Human Protein Atlas Image Classification](https://www.kaggle.com/competitions/human-protein-atlas-image-classification) | 已结束 | 多标签 |
| [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) | 已结束 | 医学二分类 + 元数据 |

### 数据集（训练主粮）

| 数据集 | 用途 |
|--------|------|
| CIFAR-10/100（`pankrzysiu/cifar10-python` 等） | P1–P2 快速迭代 |
| Fashion-MNIST / Fruits-360 / Intel Image Classification | 入门迁移 |
| ImageNet 子集 / 竞赛官方数据 | P4–P6 协议实验 |

### Models Hub（示例）

`google/resnet-v2` · `tensorflow/resnet-50` · `keras/resnetv1` · 本地 **timm** 预训练为主

---

## 约定

- 每个知识点：**原理 → 手写 → 源码 → Kaggle → 消融 → 解释**
- 每次实质改动：**阿里规范 pull + commit + push**
- 密钥与大权重：**永不入库**
- 研究主指标用固定协议；Kaggle LB 只作工程闭环
