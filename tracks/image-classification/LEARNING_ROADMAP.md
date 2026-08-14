# 图像分类 · 研究导向学习蓝图

> **本文件只做规划，不教学。**  
> 目标：最短路径达到「能独立读最新论文、判断贡献、复现、设计 baseline/ablation、提出研究假设」的研究能力。  
> 原则：每个主题 **1–3 个最优来源**；顺序按研究能力最短路径，不按教材章节。  
> 领域锁定：**Image Classification（图像分类）** — 含多类/细粒度/长尾/鲁棒性；检测分割等相邻任务仅作对照。

最后更新：2026-08-10

---

## 0. 一句话问题定义

| 任务 | 输入 | 输出 | 核心难点 |
|------|------|------|----------|
| **闭集图像分类** | 图像 \(x\) | 类别 \(y \in \{1..C\}\) | 表征、优化、过拟合、分辨率/算力权衡 |
| **迁移 / 微调** | 预训练表征 + 目标域 | 目标域标签 | 域移、标签噪声、少样本、超参敏感性 |
| **开放世界 / 鲁棒** | 分布外、扰动、长尾 | 校准预测 / 拒绝 | OOD、校准、公平、细粒度混淆 |

研究主线默认：**监督 ImageNet 范式 → 现代架构与训练配方 → 自监督/多模态预训练 → 迁移与消融研究**。

---

## 1. 领域知识地图

### 1.1 分层（前置 / 核心 / 进阶 / 前沿 / 可跳过）

```text
前置 PRE
  ├─ 线性代数直觉：矩阵乘、范数；概率：交叉熵、softmax、校准直觉
  ├─ 优化：SGD/AdamW、学习率、正则（weight decay）、batch 与噪声
  ├─ 数据：train/val/test 泄漏；类别不平衡；增强（几何/颜色/Mix）
  ├─ 指标：Top-1/5、macro-F1、AUROC（不平衡）、ECE（校准）
  └─ 工程：PyTorch Dataset/DataLoader、混合精度、固定 seed 复现

核心 CORE（研究能力的骨架）
  ├─ 卷积归纳偏置：conv/pool、感受野、参数共享
  ├─ 残差与归一化：ResNet、BN/LN、初始化与可训深度
  ├─ 现代 CNN / ViT：ConvNeXt 设计课、ViT 补丁+注意力
  ├─ 训练配方：cosine LR、warmup、label smoothing、EMA、RandAug/Mixup/CutMix
  ├─ 迁移学习：线性探测 vs 全微调 vs 差分 LR；冻结策略
  └─ 评测协议：ImageNet 协议、固定 seed、多次 run 方差、公平对照

进阶 ADV
  ├─ 高效架构：EfficientNet 复合缩放思想；MobileNet 深度可分（部署向）
  ├─ 自监督：SimCLR / MoCo 对比学习；MAE 掩码重建
  ├─ 多模态预训练：CLIP 式图文对齐 → zero-shot / linear probe
  ├─ 长尾 / 细粒度 / 标签噪声：重采样、logit 调整、损失函数族
  └─ 鲁棒与校准：对抗扰动入门、温度缩放、OOD 检测接口

前沿 FRONTIER（读论文主战场）
  ├─ 训练即架构：DeiT/DeiT-III、ConvNeXt-V2、现代 ImageNet recipe
  ├─ 缩放律与数据质量：更大预训练 vs 清洗数据；合成数据
  ├─ 高效微调：LoRA/Adapter 在 ViT 上；test-time adaptation
  ├─ 开放词汇分类：CLIP 族、SigLIP、与封闭集 fine-tune 权衡
  └─ 可信：公平、校准、分布移、医学/遥感跨域

可跳过 SKIP（除非课题需要）
  ├─ 手写完整 Caffe/旧框架教程
  ├─ 穷尽 2012–2016 所有 CNN 变体（Alex/VGG 够用直觉即可）
  ├─ 纯 AutoML 搜索栈（NAS 论文海）作为主路径
  ├─ 检测/分割/跟踪完整栈（先分类主线，需要时再开）
  ├─ 大量二流“从零实现XXX”合集（用 1 条官方+1 次手写即可）
  └─ 以 Kaggle 金牌为唯一目标而无固定评测协议
```

### 1.2 概念关系（研究时用这张图导航）

```text
数据与增强 ──决定可学信号──▶ 损失（CE / 对比 / 掩码）
        │
        ▼
架构：CNN 归纳偏置 ──▶ 残差深度 ──▶ 现代 CNN(ConvNeXt) ‖ ViT 全局注意
        │
预训练：监督 ImageNet ──▶ SSL (对比/MAE) ──▶ 多模态 (CLIP)
        │
迁移：linear probe ──▶ full FT ──▶ 领域特化（长尾/细粒度/医学）
        │
评测：In-domain Top-1 ──▶ 鲁棒/OOD/校准 ──▶ 真实部署约束（延迟/显存）
```

### 1.3 能力终点（验收标准）

你能独立完成：

1. 读 2024–2026 分类/表征新论文，5 分钟说清 **问题设定 / 假设 / 方法增量 / 实验是否支撑 claim**  
2. 在 CIFAR-10/100 与 **ImageNet-子集或完整协议** 上训通 **ResNet 与 ViT/ConvNeXt 至少各一**，报告可复现 Top-1  
3. 精读 **timm 训练脚本 + 至少 1 个官方模型实现**，能定位：增强、优化器、EMA、模型 forward  
4. 做迁移消融：backbone × 冻结策略 × 增强 × LR schedule；解释为何某一配置胜出  
5. 提出可检验假设（例：错误来自类间外观混淆 vs 标签噪声；ViT 需要更强增强 vs 更大预训练）

---

## 2. 全球最佳资料筛选（每个主题 ≤3）

> 交叉比较后 **只保留最值得投入的**；未列入 ≠ 差，而是边际收益低。

### 2.1 前置：ML/DL 语言与训练直觉

| # | 来源 | 为何入选 | 用法 |
|---|------|----------|------|
| 1 | **Stanford CS231n** 笔记 + Assignments（[cs231n.github.io](https://cs231n.github.io/) · [cs231n.stanford.edu](https://cs231n.stanford.edu/)） | 全球事实标准：分类问题定义、softmax、CNN、训练调试 | **主路径**：A1/A2 相关模块必做；Lecture 5–6 精读 |
| 2 | **MIT 6.S191** Deep Computer Vision 讲 + Lab（[introtodeeplearning.com](https://introtodeeplearning.com/) · [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning)） | 高强度短课，快速对齐现代 DL 语汇 | 1–2 天过 CV 讲，**不替代** CS231n 作业深度 |
| 3 | **PyTorch 官方 Tutorials**（Training a Classifier / Transfer Learning） | 工程 API 真源，避免过时博客 | 只当 API 字典 |

**不优先：** 冗长中文“百日打卡”、过时 Theano/TF1 教程。

### 2.2 核心：经典 CNN 与可训深度

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **He et al. ResNet** (CVPR 2016) *Deep Residual Learning for Image Recognition* | 深度可训的分水岭；残差=研究公共语 | 手写 Mini-ResNet + 读 `torchvision`/`timm` 实现 |
| 2 | **CS231n** Convolutional Networks notes + A2 CNN 部分 | 强制手写 conv 与 backprop 直觉 | 作业代码 |
| 3 | （对照）**Ioffe & Szegedy BN** 或讲义中 BN 一节 | 理解归一化如何改变优化景观 | 概念+小消融即可 |

### 2.3 核心：现代架构（CNN 现代化 + ViT）

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Liu et al. ConvNeXt** (CVPR 2022) *A ConvNet for the 2020s* | “用 Transformer 训练配方重塑 CNN”的设计课，消融教科书级 | [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) **必读** |
| 2 | **Dosovitskiy et al. ViT** (ICLR 2021) *An Image is Worth 16x16 Words* | 视觉 Transformer 原点；理解 patch、位置编码、规模依赖 | 读论文 + `timm` ViT 实现 |
| 3 | **Touvron et al. DeiT** (ICML 2021) | 证明 **数据高效训练配方** 让 ViT 在 ImageNet-1k 可训 | [facebookresearch/deit](https://github.com/facebookresearch/deit) |

### 2.4 核心：训练配方与工程复现

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **timm (PyTorch Image Models)** [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 分类领域事实标准库：模型+训练脚本+可复现 recipe | **精读** `train.py` / `validate.py` / 目标模型 `*.py` |
| 2 | **Wightman / 社区 ImageNet training notes**（timm docs + model cards） | 把“能跑”变成“能复现论文量级” | 对照论文表设超参 |
| 3 | **torchvision references/classification** | 官方第二参考实现，适合对照差异 | 差异列表写入笔记 |

### 2.5 进阶：自监督表征

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Chen et al. SimCLR** (ICML 2020) | 对比学习范式清晰、消融完整 | 思想复现 + 官方/轻量实现 |
| 2 | **He et al. MAE** (CVPR 2022) | 生成式 SSL 主线；与对比学习对照 | [facebookresearch/mae](https://github.com/facebookresearch/mae) |
| 3 | （可选）**MoCo v2/v3** 一篇 | 理解 memory bank / 动量编码器工程 | 只读关键节 |

### 2.6 进阶：多模态与开放词汇

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Radford et al. CLIP** (ICML 2021) | 图文预训练 → zero-shot 分类范式转移 | [openai/CLIP](https://github.com/openai/CLIP) |
| 2 | **OpenCLIP** 或 **timm + open_clip** 生态 | 可复现训练/权重的社区真源 | 推理+线性探测实验 |

### 2.7 进阶：迁移、长尾、细粒度（研究常用设定）

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | **CS231n Transfer Learning** 讲义 + 自设计 Kaggle 迁移实验 | 先掌握正确 baseline，再谈新方法 |
| 2 | **长尾综述级单篇**（如 LTML 领域高引 survey **择一 2022–2024**）+ **LDAM/Balanced Softmax 等一篇方法** | 建立长尾问题与损失改造直觉 |
| 3 | **细粒度**（CUB/Stanford Cars 协议）任选 1 篇 2020 后 SOTA 作对照 | 理解 part/注意力/强增强路线，不陷入竞赛刷点 |

### 2.8 高校课程：怎么用（不堆课）

| 课 | 角色 | 取舍 |
|----|------|------|
| **Stanford CS231n** | **主课** | 作业+笔记贯穿 P0–P2 |
| **MIT 6.S191** | 加速器 | 只取 CV 讲与 lab |
| **Berkeley EECS 182 / CS182**（Deep Nets） | 训练与可视化加深 | 可选 2–3 讲；[eecs182.org](https://eecs182.org/) / 历史 sp21 |
| **CMU 16-385 / 16-720** | 经典视觉广度 | 分类仅取 recognition 相关；**不**转去几何主线 |
| **THU/PKU** 公开 CV/DL 课 | 中文术语对照 | **不作为主路径**；有公开作业再局部引用 |
| Fast.ai Practical DL | 工程速度 | 可作补充，**主张与论文协议冲突时以论文/timm 为准** |

### 2.9 评测基准与 Kaggle 任务（工程闭环）

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | **CIFAR-10/100** | 快速迭代架构与配方；单元测试级实验 |
| 2 | **ImageNet-1k 协议**（或 ImageNette/ImageWoof 子集做算力受限替代） | 论文对话语言；子集须在报告中声明 |
| 3 | **Kaggle 闭环**（见 `catalog.json`）：Cassava、ISIC、RSNA Knee（进行中）等 | GPU 提交与真实脏数据；**研究主指标仍用固定协议** |

---

## 3. 最优学习顺序（最短路径 → 研究能力）

> 顺序逻辑：**先能量化对错 → 手写最小可训 → 残差与现代架构 → 工业级训练配方 → SSL/多模态 → 迁移研究 → 独立假设**。  
> **不是** 从 AlexNet 编年史线性扫到 2026。

| Phase | 名称 | 周期感 | 出口能力 | 绑定实验 |
|-------|------|--------|----------|----------|
| **P0** | 指标、数据协议、复现纪律 | 2–4 天 | 会算 Top-1/F1/ECE；会查泄漏；固定 seed；能解释“假 SOTA” | `cls-00` + `tests/test_metrics.py` |
| **P1** | 线性分类与 softmax 地基 | 3–5 天 | 手写 softmax/CE；kNN/线性分类基线；偏差-方差直觉 | `cls-01`（CS231n A1 思想） |
| **P2** | CNN + ResNet 最小可训 | 1–2 周 | CIFAR 上训通小 CNN 与 ResNet-18；画训练曲线；debug 过拟合 | `cls-02` 手写 train loop |
| **P3** | 现代架构对照 | 1–2 周 | ConvNeXt 设计消融读透；ViT/DeiT 推理+小规模微调；对比归纳偏置 | `cls-03` + vendor ConvNeXt/DeiT |
| **P4** | 训练配方（timm 级） | 1–2 周 | 复现一组增强+schedule 消融；读透 timm `train.py` | `cls-04` recipe ablations |
| **P5** | SSL 与 CLIP 迁移 | 1–2 周 | MAE/SimCLR 思想实验；CLIP zero-shot vs linear probe vs FT | `cls-05` |
| **P6** | 真实域 / Kaggle 研究协议 | 1–2 周 | 在 Cassava 或医学赛上做 **可辩护** baseline+ablation | `cls-06` |
| **P7** | 独立研究 | 持续 | 读新论文 claim 审查；提出假设；阴性结果也写清楚 | `07-research-lab` |

### 3.1 每个核心知识点的六段绑定（模板）

所有 `cls-NN` 必须填满：

| 段 | 含义 | 交付物 |
|----|------|--------|
| **原理** | 公式/假设/失效条件 | `notes.md` 半页 |
| **手写实现** | 最小可运行代码（不抄整库） | `src/` 或 notebook 核心 cell |
| **GitHub 源码** | 指定文件/函数精读 | `papers/*/SOURCE_MAP.md` |
| **Kaggle 实验** | 可 GPU 复现 | `.ipynb` + 固定 seed |
| **消融** | ≥2 个可控变量 | 表：变量→指标→解释 |
| **结果解释** | 成功/失败归因 | 文字 + 误差可视化（混淆矩阵/困难样本） |

### 3.2 关键论文阅读栈（按此顺序读原文）

1. **ResNet** 2016（深度与残差 DNA）  
2. **ViT** 2021 + **DeiT** 2021（架构分叉与数据/配方）  
3. **ConvNeXt** 2022（设计消融课）  
4. **SimCLR** 或 **MAE** 二选一深读，另一篇扫实验表  
5. **CLIP** 2021（开放词汇与迁移）  
6. 当周 arXiv 分类/表征 1 篇（训练“判断贡献”）

---

## 4. GitHub / Kaggle 学习工程结构

本仓库路径（已创建）：

```text
tracks/image-classification/
├── LEARNING_ROADMAP.md          # 本蓝图
├── README.md                    # 任务地图（竞赛/数据速查）
├── catalog.json                 # 机器可读索引
│
├── 00-map/                      # 概念卡、术语表、指标定义
├── 01-foundations/              # P0–P1：指标、线性分类、softmax
├── 02-cnn-classic/              # P2：CNN / ResNet 手写与训练
├── 03-modern-arch/              # P3：ConvNeXt / ViT / DeiT
├── 04-training-recipe/          # P4：timm 配方与消融
├── 05-ssl-transfer/             # P5：SSL / CLIP / 迁移
├── 06-frontier/                 # P6–P7：新论文、失败 case、Kaggle 域
├── 07-research-lab/             # 个人假设、实验日志、claim 审查
│
├── papers/                      # 阅读笔记（不存 PDF 大文件）
│   └── TEMPLATE.md
├── scripts/                     # metrics、增强、可视化、下载
├── tests/                       # 指标与数据管线单元测试（防自欺）
├── data/                        # 本地缓存（gitignore）
├── vendor/                      # clone 官方源码（gitignore 权重）
└── results/                     # 图、表、checkpoint 指针（大文件不入库）
```

### 4.1 Notebook 命名

```text
cls-00-metrics-protocol-seed.ipynb
cls-01-softmax-knn-linear.ipynb
cls-02-cifar-resnet-from-scratch.ipynb
cls-03-convnext-vit-contrast.ipynb
cls-04-timm-recipe-ablation.ipynb
cls-05-ssl-clip-transfer.ipynb
cls-06-kaggle-domain-baseline.ipynb
cls-rNN-<hypothesis-slug>.ipynb    # 研究假设实验
```

### 4.2 源码阅读约定（强制）

| 库 | 必读入口（实现时锁定具体文件） |
|----|--------------------------------|
| timm | `train.py`、`validate.py`、目标模型定义、`data/` 增强 |
| ConvNeXt 官方 | 网络 block、训练配置、消融对应 flag |
| DeiT / ViT | patch embed、attention、distillation token（DeiT） |
| torchvision ResNet | `forward`、downsample 捷径、与 timm 差异 |
| CLIP / OpenCLIP | encode_image/text、logit scale、preprocess |

阅读产出：`papers/<paper>/SOURCE_MAP.md`（文件→函数→对应论文公式/表格）。

### 4.3 Kaggle 使用策略

| 用途 | 做法 |
|------|------|
| GPU 训练/推理 | Notebook 挂 CIFAR / 竞赛数据；固定 `seed`、记录 commit hash |
| 模型权重 | Kaggle Models / timm 预训练；**记录权重版本** |
| 提交闭环 | Cassava 历史赛复盘 **或** 进行中 RSNA Knee 等；**主研究指标仍用固定协议集** |
| 对照 notebook | 高票仅作数据管线参考，**不以抄结果为终点** |

### 4.4 实验日志最小字段

```yaml
exp_id: cls-02-a1
commit: <git sha>
data: cifar10_v1
model: resnet18
recipe: {lr: 0.1, aug: basic, epochs: 50}
metrics: {top1: …, macro_f1: …, ece: …}
ablation: {var: value}
claim: "…"
result_interpretation: "…"
next: "…"
```

---

## 5. 持续测试协议（后续教学阶段执行）

每结束一个 Phase：

1. **解释**：用自己的话推导关键公式（无笔记）  
2. **预测**：改一个超参/消融，先写预测再跑  
3. **Debug**：故意注入错误（标签打乱、增强过强、BN 在 eval 误用、数据泄漏），定位  
4. **独立实现**：合上参考代码写最小版  
5. **缺口回退**：任一失败 → 回到 PRE/CORE 对应节点，不跳级  

---

## 6. 刻意不做什么（保证路径最短）

- 不从“100 篇综述列表”开始  
- 不并行主修检测/分割/生成完整栈  
- 不收集 20 个相似 GitHub “classification zoo”  
- 不把 Kaggle 排行榜当研究贡献  
- 不跳过指标与 seed 纪律直接抄 SOTA notebook  
- 不把“调用 `timm.create_model` 微调一次”当成完成 P3–P4  

---

## 7. 下一步（等你确认后开始教学）

1. 冻结本蓝图 v1  
2. 从 **P0：指标与复现纪律** 开工（`cls-00` + `tests/test_metrics.py`）  
3. 同步 clone 必读源码到 `vendor/`（仅代码，权重按需）  
4. 每次阶段交付后 **阿里规范：pull → commit → push**

确认后回复：**开始 P0** 或指定想先攻的 Phase。

---

## 8. 执行修订记录（2026-08-10）

| 修订 | 原因 | 影响 |
|------|------|------|
| 默认实验数据：FashionMNIST→32×32 RGB | 沙箱内 CIFAR-10 官方源带宽不可用 | 协议/代码不变；数字不可直接对比 ImageNet 论文 |
| ResNet 使用 thin residual（非完整 R-18） | CPU 时限 | 仍覆盖残差学习与训练循环 |
| P0–P7 已跑通并写入 `results/` + `PROGRESS.md` | 持续执行模式 | 见 `PROGRESS.md` 验收表 |

