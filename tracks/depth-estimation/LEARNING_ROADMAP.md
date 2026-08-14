# 深度估计 · 研究导向学习蓝图

> **本文件只做规划，不教学。**  
> 目标：最短路径达到「能独立读最新论文、判断贡献、复现、设计 baseline/ablation、提出研究假设」的研究能力。  
> 原则：每个主题 **1–3 个最优来源**；顺序按研究能力最短路径，不按教材章节。

最后更新：2026-08-10

---

## 0. 一句话问题定义

| 任务 | 输入 | 输出 | 核心难点 |
|------|------|------|----------|
| **单目深度估计 (MDE)** | 单张 RGB | 稠密深度图 \(D(u,v)\) | 尺度歧义、遮挡、泛化、边界、度量尺度 |
| **双目/多视深度** | 多图 + 几何 | 深度 / 视差 | 匹配歧义、标定、非朗伯 |
| **度量深度 (MMDE)** | 单图（±相机） | **米制**深度 / 3D 点 | 焦距、绝对尺度、跨域 |

研究主线默认：**单目（相对 → 零样本基础模型 → 度量）**；双目/多视作为几何地基与对照。

---

## 1. 领域知识地图

### 1.1 分层（前置 / 核心 / 进阶 / 前沿 / 可跳过）

```text
前置 PRE
  ├─ 相机模型、投影、内外参、畸变
  ├─ 极几何 / 本质矩阵 / 基础矩阵 / 三角化
  ├─ 立体匹配直觉（代价体积、半全局）
  ├─ 深度学习：CNN encoder-decoder、损失、归一化
  └─ 指标：AbsRel, RMSE, δ<1.25, SI-RMSE, scale-shift align

核心 CORE（研究能力的骨架）
  ├─ 有监督 dense prediction：Eigen → U-Net/FPN 头 → 尺度/位移对齐
  ├─ 自监督：光度重投影、min reproj、automask、位姿网络（Monodepth2）
  ├─ 多数据集 / 相对深度：MiDaS 系 + scale-invariant loss
  ├─ Transformer dense head：DPT 结构与为何利于全局一致
  ├─ 评测协议：Eigen split、cap 80m、median scaling、zero-shot 协议
  └─ 数据：NYU-Depth-v2、KITTI、（可选）ETH3D / DIODE / SYNS

进阶 ADV
  ├─ 基础模型训练范式：伪标签、合成+真实、教师-学生（Depth Anything 系）
  ├─ 度量深度：相机内参条件化、canonical camera、焦距估计
  ├─ 边界与锐度：edge-aware loss、高分辨率多尺度推理
  ├─ Diffusion / generative depth（Marigold 系）
  └─ Challenge 协议：MDEC / 课程竞赛 SI-RMSE

前沿 FRONTIER（读论文主战场）
  ├─ Depth Anything V2、Depth Pro、Metric3D v2、UniDepth(V2)
  ├─ Video / temporal consistency；4K metric prompting
  ├─ 与 3D 表示耦合：Gaussian / multi-view feed-forward（旁支）
  └─ 鲁棒性：镜面/透明、恶劣天气、任意相机模型

可跳过 SKIP（除非课题需要）
  ├─ 手写完整 SGM/传统 stereo 全工程（够用直觉即可）
  ├─ 过时单论文“刷榜 NYU”而无泛化（2015–2018 刷榜族）
  ├─ 纯全景/鱼眼专用管线（进阶后再开）
  ├─ NeRF/3DGS 完整训练栈（属相邻领域，先深度主线）
  └─ 大量二流综述/中文科普转述（用 1 篇英/1 篇中文索引即可）
```

### 1.2 概念关系（研究时用这张图导航）

```text
多视几何 ──保证可观性──▶ 深度真值 / 自监督信号
        │
        ▼
表示：相对视差 z=1/d │ 度量深度 d(m) │ 3D 点云 X=K^{-1} d u
        │
训练信号：有监督 GT ──▶ 自监督光度 ──▶ 大规模伪标签 / 合成
        │
架构：CNN U-Net ──▶ ViT+DPT ──▶ 多尺度 ViT / Diffusion
        │
泛化：单数据集 ──▶ 混合数据集 ──▶ 零样本基础模型 ──▶ 度量零样本
```

### 1.3 能力终点（验收标准）

你能独立完成：

1. 读 2024–2026 新论文，用 5 分钟说清 **问题设定 / 假设 / 方法增量 / 实验是否支撑 claim**  
2. 在 NYU + KITTI 上复现 **supervised baseline** 与 **self-supervised Monodepth2 思想实验**  
3. 跑通并对比 **DA-V2 / Depth Pro / Metric3D 或 UniDepth** 至少三者零样本  
4. 设计消融：损失、对齐方式、分辨率、数据混合、相机条件  
5. 提出可检验假设（例：边界误差来自下采样 vs 标签噪声；度量失败来自焦距 vs 场景先验）

---

## 2. 全球最佳资料筛选（每个主题 ≤3）

> 交叉比较后 **只保留最值得投入的**；未列入 ≠ 差，而是边际收益低。

### 2.1 前置：几何与相机

| # | 来源 | 为何入选 | 用法 |
|---|------|----------|------|
| 1 | **Stanford CS231A** 课程笔记（相机 / 立体 / 多视） [web.stanford.edu/class/cs231a](https://web.stanford.edu/class/cs231a/) | 顶级 CV 几何课，笔记自洽，直达作业级理解 | 精读相机+立体相关 notes，手推投影 |
| 2 | **Hartley & Zisserman – Multiple View Geometry**（Ch.6–9 精读） | 领域圣经；F/E、三角化标准表述 | 当字典，不线性通读全书 |
| 3 | **CMU 16-385** Stereo lecture PDF（Kitani 等） | 短、工程向立体匹配 | 1 次过完建立匹配直觉 |

**不优先：** 纯图形学相机玩具 demo、过长中文转述。

### 2.2 前置：深度学习稠密预测

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | **Stanford CS231N**（CNN / 训练 / 表征，公开 lecture） | 统一训练与表征语言 |
| 2 | **U-Net / FPN 原论文 + 任一标准实现** | MDE decoder 通用骨架 |

### 2.3 核心：有监督单目

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Eigen et al. 2014** *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network* | 现代学习式 MDE 开端；多尺度、scale-invariant 思想源头 | 思想复现即可 |
| 2 | 标准 **encoder-decoder + SILog / L1** 教学实现（自写） | 强制掌握数据管线与指标 | 本仓库 `02-supervised-classic` |
| 3 | **评测协议笔记**：Eigen split、median scaling、depth cap | 论文表格是否可信的钥匙 | `scripts/metrics.py` |

### 2.4 核心：自监督

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Monodepth2** Godard et al. ICCV 2019 | 自监督范式“标准答案”：min reproj / automask / full-res multi-scale | [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2) **必读源码** |
| 2 | （可选对照）Zhou et al. 2017 SfMLearner | 理解 pose+depth 联合的起源 | 只读思想，不深陷 |

### 2.5 核心：稳健相对深度 / 多数据

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **MiDaS** Ranftl et al.（及 v2/v3 演进） | 多数据集混合 + 相对深度工业标准 | [isl-org/MiDaS](https://github.com/isl-org/MiDaS) |
| 2 | **DPT** Ranftl et al. ICCV 2021 | ViT 做 dense prediction 的结构范式 | [isl-org/DPT](https://github.com/isl-org/DPT) |

### 2.6 进阶：基础模型（相对/伪标签）

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Depth Anything V2** Yang et al. NeurIPS 2024 | 当前相对深度基础模型事实标准；训练配方可迁移 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| 2 | **Depth Anything V1** | 理解伪标签与大规模无标注数据的论证链 | 读论文 + 与 V2 对比 |

### 2.7 进阶：度量深度

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Depth Pro** Richter & Koltun et al. ICLR 2025 | 零样本度量 + 边界 + 焦距；消融示范强 | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) |
| 2 | **Metric3D / Metric3D v2** | 规范相机空间、零样本度量与法线 | [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) |
| 3 | **UniDepth / UniDepthV2** | 直接预测度量 3D、相机无关设定 | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |

**中文索引（仅 1 本）：**  
- 《基于深度学习的单目深度估计方法综述》等 **择一 2022–2025 核心期刊综述** 做中文术语对照；**论证与公式以英文原论文为准**。

### 2.8 进阶：生成式深度

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | **Marigold**（diffusion dense prediction） | 生成式深度代表；理解与判别式 trade-off |

### 2.9 评测、竞赛与研究社区

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | **MDEC**（CVPR Monocular Depth Estimation Challenge） [jspenmar.github.io/MDEC](https://jspenmar.github.io/MDEC/) | 零样本泛化、对齐协议的社区标准 |
| 2 | **NYU-Depth-v2 + KITTI Eigen** | 不可绕开的室内/室外基准 |
| 3 | **ETHZ CIL Monocular Depth 2025**（Kaggle 课程竞赛） | 可提交、SI-RMSE、适合工程闭环 | [竞赛页](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025) |

### 2.10 高校课程：怎么用（不堆课）

| 课 | 角色 | 取舍 |
|----|------|------|
| **Stanford CS231A** | 几何主课 | **精读** 相机/立体 notes |
| **Stanford CS231N** | DL 主课 | 只补缺口 |
| **MIT 6.8300 Advances in CV**（多视几何 lecture） | 几何加深 | 可选 1–2 讲 |
| **CMU 16-385** | 立体短讲 | 1 次过 |
| Berkeley 技术报告 *Progress and Proposals: MDE* (2021) | 历史视角 | 可选浏览，不主修 |
| THU/PKU 公开课 | 国内对照 | **不作为主路径**；有公开讲义再局部引用 |

---

## 3. 最优学习顺序（最短路径 → 研究能力）

> 顺序逻辑：**先能度量对错 → 能做监督 baseline → 理解无 GT 信号 → 理解今日 SOTA 训练配方 → 度量与零样本 → 独立研究**。  
> **不是** 从 2014 线性扫到 2026。

| Phase | 名称 | 周期感 | 出口能力 | 绑定实验 |
|-------|------|--------|----------|----------|
| **P0** | 评测与数据直觉 | 2–4 天 | 会算指标、会做 scale-shift 对齐、能解释“假 SOTA” | `de-00` 指标单元测试 + 假深度 sanity |
| **P1** | 几何最小集 | 4–7 天 | 手推针孔投影、双目视差→深度、知尺度与焦距关系 | `de-01` 几何 notebook（无网络） |
| **P2** | 有监督最小可训 | 1–2 周 | 在 NYU 子集上训通小 U-Net，复现合理 AbsRel 量级 | `de-02` 手写 train loop |
| **P3** | 自监督思想闭环 | 1–2 周 | 读透 Monodepth2 源码；复现/迷你版光度损失；预测消融结果 | `de-03` + vendor monodepth2 |
| **P4** | 相对深度基础模型 | 1 周 | MiDaS/DPT 推理；理解多数据集与对齐；DA-V2 推理与失败模式 | `de-04` Kaggle 推理对比 |
| **P5** | 度量与零样本 SOTA | 1–2 周 | Depth Pro / Metric3D / UniDepth 对照；焦距与尺度实验 | `de-05` 三模型协议评测 |
| **P6** | 研究协议 | 持续 | 读 MDEC/新论文；写 1 页 claim 审查；设计 ablation 表 | `07-research-lab` |
| **P7** | 个人假设 | 持续 | 提出假设 → baseline → ablation → 阴性/阳性结果解释 | 独立项目 |

### 3.1 每个核心知识点的六段绑定（模板）

所有 `de-NN` 必须填满：

| 段 | 含义 | 交付物 |
|----|------|--------|
| **原理** | 公式/假设/失效条件 | `notes.md` 半页 |
| **手写实现** | 最小可运行代码（不抄整库） | `src/` 或 notebook 核心 cell |
| **GitHub 源码** | 指定文件/函数精读 | `papers/*/SOURCE_MAP.md` |
| **Kaggle 实验** | 可 GPU 复现 | `.ipynb` + 固定 seed |
| **消融** | ≥2 个可控变量 | 表：变量→指标→解释 |
| **结果解释** | 成功/失败归因 | 文字 + 误差可视化 |

### 3.2 关键论文阅读栈（按此顺序读原文）

1. Eigen 2014（问题与指标 DNA）  
2. Monodepth2 2019（自监督标准）  
3. MiDaS / DPT（稳健相对深度 + ViT dense）  
4. Depth Anything V2 2024（基础模型配方）  
5. Depth Pro 2025 **或** Metric3D v2 **或** UniDepth（度量三选一深读，其余扫摘要+实验表）  
6. 当周 arXiv 1 篇（训练“判断贡献”）

---

## 4. GitHub / Kaggle 学习工程结构

本仓库路径（已创建）：

```text
tracks/depth-estimation/
├── LEARNING_ROADMAP.md          # 本蓝图
├── README.md                    # 任务地图（竞赛/数据速查）
├── catalog.json                 # 机器可读索引
│
├── 00-map/                      # 概念卡、术语表、指标定义
├── 01-geometry/                 # P1：相机与立体最小实现
├── 02-supervised-classic/       # P2：Eigen 思想 + 手写监督
├── 03-self-supervised/          # P3：Monodepth2 源码精读与迷你复现
├── 04-foundation/               # P4：MiDaS / DPT / Depth Anything
├── 05-metric/                   # P5：Depth Pro / Metric3D / UniDepth
├── 06-frontier/                 # P6–P7：新论文跟踪、失败 case
├── 07-research-lab/             # 个人假设、实验日志、论文 claim 审查
│
├── papers/                      # 阅读笔记（不存 PDF 大文件）
│   └── TEMPLATE.md
├── scripts/                     # metrics、对齐、可视化、下载
├── tests/                       # 指标与几何单元测试（防自欺）
├── data/                        # 本地缓存（gitignore）
├── vendor/                      # git submodule / clone 官方源码（gitignore 权重）
└── results/                     # 图、表、checkpoint 指针（大文件不入库）
```

### 4.1 Notebook 命名

```text
de-00-metrics-and-alignment.ipynb
de-01-pinhole-and-stereo-depth.ipynb
de-02-nyu-unet-supervised.ipynb
de-03-monodepth2-source-and-mini.ipynb
de-04-dav2-midas-zero-shot.ipynb
de-05-metric-sota-protocol.ipynb
de-06-ethz-cil-or-mdec-style.ipynb
de-rNN-<hypothesis-slug>.ipynb    # 研究假设实验
```

### 4.2 源码阅读约定（强制）

| 库 | 必读入口（实现时锁定具体文件） |
|----|--------------------------------|
| monodepth2 | `layers.py`（投影）、`datasets/`、`trainer.py` 损失 |
| MiDaS / DPT | 网络 forward、transform、inverse depth |
| Depth Anything V2 | 推理 API、metric fine-tune 入口 |
| Depth Pro | 多尺度推理、焦距、边界相关模块 |
| Metric3D / UniDepth | 相机条件化 / canonical space |

阅读产出：`papers/<paper>/SOURCE_MAP.md`（文件→函数→对应论文公式）。

### 4.3 Kaggle 使用策略

| 用途 | 做法 |
|------|------|
| GPU 训练/推理 | Notebook 挂 NYU 子集 / 竞赛数据；固定 `seed`、记录 commit hash |
| 模型权重 | Kaggle Models：`intel/midas`、`artemmmtry/depth-anything-v2` 等 |
| 提交闭环 | ETHZ CIL 或自建 private LB；**主评测仍用 NYU/KITTI 协议** |
| 对照 notebook | 高票仅作参考，**不以抄结果为终点** |

### 4.4 实验日志最小字段

```yaml
exp_id: de-02-a1
commit: <git sha>
data: nyu_subset_v1
model: unet_res18
loss: silog
metrics: {abs_rel: …, rmse: …, delta1: …}
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
3. **Debug**：故意注入错误（反深度、错误对齐、漏 automask），定位  
4. **独立实现**：合上参考代码写最小版  
5. **缺口回退**：任一失败 → 回到 PRE/CORE 对应节点，不跳级  

---

## 6. 刻意不做什么（保证路径最短）

- 不从“100 篇综述列表”开始  
- 不并行学 NeRF/3DGS 完整栈  
- 不收集 20 个相似 GitHub baseline  
- 不把 Kaggle 排行榜当研究贡献  
- 不跳过指标与对齐直接抄 SOTA 推理 demo  

---

## 7. 下一步（等你确认后开始教学）

1. 冻结本蓝图 v1  
2. 从 **P0：指标与对齐** 开工（`de-00` + `tests/test_metrics.py`）  
3. 同步 clone 必读源码到 `vendor/`（仅代码，权重按需）  
4. 每次阶段交付后 **阿里规范 commit + push**

确认后回复：**开始 P0** 或指定想先攻的 Phase。
