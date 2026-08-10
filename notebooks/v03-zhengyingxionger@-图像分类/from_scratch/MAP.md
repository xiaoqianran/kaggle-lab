# 图像分类 · From Scratch 实验地图

> 类比 **LLM from scratch**：不是零散 Demo，而是一条  
> **最小可运行 → 经典方法 → 关键突破 → 现代系统 → 前沿方法** 的阶梯。  
> 每一步固定五件套：**概念 → 最小实现 → 真实输入 → 可观察输出 → 与上一步对比**。

**领域：** 图像分类（Image Classification）  
**默认数据：** FashionMNIST → 32×32 RGB（沙箱可复现；有 GPU 可换 CIFAR-10/ImageNet）  
**入口：** `python from_scratch/run_ladder.py` 或 `python from_scratch/run_ladder.py S00 S04 S08`  
**结果：** `from_scratch/results/ladder.json` + 逐步 `Sxx_*.json`

---

## 0. 如何读这张地图

| 符号 | 含义 |
|------|------|
| **Sxx** | 阶梯编号，必须按序理解；运行可跳步但对比会变弱 |
| **解决的问题** | 相对上一步，这项技术“修了什么病” |
| **最小实现** | 本仓库路径；优先手写，不先调 timm 黑盒 |
| **观测** | 你必须亲眼看到的数字/图（否则等于没做） |
| **Δ** | 相对上一步的变化（准确率、损失曲线、特征可视化） |

```text
S00 像素即数据
 │
S01 最近邻（无学习）
 │
S02 线性 Softmax（有学习，无结构）
 │
S03 MLP（深度，仍无视空间）
 │
S04 手写卷积（看见局部模式）
 │
S05 浅层 CNN / LeNet 式
 │
S06 BN + 稳定训练
 │
S07 更深的 3×3 堆叠（VGG 直觉）
 │
S08 残差（ResNet 关键突破）──── 深度终于可优化
 │
S09 深度可分卷积（效率轴）
 │
S10 现代训练配方（数据/损失/LR）
 │
S11 通道注意力 SE（轻量注意）
 │
S12 Vision Transformer（全局注意）
 │
S13 ConvNeXt（CNN 现代化）
 │
S14 对比自监督 SimCLR
 │
S15 原型/CLIP 式开放词汇直觉
 │
S16 迁移：冻表征 + 线性头 / 微调
 │
S17+ 前沿阅读与复现（MAE / SigLIP / 缩放与数据质量）
```

---

## 1. 全阶梯表（核心地图）

### 阶段 A · 从零：数据与“会学习”之前

#### S00 · 像素即张量
| | |
|--|--|
| **概念** | 图像 = \(H\times W\times C\) 数组；标签 = 离散类；归一化改变优化景观 |
| **解决的问题** | 还没有模型；先建立“模型吃什么” |
| **最小实现** | `from_scratch/steps/s00_pixels.py` |
| **真实输入** | FashionMNIST 原图（灰度）→ resize 32、复制 3 通道 |
| **可观察输出** | shape、mean/std、每类 1 张网格图、标签分布直方图 |
| **Δ vs 上步** | （起点）无 |

#### S01 · 最近邻 1-NN / k-NN（无参数基线）
| | |
|--|--|
| **概念** | 预测 = 训练集中最相似样本的标签；距离 = L2 / cosine |
| **解决的问题** | “完全不学”的下限；后面任何模型必须显著超过它 |
| **最小实现** | `s01_knn.py`（或复用 `scripts/linear_models.KNNClassifier`） |
| **真实输入** | 展平像素 / 小随机投影特征 |
| **可观察输出** | test top1；错误对：query 与 nearest 拼图 |
| **Δ** | vs 随机 10%：kNN 通常远高 → **数据本身有可分结构** |

#### S02 · 线性 Softmax（从零实现）
| | |
|--|--|
| **概念** | \(z=Wx+b\)，\(p=\mathrm{softmax}(z)\)，CE 损失 + SGD |
| **解决的问题** | 参数学习；决策边界是超平面 |
| **最小实现** | `scripts/linear_models.SoftmaxClassifier` + `s02_linear.py` |
| **真实输入** | 标准化展平像素 |
| **可观察输出** | loss 下降曲线、W 可视化为“类模板图”、top1/macro-F1/ECE |
| **Δ** | vs S01：可泛化、可校准；高维像素上仍弱 → **需要结构或更好特征** |

#### S03 · MLP（深度但无视空间）
| | |
|--|--|
| **概念** | 多层 ReLU MLP；万能逼近但仍把像素当独立特征 |
| **解决的问题** | 非线性；仍无局部性/平移归纳偏置 |
| **最小实现** | `s03_mlp.py`（2–3 隐层） |
| **真实输入** | 同 S02 展平向量 |
| **可观察输出** | top1 vs S02；过拟合曲线（train≫test 更明显） |
| **Δ** | 常略好于线性，但仍脆 → **引出卷积** |

---

### 阶段 B · 经典：卷积时代

#### S04 · 手写 2D 卷积 + 特征图
| | |
|--|--|
| **概念** | 局部连接、权值共享、滑动窗口；边缘/纹理滤波器 |
| **解决的问题** | 参数爆炸 + 无空间结构 |
| **最小实现** | `s04_conv_scratch.py`（纯 numpy/torch 循环或 `F.conv2d` 自写 kernel 初始化） |
| **真实输入** | 单张图 + 可解释 kernel（Sobel/随机可学习） |
| **可观察输出** | **特征图可视化**；参数量 vs 同等全连接 |
| **Δ** | 第一次“看见”局部模式；这是感知里程碑 |

#### S05 · 浅层 CNN（LeNet 精神）
| | |
|--|--|
| **概念** | Conv→Pool→Conv→Pool→FC；端到端分类 |
| **解决的问题** | 把卷积接到完整分类闭环 |
| **最小实现** | `scripts.models.MiniCNN` / `s05_lenet.py` |
| **真实输入** | 完整 train loader |
| **可观察输出** | top1、混淆矩阵；中间层激活 |
| **Δ** | vs S03：**同参量级下显著涨点** → 归纳偏置的胜利 |

#### S06 · BatchNorm + 稳定优化
| | |
|--|--|
| **概念** | 批量标准化；更稳的梯度与学习率容忍度 |
| **解决的问题** | 深一点就难训 / 对 lr 过敏 |
| **最小实现** | 同架构去 BN vs 加 BN 对照 `s06_bn_ablation.py` |
| **真实输入** | 固定 seed、相同 lr |
| **可观察输出** | 两条 loss 曲线；最终 top1 |
| **Δ** | BN 版收敛更快/更稳 → **训练技术开始与架构同等重要** |

#### S07 · 3×3 堆叠加深（VGG 直觉）
| | |
|--|--|
| **概念** | 小卷积核堆叠 ≈ 大感受野 + 更多非线性 |
| **解决的问题** | 感受野与深度的经典配方 |
| **最小实现** | `s07_vggish.py`（多层 3×3） |
| **真实输入** | 同数据预算 |
| **可观察输出** | 深度 3/5/7 的 trainability；退化现象预告 |
| **Δ** | 更深不一定更好 → **为残差铺垫** |

---

### 阶段 C · 关键突破：深度可训练

#### S08 · 残差连接（ResNet）
| | |
|--|--|
| **概念** | \(y=F(x)+x\)；学习残差而非完整映射 |
| **解决的问题** | 深度网络退化（更深 train error 更高） |
| **最小实现** | `scripts.models.ResNetCIFAR` + `s08_resnet.py`：plain deep vs residual |
| **真实输入** | 相同深度/通道 |
| **可观察输出** | **train loss 是否仍下降**；test top1 |
| **Δ** | 残差可继续优化 → **现代视觉主干的分水岭** |

#### S09 · 深度可分卷积（MobileNet 思想）
| | |
|--|--|
| **概念** | DWConv + PWConv；参数/FLOPs 效率 |
| **解决的问题** | 算力/部署约束 |
| **最小实现** | `s09_depthwise.py` |
| **真实输入** | 同 top1 目标下比参数量 |
| **可观察输出** | params、一步 forward ms、top1 |
| **Δ** | 精度–效率 Pareto；研究也要报告成本 |

---

### 阶段 D · 现代系统：配方 + 双路线架构

#### S10 · 训练配方（timm 精神）
| | |
|--|--|
| **概念** | cosine LR、label smoothing、Mixup、强增强、Random Erasing |
| **解决的问题** | 同架构不同 recipe 可差几个点 |
| **最小实现** | `run_phase.py P4` / `s10_recipe.py` |
| **真实输入** | 固定 ResNet/MiniCNN，4 组消融 |
| **可观察输出** | 消融表 top1 + ECE |
| **Δ** | 最好 recipe vs base → **现代 SOTA 大半来自配方** |

#### S11 · SE 通道注意力
| | |
|--|--|
| **概念** | 全局池化 → 两层 FC → 通道重标定 |
| **解决的问题** | 通道“哪个特征图重要” |
| **最小实现** | `s11_se.py`：CNN ± SE |
| **真实输入** | 同预算 |
| **可观察输出** | top1；SE 权重在各类上的平均响应（可选） |
| **Δ** | 轻量注意力先于 ViT 的实用形态 |

#### S12 · Vision Transformer（最小 ViT）
| | |
|--|--|
| **概念** | patch embed + pos + Transformer + cls head |
| **解决的问题** | 全局依赖；与 CNN 归纳偏置对照 |
| **最小实现** | `scripts.models.MiniViT` / `s12_vit.py` |
| **真实输入** | 小数据（故意）+ 可选更长训 |
| **可观察输出** | top1 vs ResNet；注意力图（可选） |
| **Δ** | 小数据常弱于 CNN → **解释为何需要 DeiT/预训练** |

#### S13 · ConvNeXt（把 Transformer 训练课搬回 CNN）
| | |
|--|--|
| **概念** | 大核 DWConv、LN、少激活、阶段设计 |
| **解决的问题** | “Transformer 是否本质必要？” |
| **最小实现** | `scripts.models.MiniConvNeXt` / `s13_convnext.py` |
| **真实输入** | 与 S12 同数据预算 |
| **可观察输出** | top1；与 ViT 对照表 |
| **Δ** | 现代 CNN 可与 ViT 同台 → **架构双路线** |

---

### 阶段 E · 前沿底座：自监督与多模态

#### S14 · 对比学习 SimCLR（NT-Xent）
| | |
|--|--|
| **概念** | 两视图正样本；batch 内负样本；投影头 |
| **解决的问题** | 无标签时如何学可迁移表征 |
| **最小实现** | `run_phase.py P5` / `s14_simclr.py` |
| **真实输入** | 无标签增强双视图 |
| **可观察输出** | SSL loss 曲线；**linear probe top1** |
| **Δ** | vs 随机初始化 probe → 表征质量可视化 |

#### S15 · 原型余弦 ≈ CLIP 开放词汇直觉
| | |
|--|--|
| **概念** | 图特征与类原型/文本嵌入对齐；零样本分类 |
| **解决的问题** | 闭集头无法扩展新类 |
| **最小实现** | P5 prototype 分支 / `s15_prototype_clip.py` |
| **真实输入** | 类均值原型（文本塔可选：随机/简单 bag-of-words 嵌入作教学） |
| **可观察输出** | zero-shot-ish top1 vs 线性 probe |
| **Δ** | 开放集接口；精度通常低于闭集微调 |

#### S16 · 迁移学习协议
| | |
|--|--|
| **概念** | linear probe / partial FT / full FT；差分学习率 |
| **解决的问题** | 目标域数据少时如何用预训练 |
| **最小实现** | `s16_transfer.py`：冻 backbone 训头 vs 全开 |
| **真实输入** | 少样本子集（每类 N=50/200） |
| **可观察输出** | 三种协议 top1 表 |
| **Δ** | 少样本下 probe 可赢乱训全网络 |

#### S17+ · 前沿（读原文 + 复现实验，不强行 CPU 全训）
| ID | 方法 | 你要亲手做的最小动作 |
|----|------|----------------------|
| S17 | **MAE** | 读论文；实现 patch mask + 重建 loss 的 **toy 版**（可 1 epoch 过拟合一张图） |
| S18 | **DeiT 配方** | 对照：ViT + 强增强/更长训 vs 裸训 |
| S19 | **CLIP 真模型** | 加载开源 CLIP，zero-shot Fashion/CIFAR 类名；与 S15 对比 |
| S20 | **现代评测** | Top-1 + macro-F1 + ECE + 简单 OOD（Gaussian 噪声）一套协议 |

---

## 2. 与历史时间线的对齐（帮助建立“为什么现在这样”）

| 年代直觉 | 历史代表 | 本阶梯 |
|----------|----------|--------|
| 前深度学习 | kNN / 线性 / 手工特征 | S01–S02（SIFT/HOG 可跳过，除非做历史专题） |
| 2012 | AlexNet | S05 + ReLU/Dropout 直觉 |
| 2014 | VGG | S07 |
| 2015 | ResNet | **S08**（必做） |
| 2017 | MobileNet | S09 |
| 2018 | SENet | S11 |
| 2020 | SimCLR / 对比学习爆发 | **S14** |
| 2020–21 | ViT / DeiT | **S12** |
| 2021 | CLIP | **S15–S19** |
| 2022 | ConvNeXt / MAE | **S13 / S17** |
| 2023+ | SigLIP、数据质量、高效微调 | S20 + 论文追踪 |

---

## 3. 每一步的“验收清单”（强制）

做完 Sxx 必须能回答：

1. **一句话：** 这一步解决了上一步的什么失败模式？  
2. **一个数：** 主指标相对 S(xx-1) 的 Δ（升/降/持平）  
3. **一个图：** loss 曲线 **或** 特征/卷积/注意力可视化 **或** 错误样例  
4. **一个反事实：** 若去掉该技术，预期哪项指标变差？  

写进 `from_scratch/results/Sxx_*.json` 的 `takeaway` 字段。

---

## 4. 推荐最短通关路径（2 周密度，研究向）

| 天数 | 必做阶梯 | 产出 |
|------|----------|------|
| D1 | S00–S02 | 像素、kNN、线性；建立下限 |
| D2 | S03–S05 | MLP→卷积→CNN；**特征图必看** |
| D3 | S06–S08 | BN 消融 + **残差 vs plain** |
| D4 | S09–S11 | 效率 + recipe + SE |
| D5 | S12–S13 | ViT vs ConvNeXt 同预算 |
| D6 | S14–S16 | SSL + 原型 + 迁移协议 |
| D7 | S17–S20 选 2 | MAE toy 或真 CLIP zero-shot + 完整评测协议 |

---

## 5. 与本仓库其他文档的关系

| 文档 | 角色 |
|------|------|
| [LEARNING_ROADMAP.md](../LEARNING_ROADMAP.md) | 研究能力路径、资料筛选、P0–P7 阶段 |
| **本 MAP** | **实现阶梯**（from scratch 感知每一项技术） |
| [PROGRESS.md](../PROGRESS.md) | P0–P7 执行状态 |
| `scripts/` + `run_phase.py` | 已实现的可运行后端（S02/S05/S08/S10/S12–S15 已部分覆盖） |

**原则：** MAP 管“顺序与对比”，ROADMAP 管“读什么论文/如何做研究”，PROGRESS 管“跑没跑通”。

---

## 6. 反模式（禁止）

- 一上来 `timm.create_model('resnet50', pretrained=True)` 然后调参 → **跳过了 S00–S08 的感知**  
- 只改准确率数字、不看特征图/错误样例 → 不知道技术“修了什么”  
- 无固定 seed / 无同预算对照 → Δ 无意义  
- 把检测/分割/生成全塞进阶梯 → 冲淡分类主线  

---

## 7. 一键运行

```bash
cd notebooks/v03-zhengyingxionger@-图像分类
source ../../.venv/bin/activate   # 若使用仓库 venv
python from_scratch/run_ladder.py           # 默认核心阶梯
python from_scratch/run_ladder.py S00 S01 S02 S05 S08 S12 S14
python from_scratch/run_ladder.py --list
```

每步打印：`step | metric | delta_vs_prev | takeaway`，并写入 `from_scratch/results/`。
