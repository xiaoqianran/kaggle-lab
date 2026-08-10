# v02 · shuhuaqaq@ · 深度估计

本目录存放 **深度估计（Depth Estimation）** 相关 notebook 与实验笔记。  
账号侧 Kaggle 用户：`shuhuaqqq`。之后本主题的 notebook 都放到这里。

调研时间：2026-08-10（CLI + 竞赛详情 API）。

---

## 一、Kaggle 上与「深度估计」直接相关的任务

### 1. 竞赛（Competitions）— 真正对口

| 竞赛 | 类型 | 截止 | 队伍 | 指标 | 说明 |
|------|------|------|------|------|------|
| [ETHZ CIL Monocular Depth Estimation 2025](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025) | Community / Kudos | 2025-05-30（已结束） | 24 | Scale Invariant RMSE | **主推练习场**：日常场景 RGB → 像素级深度；host: CIL Lecture |
| 其他含 “depth” 关键词的竞赛 | — | — | — | — | 多为地质/地震/手势等 **误匹配**，不是 monocular depth |

**竞赛要点（ETHZ CIL 2025）**

- slug：`ethz-cil-monocular-depth-estimation-2025`
- 任务：从日常场景 RGB 预测逐像素深度
- 评测：Scale Invariant RMSE（SI-RMSE）
- 每日提交上限：8；队伍上限：4
- 数据含 `create_prediction_csv.py` + `test/test/test_XXXXXX_rgb.png` 等
- 官方示例 notebook：[cilabeth/monocular-depth-example-notebook](https://www.kaggle.com/code/cilabeth/monocular-depth-example-notebook)

> 目前 **没有进行中的 Featured 大奖 monocular depth 竞赛**；社区课设/课程竞赛是主要练习入口。

### 2. 高票 / 可复现 Notebooks（学习路径参考）

| Notebook | 主题 | 票数约 |
|----------|------|--------|
| [shreydan/monocular-depth-estimation-nyuv2](https://www.kaggle.com/code/shreydan/monocular-depth-estimation-nyuv2) | NYUv2 单目深度 | 157 |
| [amanattheedge/depth-anything-v2-metric-fine-tunning-on-nyu](https://www.kaggle.com/code/amanattheedge/depth-anything-v2-metric-fine-tunning-on-nyu) | Depth Anything V2 度量微调 | 155 |
| [muhammadhafil/monocular-depth-estimation](https://www.kaggle.com/code/muhammadhafil/monocular-depth-estimation) | 单目深度入门 | 106 |
| [greg115/pix2pix-depth-estimation](https://www.kaggle.com/code/greg115/pix2pix-depth-estimation) | Pix2Pix 深度 | 81 |
| [sohaibanwaar1203/depth-net-image-depth-estimation](https://www.kaggle.com/code/sohaibanwaar1203/depth-net-image-depth-estimation) | Depth-net | 51 |
| [harits/indoor-depth-estimation-u-net](https://www.kaggle.com/code/harits/indoor-depth-estimation-u-net) | 室内 U-Net | 45 |
| [thomasrochefort/how-to-offline-depth-estimation-with-midas-dpt](https://www.kaggle.com/code/thomasrochefort/how-to-offline-depth-estimation-with-midas-dpt) | 离线 MiDaS/DPT | 23 |
| [stpeteishii/image-depth-estimation-in-kitti-data](https://www.kaggle.com/code/stpeteishii/image-depth-estimation-in-kitti-data) | KITTI | 21 |
| [cilabeth/monocular-depth-example-notebook](https://www.kaggle.com/code/cilabeth/monocular-depth-example-notebook) | ETHZ 竞赛官方示例 | 20 |
| [patiencechewyeecheah/monocular-depth-estimation-with-depth-pro](https://www.kaggle.com/code/patiencechewyeecheah/monocular-depth-estimation-with-depth-pro) | Depth Pro | 10 |

### 3. 数据集（Datasets）

| 数据集 ref | 标题 | 备注 |
|------------|------|------|
| [goubeast/nyuv2-depth](https://www.kaggle.com/datasets/goubeast/nyuv2-depth) | NYU-v2 Depth | 经典室内深度（体积很小，可能是索引/子集） |
| [polinastepanenko/nyudepthv2](https://www.kaggle.com/datasets/polinastepanenko/nyudepthv2) | Nyudepthv2 | 另一 NYU 镜像 |
| [prathamgrover/depth-estimation](https://www.kaggle.com/datasets/prathamgrover/depth-estimation) | Depth Estimation | 体积极小，需核验可用性 |
| [uditkumarrana/carla-depth-estimation-dataset](https://www.kaggle.com/datasets/uditkumarrana/carla-depth-estimation-dataset) | CARLA Depth | 仿真驾驶，体积很大（~30GB+） |
| [uditkumarrana/carla-depth-estimation-fine-tune-1080p](https://www.kaggle.com/datasets/uditkumarrana/carla-depth-estimation-fine-tune-1080p) | CARLA Fine-tune 1080p | 更大（~36GB） |

> CLI 搜 `kitti depth` 当前几乎无结果；KITTI 更多出现在 notebook 代码里（外链或自定义数据）。

### 4. Models Hub（可直接挂载推理）

| Model ref | 说明 |
|-----------|------|
| [intel/midas](https://www.kaggle.com/models/intel/midas) | 经典单目深度（相对深度） |
| [intel/midas-3.0](https://www.kaggle.com/models/intel/midas-3.0) | MiDaS 3.0 / DPT 系列 |
| [artemmmtry/depth-anything-v2](https://www.kaggle.com/models/artemmmtry/depth-anything-v2) | Depth Anything V2 |
| [artemmmtry/depth-anything-v1](https://www.kaggle.com/models/artemmmtry/depth-anything-v1) | Depth Anything V1 |
| [keras/depth-anything](https://www.kaggle.com/models/keras/depth-anything) | Keras 版 Depth Anything |
| [tensorflow/ar-portrait-depth](https://www.kaggle.com/models/tensorflow/ar-portrait-depth) | 人像深度 |
| [mediapipe/fastdepth](https://www.kaggle.com/models/mediapipe/fastdepth) | 室内快速深度 |
| [yeoyunsianggeremie/depth-anything-3](https://www.kaggle.com/models/yeoyunsianggeremie/depth-anything-3) | Depth Anything 3 镜像 |

---

## 二、建议的学习 / 实验路线（本目录后续 notebook 命名）

| 阶段 | 建议文件名前缀 | 内容 |
|------|----------------|------|
| 00 | `de-00-survey.ipynb` | 本调研摘要 + 环境冒烟 |
| 01 | `de-01-midas-infer.ipynb` | Hub 上 MiDaS 单图推理 + 可视化 |
| 02 | `de-02-depth-anything-v2.ipynb` | Depth Anything V2 推理 / 对比 |
| 03 | `de-03-nyuv2-train-lite.ipynb` | NYU-v2 小规模训练（U-Net / encoder-decoder） |
| 04 | `de-04-ethz-cil-baseline.ipynb` | 复现 ETHZ 竞赛 baseline 与 SI-RMSE |
| 05+ | 指标/蒸馏/视频深度等 | 按兴趣扩展 |

命名约定：`de-NN-slug.ipynb`（depth estimation 缩写）。

---

## 三、本目录文件

| 文件 | 说明 |
|------|------|
| `README.md` | 本文件：任务地图 + 路线 |
| （后续 `.ipynb`） | 所有深度估计实验 notebook |

---

## 四、常用命令

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
# 竞赛数据（体积视数据而定）
kaggle competitions download -c ethz-cil-monocular-depth-estimation-2025 -p ./data/ethz-cil
# 搜索
kaggle competitions list -s "depth"
kaggle datasets list -s "depth estimation"
kaggle kernels list -s "monocular depth"
kaggle models list -s "depth anything"
```
