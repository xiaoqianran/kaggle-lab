# v03 · zhengyingxionger@ · 图像分类 / 深度估计工作区

本目录用于存放后续 Kaggle notebook（图像分类与相关 CV 实验）。

> 侦察主题按用户指令：**深度估计（Depth Estimation）**。  
> 文件夹名保留用户指定：`v03-zhengyingxionger@-图像分类`。

---

## 环境（本机已配置）

| 项 | 状态 |
|---|---|
| GitHub (`gh`) | 已登录 `xiaoqianran` |
| Kaggle CLI | `kaggle-lab/.venv` → **Kaggle CLI 2.2.4** |
| Token | `~/.kaggle/access_token`（`KGAT_…`） / `KAGGLE_API_TOKEN` |
| 账号 | `zhengyingxiong`（ACCESS_TOKEN） |
| GPU 配额 | 30h GPU / 20h TPU（本周未使用，刷新 2026-08-15） |
| MCP | `https://www.kaggle.com/mcp`（Bearer `KAGGLE_API_TOKEN`） |
| Skills | 仓库内 `docs/skills/`（官方镜像） |

```bash
cd kaggle-lab
source .venv/bin/activate
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle quota
```

---

## 深度估计 · 竞赛（Kaggle）

检索词：`depth estimation` / `monocular depth`（CLI + MCP `search_competitions`）。

| 竞赛 | 类别 | 奖励 | 队伍 | 截止 | 指标 | 说明 |
|------|------|------|------|------|------|------|
| [ETHZ CIL Monocular Depth Estimation 2025](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025) | Community | Kudos | 24 | 2025-05-30 | Scale Invariant RMSE | **最直接相关**：RGB → 像素级深度；日提交 8 次，最多 4 人 |

说明：站内**仍在架且标题/描述明确为 monocular depth** 的竞赛目前以 ETHZ CIL 2025 为主（已过提交截止，仍可学习数据与 baseline）。

---

## 深度估计 · 推荐数据集

| 数据集 | 用途 | 链接 |
|--------|------|------|
| NYU Depth V2 (`soumikrakshit/nyu-depth-v2`) | 室内 RGB-D 经典基准 | https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2 |
| NYU Depth Dataset V2 (`artemmmtry/nyu-depth-v2`) | 完整 NYU-v2 | https://www.kaggle.com/datasets/artemmmtry/nyu-depth-v2 |
| KITTI Eigen Split (`awsaf49/kitti-eigen-split-dataset`) | 室外自动驾驶深度 | https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset |
| KITTI LiDAR 2D Depth (`ahmedfawzyelaraby/kitti-lidar-based-2d-depth-images`) | KITTI 深度图 | https://www.kaggle.com/datasets/ahmedfawzyelaraby/kitti-lidar-based-2d-depth-images |
| CityScapes Depth + Seg (`sakshaymahna/cityscapes-depth-and-segmentation`) | 城市场景深度/分割 | https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation |
| Image Depth Estimation (`sohaibanwaar1203/image-depth-estimation`) | 通用深度数据包 | https://www.kaggle.com/datasets/sohaibanwaar1203/image-depth-estimation |
| DDAD (`artemmmtry/ddad-dense-depth-for-autonomous-driving`) | 自动驾驶稠密深度 | https://www.kaggle.com/datasets/artemmmtry/ddad-dense-depth-for-autonomous-driving |

---

## 深度估计 · 模型（Kaggle Models）

| 模型 | 说明 |
|------|------|
| [intel/midas](https://www.kaggle.com/models/intel/midas) | MiDaS 单目深度 |
| [intel/midas-3.0](https://www.kaggle.com/models/intel/midas-3.0) | MiDaS 3.0 ViT |
| [keras/depth-anything](https://www.kaggle.com/models/keras/depth-anything) | Depth Anything |
| [artemmmtry/depth-anything-v2](https://www.kaggle.com/models/artemmmtry/depth-anything-v2) | Depth Anything V2 |
| [rtarun/depth_pro](https://www.kaggle.com/models/rtarun/depth_pro) | Apple Depth Pro |
| [mediapipe/fastdepth](https://www.kaggle.com/models/mediapipe/fastdepth) | FastDepth 轻量 |

---

## 深度估计 · 高票 Notebook（学习向）

| Notebook | 作者 | 票 |
|----------|------|-----|
| [monocular-depth-estimation - NYUv2](https://www.kaggle.com/code/shreydan/monocular-depth-estimation-nyuv2) | Shreyas Daniel Gaddam | 157 |
| [Monocular depth estimation](https://www.kaggle.com/code/muhammadhafil/monocular-depth-estimation) | Muhammad Hafil T | 106 |
| [Image Depth Estimation in KITTI Data](https://www.kaggle.com/code/stpeteishii/image-depth-estimation-in-kitti-data) | stpete_ishii | 21 |
| [Monocular Depth Example Notebook](https://www.kaggle.com/code/cilabeth/monocular-depth-example-notebook) | CIL Lecture（ETHZ 竞赛方） | 20 |
| [Depth Anything V2](https://www.kaggle.com/code/patiencechewyeecheah/depth-anything-v2) | Patience Chew | 17 |
| [Monocular Depth Estimation: Guide For Beginners](https://www.kaggle.com/code/patiencechewyeecheah/monocular-depth-estimation-guide-for-beginners) | Patience Chew | 12 |

---

## 图像分类 · 可练习竞赛（文件夹主题）

若后续 notebook 走「图像分类」主线，可优先这些（含仍开放/近期）：

| 竞赛 | 奖金 | 截止 |
|------|------|------|
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | $77k | 2026-10-22 |
| [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) | $50k | 2025-10-14 |
| [ISIC 2024](https://www.kaggle.com/competitions/isic-2024-challenge) | $80k | 已结束，学习用 |
| [Human Protein Atlas Image Classification](https://www.kaggle.com/competitions/human-protein-atlas-image-classification) | 经典多标签 | 已结束 |

经典入门数据集：`moltean/fruits`、`zalando-research/fashionmnist`、`puneet6060/intel-image-classification`。

---

## 本目录约定

- 新 notebook 直接放在此文件夹下，命名建议：`cls-001-….ipynb` / `depth-001-….ipynb`
- 大文件、`.venv`、密钥不入库（见仓库根 `.gitignore`）
- 在 Kaggle 上跑时选 T4/P100，注意周 GPU 配额

