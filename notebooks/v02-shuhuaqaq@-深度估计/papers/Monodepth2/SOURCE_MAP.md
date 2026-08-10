# Monodepth2 SOURCE_MAP

Repo: `vendor/monodepth2` (https://github.com/nianticlabs/monodepth2)  
Paper: Godard et al., *Digging Into Self-Supervised Monocular Depth Estimation*, ICCV 2019

| Paper concept | File | Function / region | Notes |
|---------------|------|-------------------|-------|
| Sigmoid disp → depth | `layers.py` | `disp_to_depth` | \(d = 1/(a + b\cdot\sigma)\); bounds min/max depth |
| Pose (axis-angle, t) → 4×4 | `layers.py` | `transformation_from_parameters`, `rot_from_axisangle` | Invert for opposite frame |
| Backproject / project | `layers.py` | `BackprojectDepth`, `Project3D` (later in file) | Meshgrid + invK / K |
| Photometric = α SSIM + (1-α)L1 | `trainer.py` | `compute_reprojection_loss` | α=0.85 |
| Multi-source min reprojection | `trainer.py` | `compute_losses` | `torch.min` over sources |
| Auto-masking | `trainer.py` | `compute_losses` ~432–482 | Compare to identity warp; keep where warped better |
| Full-res multi-scale | `trainer.py` | `generate_images_pred` | Upsample disp before warp |
| Depth network | `networks/depth_decoder.py`, `resnet_encoder.py` | U-Net skips | ResNet18 encoder default |
| Pose network | `networks/pose_decoder.py` | | Video modes only |

## Local mini reimplementation

| Concept | Our code |
|---------|----------|
| SSIM + L1 photo | `scripts/photometric.py` `photometric_error` |
| Warp with pose | `scripts/photometric.py` `warp_with_pose` |
| Min reprojection | `scripts/photometric.py` `min_reprojection_loss` |
| Experiment | `scripts/run_p3_photometric.py` |

## Read verdict

Monodepth2's contribution is **loss design + training details**, not a fancy backbone. Any reimplementation that skips automask or min-reproj will look soft/ghosted on dynamic objects.
