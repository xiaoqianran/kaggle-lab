# Claim reviews (P6)

## Eigen et al. 2014 — Multi-Scale Depth

| | |
|--|--|
| **Claim** | Multi-scale CNN can predict dense depth from one image better than hand features |
| **Evidence** | NYU/KITTI tables; coarse-to-fine stacks |
| **Protocol DNA** | scale-invariant loss; later Eigen split |
| **Gap** | No zero-shot story; small data era |
| **Reuse** | Always cite for metrics ancestry |

## Monodepth2 2019

| | |
|--|--|
| **Claim** | Simple model + min-reproj + automask + full-res multi-scale beats complex self-sup stacks |
| **Evidence** | KITTI Eigen; ablations per component |
| **Gap** | Metric scale needs stereo/post; dynamic objects still hard |
| **Reuse** | Default self-sup baseline; **read trainer.py** |

## Depth Anything V2 2024

| | |
|--|--|
| **Claim** | Large-scale pseudo-labeling recipe yields robust relative foundation models; metric via fine-tune |
| **Evidence** | Broad zero-shot relative; student sizes S–G |
| **Gap** | Relative ≠ metric; data engine not fully open |
| **Reuse** | Default relative backbone 2024–25 |

## Depth Pro 2025 (ICLR)

| | |
|--|--|
| **Claim** | Zero-shot **metric** depth with sharp boundaries in <1s; multi-scale ViT + real/synth mix + focal |
| **Evidence** | Boundary metrics + metric tables vs DA-V2 / Metric3D / Marigold |
| **Gap** | Weight/license constraints; still camera-domain issues |
| **Reuse** | Metric SOTA reference; study focal + boundary eval |

## How we use this

P7 hypotheses should attack a **gap** row, not re-claim SOTA on synthetic planes.
