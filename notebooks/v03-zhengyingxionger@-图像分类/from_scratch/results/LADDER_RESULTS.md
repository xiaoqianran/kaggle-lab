# From-Scratch Ladder Results

| Step | Title | Metric | Δ top1 | Takeaway |
|------|-------|--------|--------|----------|
| S00 | pixels as tensors | None | — | Model input is a numeric tensor + discrete label; always check shape/scale first. |
| S01 | kNN baseline (no learning) | 0.7100 | — | Data has structure: kNN >> chance without any training. |
| S02 | linear softmax from scratch | 0.8063 | +0.0963 | Learning finds class templates (rows of W); still no spatial prior. |
| S03 | MLP on flat pixels | 0.8200 | +0.0137 | Nonlinearity helps some; still treats pixels as bag of values → need convolution. |
| S04 | hand convolution + feature maps | 0.5188 | — | Conv reuses local filters: orders-of-magnitude fewer params than dense spatial maps; edges emerge from simple kernels. |
| S05 | shallow CNN (LeNet spirit) | 0.6625 | -0.1575 | End-to-end CNN uses spatial structure; should beat flat MLP under similar budget. |
| S06 | BatchNorm ablation | 0.5983 | -0.0642 | BN changes trainability/lr robustness; compare curves not only final top1. |
| S08 | residual vs plain deep | 0.7300 | +0.1317 | Residual path keeps deep nets optimizable — key 2015 breakthrough. |
| S09 | depthwise separable conv | 0.4917 | -0.2383 | Factorizing conv buys efficiency; report params/latency with accuracy. |
| S10 | training recipe ablation | 0.6267 | +0.1350 | Same model, different recipe → different top1/ECE; modern gains are often recipe. |
| S11 | Squeeze-and-Excitation | 0.6283 | +0.0017 | Channel attention is a cheap pre-ViT way to reweight features. |
| S12 | minimal Vision Transformer | 0.4600 | -0.1683 | Global attention works but is data-hungry without strong recipe/pretrain. |
| S13 | Mini ConvNeXt | 0.5033 | +0.0433 | Modernized CNNs compete with ViTs; architecture is not settled on attention-only. |
| S14 | SimCLR-style contrastive SSL | 0.6160 | +0.1127 | Labels optional: contrastive pretrain + linear probe measures representation quality. |
| S15 | prototype cosine (CLIP intuition) | 0.6250 | +0.0090 | Cosine-to-prototype is the interface of open-vocab models; usually trails closed heads. |
| S16 | transfer protocols | 0.7067 | +0.0817 | With few labels, reusing pretrained features (probe/FT) beats training from scratch. |
