# P4 · recipe ablation

| exp | aug | ls | mixup | top1 | ece |
|-----|-----|----|-------|------|-----|
| p4_base | False | 0.0 | 0.0 | 0.6633 | 0.1658 |
| p4_strong_aug | True | 0.0 | 0.0 | 0.6433 | 0.1663 |
| p4_ls | True | 0.1 | 0.0 | 0.6450 | 0.2201 |
| p4_mixup | True | 0.0 | 0.2 | 0.6500 | 0.1782 |

Best: p4_base
