# FS14 Research Closed Loop Report

## Claim card
- **Claim:** Boundary-weighted training reduces AbsRel more than width 8→12 at fixed steps.
- **Supported (this repo run):** True
- **Threats:** Synthetic easy data, CPU short budget, No NYU/KITTI confirmation
- **Next:** Repeat H1 on NYU subset with fixed FLOPs and edge IoU metrics.

## Ablation table (from chain evidence)
```json
[
  {
    "method": "FS04 tiny monocular",
    "abs_rel_in_domain": 0.03627622645018336,
    "abs_rel_shift": 0.41309624044030596
  },
  {
    "method": "UNet skips",
    "abs_rel": 0.01166393095868399,
    "boundary": 0.02195672473559777
  },
  {
    "method": "No skips",
    "abs_rel": 0.07015224476028659,
    "boundary": 0.20046260207891464
  },
  {
    "method": "H1 edge vs capacity",
    "supported": true,
    "edge_gain": -0.022838827536228012,
    "cap_gain": -0.042729073131275885
  }
]
```

## Capability ladder (what we can do now)
1. Evaluate depth honestly (alignment protocols).
2. Recover depth from stereo matching and see texture failure.
3. Train monocular nets; measure domain shift.
4. Use photometric self-supervision signals.
5. Distill relative teachers; demand metric tape tests.
6. Write claim cards with threats to validity.
