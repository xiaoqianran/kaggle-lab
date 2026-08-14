# DiffusionGemma 26B-A4B-it · Kaggle T4×2

在 Kaggle **2× Tesla T4** 上跑通官方 `google/diffusiongemma`（Transformers，约 52 GB BF16）。

在线 notebook：https://www.kaggle.com/code/yaoyunqqq/diffusiongemma-dual-t4  
账号：`yaoyunqqq` · kernel v4 **COMPLETE**

## 怎么跑

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle kernels push -p notebooks/v05-yaoyunqqq-diffusiongemma-t4x2 \
  --accelerator NvidiaTeslaT4 \
  --timeout 14400
kaggle kernels status yaoyunqqq/diffusiongemma-dual-t4
kaggle kernels output yaoyunqqq/diffusiongemma-dual-t4 -p /tmp/dg-out
```

`kernel-metadata.json` 已挂官方模型 `google/diffusiongemma/transformers/diffusiongemma-26b-a4b-it/1`。

## 已跑通的结果（v4）

| 项 | 值 |
| --- | --- |
| GPU | 2× Tesla T4（各 14.56 GiB） |
| 策略 | FP16 `device_map=auto` + CPU offload |
| 切分 | cuda:0 6.45B / 12.0 GiB；cuda:1 7.34B / 13.7 GiB；meta 23.5B |
| 加载 | 58 s |
| 生成 | 70.5 s |

NF4 无法 GPU-only 塞进 2×14.56 GiB（会溢到 CPU，bitsandbytes 拒绝）。能稳定出字的是两张 T4 切 FP16。

样例回答见 `sample_generation.txt`，完整 JSON 见 `sample_results.json`。
