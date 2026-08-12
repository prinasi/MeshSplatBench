# 2DTS MipNeRF-360 opaque-triangle experiment

This config collection is an isolated experiment set for 2DTS on MipNeRF-360.

- It keeps the default `configs/2dts/mipnerf360/*.yaml` configs unchanged.
- It redirects outputs to `outputs/opaque_2dts_mipnerf360/...`.
- It uses a dedicated native 2DTS config that applies the DTU-style late-stage opacity solidification strategy.

Run the full dataset with:

```bash
bash single_train.sh 2dts mipnerf360/all 0 --config_root configs/opaque_2dts_mipnerf360
```

`tribench train --config ...` now supports short smoke tests and automatic resume for
this 2DTS setup:

- `--max-steps N` overrides the structured config's default iteration count.
- If `outputs/opaque_2dts_mipnerf360/.../ckpt/` already contains a local
  `*.ckpt` or `point_cloud/*.ply` checkpoint whose step is below `N`, training
  automatically resumes from the latest one.
- If the latest checkpoint already reached `N`, TriBench skips the remaining
  training work.

Example smoke/resume flow:

```bash
CUDA_VISIBLE_DEVICES=0 tribench train \
  --config configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml \
  --max-steps 30 --quiet

CUDA_VISIBLE_DEVICES=0 tribench train \
  --config configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml \
  --max-steps 31 --quiet
```

Summarize metrics with:

```bash
python3 tools/format_metrics.py --dataset mipnerf360/all --method 2dts --outputs-root outputs/opaque_2dts_mipnerf360
```
