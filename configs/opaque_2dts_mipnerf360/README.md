# 2DTS MipNeRF-360 opaque-triangle experiment

This config collection is an isolated experiment set for 2DTS on MipNeRF-360.

- It keeps the default `configs/2dts/mipnerf360/*.yaml` configs unchanged.
- It redirects outputs to `outputs/opaque_2dts_mipnerf360/...`.
- It uses a dedicated native 2DTS config that applies the DTU-style late-stage opacity solidification strategy.

Run the full dataset with:

```bash
bash single_train.sh 2dts mipnerf360/all 0 --config_root configs/opaque_2dts_mipnerf360
```

Summarize metrics with:

```bash
python3 tools/format_metrics.py --dataset mipnerf360/all --method 2dts --outputs-root outputs/opaque_2dts_mipnerf360
```
