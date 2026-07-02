# TriBench Config Layout

Scene configs are grouped by method, then dataset:

```text
configs/
├── base/
├── 2dts/
│   ├── dtu/
│   ├── mipnerf360/
│   ├── nerf_synthetic/
│   ├── native/
│   └── tandt/
├── diffsoup/
│   ├── dtu/
│   ├── mipnerf360/
│   ├── nerf_synthetic/
│   └── tandt/
├── triangle-splatting/
    ├── dtu/
    │   └── scan24.yaml
    ├── mipnerf360/
    │   ├── bicycle.yaml
    │   └── room.yaml
    ├── nerf_synthetic/
    │   ├── lego.yaml
    │   └── chair.yaml
    └── tandt/
        ├── train.yaml
        └── truck.yaml
└── mesh-splatting/
    ├── dtu/
    │   └── scan24.yaml
    ├── mipnerf360/
    │   ├── bicycle.yaml
    │   └── room.yaml
    ├── nerf_synthetic/
    │   ├── lego.yaml
    │   └── chair.yaml
    └── tandt/
        ├── train.yaml
        └── truck.yaml
```

Each scene config inherits one dataset base plus one method base:

```yaml
_base_:
  - ../../base/mipnerf360.yaml
  - ../../base/triangle-splatting.yaml
```

Run one scene directly from its config:

```bash
tribench train --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench eval images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench train --config configs/mesh-splatting/mipnerf360/bicycle.yaml
tribench eval images --config configs/mesh-splatting/mipnerf360/bicycle.yaml
tribench train --config configs/triangle-splatting/nerf_synthetic/lego.yaml
tribench eval images --config configs/triangle-splatting/nerf_synthetic/lego.yaml
tribench train --config configs/2dts/dtu/scan24.yaml
tribench eval images --config configs/2dts/dtu/scan24.yaml
```

Train/evaluate a complete built-in dataset with the scene pipeline helper:

```bash
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting nerf_synthetic/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting nerf_synthetic/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh diffsoup nerf_synthetic/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts nerf_synthetic/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0
```

Use `all` as the target to run all built-in datasets:

```bash
bash single_train.sh triangle-splatting all 0
```
