# Dataset Setup & Directory Structure Guide

`MeshSplatBench` (MeshSplatBench) benchmarks triangle-based neural rendering methods across four standard novel view synthesis datasets:
1. **Mip-NeRF 360** (Real-world unbounded 360° indoor/outdoor scenes)
2. **NeRF-Synthetic / Blender** (Synthetic 360° bounded objects)
3. **Tanks and Temples** (Real-world forward-facing / large-scale scenes)
4. **DTU** (Multi-view indoor object scans with calibrated point clouds)

---

## 1. Directory Tree

You can either place the uncompressed datasets directly under `data/`, or store them on a separate storage drive and symlink them into `data/`. The directory structure expected by the benchmark configs is as follows:

```text
msbench/
└── data/
    ├── mipnerf360/
    │   ├── bicycle/
    │   │   ├── images/          (or images_2, images_4, images_8)
    │   │   ├── sparse/0/        (COLMAP reconstruction: cameras.bin, images.bin, points3D.bin)
    │   │   └── poses_bounds.npy
    │   ├── bonsai/
    │   ├── counter/
    │   ├── flowers/
    │   ├── garden/
    │   ├── kitchen/
    │   ├── room/
    │   ├── stump/
    │   └── treehill/
    │
    ├── nerf_synthetic/
    │   ├── chair/
    │   │   ├── train/
    │   │   ├── val/
    │   │   ├── test/
    │   │   ├── transforms_train.json
    │   │   ├── transforms_val.json
    │   │   └── transforms_test.json
    │   ├── drums/
    │   ├── ficus/
    │   ├── hotdog/
    │   ├── lego/
    │   ├── materials/
    │   ├── mic/
    │   └── ship/
    │
    ├── tandt/
    │   ├── train/
    │   │   ├── images/
    │   │   └── sparse/0/
    │   └── truck/
    │       ├── images/
    │       └── sparse/0/
    │
    └── dtu/
        ├── scan24/
        │   ├── image/
        │   └── cameras.npz
        ├── scan37/
        ├── scan40/
        ├── scan55/
        ├── scan63/
        ├── scan65/
        ├── scan69/
        ├── scan83/
        ├── scan97/
        ├── scan105/
        ├── scan106/
        ├── scan110/
        ├── scan114/
        ├── scan118/
        └── scan122/
```

---

## 2. Dataset Download Links

### 1. Mip-NeRF 360
- **Official Site**: [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)
- Download `bicycle`, `bonsai`, `counter`, `flowers`, `garden`, `kitchen`, `room`, `stump`, `treehill`.
- Extract each scene archive into `data/mipnerf360/<scene_name>`.

### 2. NeRF-Synthetic (Blender)
- **Official Source**: [Google Drive (Mildenhall et al.)](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp832luTUBIKgK3vK)
- Download `nerf_synthetic.zip`.
- Extract into `data/nerf_synthetic/`.

### 3. Tanks and Temples
- **Official Source**: [Tanks and Temples](https://www.tanksandtemples.org/)
- Download the COLMAP-processed sequences for `train` and `truck` (as prepared in standard 3DGS/Gaussian Splatting releases).
- Place under `data/tandt/<scene_name>`.

### 4. DTU
- **Download**: [DTU Dataset (PixelNeRF / MVSNet processed)](https://github.com/sxyu/pixel-nerf)
- Download the test scans: `scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65`, `scan69`, `scan83`, `scan97`, `scan105`, `scan106`, `scan110`, `scan114`, `scan118`, `scan122`.
- Place under `data/dtu/scan<id>`.

---

## 3. Using Symlinks (Recommended)

If your datasets are already stored on a shared data disk (e.g. `/data/datasets/...`), create symlinks:

```bash
mkdir -p data
ln -s /path/to/MipNeRF360 data/mipnerf360
ln -s /path/to/nerf_synthetic data/nerf_synthetic
ln -s /path/to/tandt data/tandt
ln -s /path/to/DTU data/dtu
```
Once linked, all configs under `configs/{method}/{dataset}/{scene}.yaml` will resolve correctly.
