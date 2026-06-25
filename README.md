<!-- @format -->

# TriBench

A Python-first developer toolkit for unified benchmarking of triangle-splatting-style radiance field methods.

## Overview

TriBench provides a single, consistent evaluation framework for comparing different triangle-based radiance field reconstruction methods. Instead of wrangling per-repo scripts and ad-hoc metrics, you get:

- **Standardized metrics** -- PSNR, SSIM, LPIPS, and geometry quality, computed the same way for every method.
- **Adapter pattern** -- plug any triangle-splatting method into the same API.
- **Reproducible experiments** -- YAML configs lock down every hyperparameter.
- **CLI-first workflow** -- inspect, profile, evaluate, and compare from the terminal.

## Supported Methods

| Method                       | Adapter                    | Status                                   |
| ---------------------------- | -------------------------- | ---------------------------------------- |
| 2D Triangle Splatting (2DTS) | `D2TSAdapter`              | Initial config/train/render/eval support |
| Triangle Splatting           | `TriangleSplattingAdapter` | Config/train/render/eval support         |
| MeshSplatting                | `MeshSplattingAdapter`     | Config/train/render/eval support         |
| DiffSoup                     | `DiffSoupAdapter`          | Planned                                  |

## Installation

```bash
git clone https://github.com/your-org/TriBench.git
cd TriBench
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train, render, evaluate, inspect, and profile from one experiment config
tribench train --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench render images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench eval images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench inspect --config configs/triangle-splatting/mipnerf360/bicycle.yaml
tribench profile --config configs/triangle-splatting/mipnerf360/bicycle.yaml

# Config values can still be overridden from the CLI
tribench render video --config configs/triangle-splatting/mipnerf360/bicycle.yaml --output-dir outputs/demo-video

# Train complete datasets with the scene pipeline helper
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0
```

## Config Inheritance

Experiment configs use `_base_` inheritance. Common defaults live in
`configs/base/_base_.yaml`, dataset-level defaults live in files such as
`configs/base/mipnerf360.yaml`, method-level shared settings live in files such
as `configs/base/triangle-splatting.yaml`, and method/dataset/scene configs
only keep the fields that are unique to that run.

Scene configs are grouped by method and dataset:

```text
configs/triangle-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/mesh-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/2dts/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
├── native/dtu.yaml
└── tandt/truck.yaml
```

For example, `configs/triangle-splatting/mipnerf360/bicycle.yaml` inherits the
MipNeRF360 dataset defaults plus Triangle Splatting defaults, then only sets the
scene and scene-specific training options:

```yaml
_base_:
  - ../../base/mipnerf360.yaml
  - ../../base/triangle-splatting.yaml

dataset:
  scene: bicycle

trainer:
  outdoor: true
```

`dataset.root` and `dataset.scene` are resolved together, so a dataset base can
declare `root: data/mipnerf360` while a scene config declares `scene: bicycle`;
the effective dataset path becomes `data/mipnerf360/bicycle`. String templates
such as `outputs/{method}/{dataset}/{scene}` are resolved after all inherited
configs are merged.

To run a whole dataset, use `single_train.sh` with a dataset target ending in
`/all`. It expands the configured scene list, trains each scene, renders/evals,
exports videos, and prints a summary table:

```bash
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0
```

Use `all` as the target to run all built-in datasets in one pass.

Triangle Splatting configs inherit scene-specific triangle count caps from
`configs/base/triangle-splatting-caps.yaml` through
`configs/base/triangle-splatting.yaml`. During config finalization, the cap for
`dataset.scene` is written to `trainer.max_shapes` unless the scene config
explicitly overrides it.

MeshSplatting configs live under `configs/mesh-splatting/` for MipNeRF360,
Tanks&Temples, and DTU. They use the bundled MeshSplatting renderer and a
method-specific trainer that preserves shared-vertex topology updates. DTU
configs follow the upstream DTU settings, including restricted Delaunay at
iteration 11000; that stage requires the MeshSplatting `effrdel` dependency to
be installed in the active environment.

## Unified Output Layout

Every method now writes to one consistent per-scene directory so training
checkpoints, rendered images, exported geometry, videos, and logs always live
in the same place regardless of backend.

### Target layout

```text
outputs/{method}/{dataset}/{scene}/
├── ckpt/                 # all training checkpoints (one folder, every method)
│   ├── 30000.ckpt                       # 2DTS
│   ├── point_cloud/{N}.ply              # 2DTS exported point clouds
│   ├── point_cloud/iteration_{N}/*.pt   # triangle-/mesh-splatting
│   └── final_params.pt                  # DiffSoup
├── renders/              # rendered images + matching ground truth, by split
│   ├── train/
│   │   ├── renders/      # predicted RGB
│   │   ├── gt/           # dataset ground truth
│   │   ├── manifest.json
│   │   └── metrics.json
│   └── test/
│       ├── renders/
│       ├── gt/
│       ├── manifest.json
│       └── metrics.json
├── mesh/                 # exported geometry used for mesh/DTU metrics
│   ├── fuse_post.ply                    # TSDF mesh (mip360/tandt/dtu)
│   └── {N}_pcd.ply                      # 2DTS DTU point-cloud export
├── video/                # trajectory video
│   ├── render_traj.mp4
│   └── frames/
├── logs/                 # every log: pipeline *.log, training stdout,
│                         # tensorboard events, cfg dumps, loss curves
├── config.yaml           # resolved config snapshot
├── metrics.json          # aggregated image metrics (test split)
├── mesh_metrics.json     # DTU/mesh geometry metrics
└── train_stats.json      # training time / peak GPU memory
```

### What changes and why

1. **Checkpoints → `ckpt/`.** Previously each method scattered checkpoints
   differently (2DTS used `point_cloud/`, `ckpt/`, `glb/`, `mesh_ply/` at the
   scene root; triangle-/mesh-splatting used `point_cloud/iteration_N/`;
   DiffSoup dropped a bare `final_params.pt`). All four are redirected under a
   single `ckpt/` folder. Each method keeps its native internal structure
   inside `ckpt/`, so adapters and resume logic keep working.
2. **Renders → `renders/{split}/`.** Rendered images and their ground truth are
   saved once, split by `train`/`test`, with `renders/` (prediction) and `gt/`
   (truth) subfolders plus a per-split `manifest.json`/`metrics.json`.
3. **Metrics reuse renders.** `eval images` no longer re-renders into a separate
   `test_metrics/` folder. It reads the already-saved `renders/{split}/renders`
   and `renders/{split}/gt` images (and reuses the per-split manifest timing),
   then writes only `metrics.json`. No duplicate images are written.
4. **Exported geometry → `mesh/`.** `mesh.ply`/`fuse_post.ply` and the 2DTS
   `{N}_pcd.ply` move under `mesh/`.
5. **Video → `video/`.** Unchanged; already unified.
6. **Logs → `logs/`.** Pipeline stage logs, native training stdout, tensorboard
   event files, config dumps, and DiffSoup loss curves are consolidated into a
   single `logs/` folder. The previous mix of `log/`, `logs/`, root-level
   `cfg_args`, and tensorboard files at the scene root is removed.

### Adapter checkpoint paths

The `adapter.checkpoint` template in each method base config points at `ckpt/`:

| Method             | `adapter.checkpoint`                                            |
| ------------------ | -------------------------------------------------------------- |
| 2DTS               | `outputs/{method}/{dataset}/{scene}/ckpt`                      |
| Triangle Splatting | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| MeshSplatting      | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| DiffSoup           | `outputs/{method}/{dataset}/{scene}/ckpt/final_params.pt`     |

### Implementation checklist

- [x] `configs/base/_base_.yaml`: renders/eval/mesh templates → unified layout.
- [x] `configs/base/dtu.yaml` + `configs/2dts/dtu/*.yaml`: `mesh/` paths.
- [x] `configs/base/{2dts,diffsoup,mesh-splatting,triangle-splatting}.yaml`:
      `adapter.checkpoint` → `ckpt/`.
- [x] `tribench/core/rendering.py`: `render_dataset_split` writes
      `renders/{split}/{renders,gt}` + per-split manifest/metrics, plus
      `compute_metrics_from_render_dir` for metrics without re-rendering.
- [x] `tribench/cli/render.py`: pass the run dir + split to the renderer.
- [x] `tribench/cli/eval.py`: compute metrics from saved renders, no re-render,
      no `test_metrics/`.
- [x] `tribench/cli/train.py`: redirect every method's checkpoint to `ckpt/`.
- [x] Vendor trainers (`triangle_splatting`, `mesh_splatting`, `d2ts`): send
      tensorboard/log output to `logs/` and 2DTS artifacts to `ckpt/`/`mesh/`.
- [x] `single_train.sh`: update completion markers for the new layout.
- [x] Tests: update render-layout expectations.


## Project Structure


```
TriBench/
├── tribench/
│   ├── core/           # Cameras, stats, registry, config
│   ├── primitives/     # Triangle primitive types (independent, mesh, convex)
│   ├── renderers/      # Adapter wrappers for each method
│   ├── trainers/       # Training loop, losses, hooks
│   ├── metrics/        # PSNR, SSIM, LPIPS, geometry metrics
│   ├── datasets/       # Scene dataset loaders
│   ├── cli/            # Typer-based CLI commands
│   └── utils/          # Logging, timing, visualization helpers
├── configs/            # YAML experiment configs
├── tests/              # pytest test suite
└── pyproject.toml
```

## Development Roadmap

The project is organized into four phases:

1. **Phase 1 -- Foundation**: Core abstractions (cameras, stats, registry), primitive types, and the first adapter (2DTS).
2. **Phase 2 -- Training & Evaluation**: Unified training loop, loss functions, full metric suite, and dataset loaders.
3. **Phase 3 -- Remaining Adapters**: Triangle Splatting and MeshSplatting adapter coverage with parity validation, followed by DiffSoup.
4. **Phase 4 -- Polish & Release**: CLI refinement, experiment configs, documentation, and public release.

### Next Development Steps

The immediate goal is to keep Triangle Splatting and MeshSplatting as trusted
reference pipelines before broadening adapter coverage further. This gives the
remaining methods a concrete standard for training, rendering, evaluation,
profiling, and configuration behavior.

1. **Make profiling functional**
   - Extend `tribench profile --config ...` beyond metadata output.
   - Report forward latency, FPS, peak CUDA memory, primitive count, checkpoint size, and optional backward latency.
   - Start with Triangle Splatting, then reuse the same profile schema for other adapters.

## License

MIT
