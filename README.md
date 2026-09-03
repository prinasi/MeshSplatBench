<!-- @format -->

# MeshSplatBench: A Unified Benchmark for Triangle-Based Neural Rendering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01306-b31b1b.svg)](https://arxiv.org/abs/2609.01306)

**MeshSplatBench** (`tribench`) is a unified, Python-first benchmarking toolkit and evaluation framework for comparing triangle-splatting-style radiance field reconstruction methods, bridging academic novel view synthesis research and real-time industrial game engine deployment (Unity).

## Overview

MeshSplatBench provides a single, consistent evaluation framework for comparing different triangle-based radiance field reconstruction methods. Instead of wrangling per-repo scripts and ad-hoc metrics, you get:

- **Standardized metrics** -- PSNR, SSIM, LPIPS, and geometry quality, computed the same way for every method.
- **Adapter pattern** -- plug any triangle-splatting method into the same API.
- **Reproducible experiments** -- YAML configs lock down every hyperparameter.
- **CLI-first workflow** -- inspect, profile, evaluate, and compare from the terminal.
- **Industrial Engine Parity** -- export method-preserving `.triasset` packages and evaluate real-time framerates and rendering fidelity natively in Unity.

## Supported Methods

| Method                       | Adapter                    | Status                                   |
| ---------------------------- | -------------------------- | ---------------------------------------- |
| 2D Triangle Splatting (2DTS) | `D2TSAdapter`              | Config/train/render/eval/export support  |
| Triangle Splatting           | `TriangleSplattingAdapter` | Config/train/render/eval/export support  |
| MeshSplatting                | `MeshSplattingAdapter`     | Config/train/render/eval/export support  |
| DiffSoup                     | `DiffSoupAdapter`          | Config/train/render/eval/export support; In-shader Micro-MLP support |

## Installation

### Option 1: Conda Environment (Recommended)

```bash
git clone https://github.com/your-org/tribench.git
cd tribench
conda env create -f environment.yml
conda activate tribench
```

### Option 2: Pip Install

```bash
git clone https://github.com/your-org/tribench.git
cd tribench
pip install -e ".[dev,cuda]"
```

## Dataset Setup

MeshSplatBench evaluates methods on standard novel view synthesis benchmarks: **Mip-NeRF 360**, **NeRF-Synthetic (Blender)**, **Tanks & Temples**, and **DTU**.

Readers should download the datasets from their official sources and place or symlink them into the `data/` directory according to the default format:

```bash
mkdir -p data
ln -s /path/to/MipNeRF360 data/mipnerf360
ln -s /path/to/nerf_synthetic data/nerf_synthetic
ln -s /path/to/tandt data/tandt
ln -s /path/to/DTU data/dtu
```

Detailed directory trees, download links, and COLMAP structures are documented in [data/README.md](data/README.md).

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

# Structured training configs also honor --max-steps, and 2DTS automatically
# resumes from local checkpoints when the latest step is still below the target.
tribench train --config configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml --max-steps 30 --quiet
tribench train --config configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml --max-steps 31 --quiet

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

# Export feature-preserving Unity-native assets from completed runs
bash single_export_unity.sh triangle-splatting mipnerf360/all 0
bash single_export_unity.sh mesh-splatting mipnerf360/garden 0

# Export missing Unity assets, capture Unity frames, profile FPS/memory, and format report tables
bash single_unity_eval.sh 2dts mipnerf360/bicycle 0 --unity "$UNITY" --unity-project "$PROJECT"

# MeshSplatting topology ablation: shared mesh, shader-level soup, materialized soup
bash single_mesh_topology_unity_eval.sh mipnerf360/all 0 --unity "$UNITY" --unity-project "$PROJECT"
```

## Unity-native Evaluation

TriBench can export method-preserving Unity `.triasset` packages, render Unity
test-view images for PSNR/SSIM/LPIPS, and profile GPU FPS/memory through a
Linux standalone Development Player. The workflow is designed for completed
config-driven runs under `outputs/{method}/{dataset}/{scene}`.

### What must be committed

Commit the source code and documentation needed to reproduce the Unity
benchmark:

```text
single_export_unity.sh
single_unity_eval.sh
single_mesh_topology_unity_eval.sh
tribench/unity_assets.py
tribench/cli/export_unity.py
tools/export_unity_triasset.py
tools/validate_unity_triasset_cpu.py
tools/create_unity_triasset_package.py
tools/patch_legacy_unity_vulkan.py
tools/run_unity_triasset_eval.py
tools/build_unity_profile_player.py
tools/run_unity_triasset_player_profile.py
tools/evaluate_deployment_images.py
tools/format_unity_metrics_report.py
tools/aggregate_unity_triasset_eval.py
tools/aggregate_unity_deployment_profile.py
tools/evaluate_unity_triasset_outputs.py
tools/unity_triasset_renderer/
tribench/tests/test_unity_assets*.py
tribench/tests/test_unity_deployment_cpu.py
docs/unity_native_assets.md
```

Do not commit generated data: `outputs/`, `.triasset/` packages, captured Unity
PNGs, runtime profile JSON/logs, generated Unity Player builds, Unity `Library/`
or `Temp/`, local datasets, or report CSV/JSON/Markdown files. These are
covered by `.gitignore`.

### Unity project setup

Install Unity with Linux standalone build support. Then create or update a
Unity project from the repository templates:

```bash
python3 tools/create_unity_triasset_package.py --unity-project "$PROJECT"
python3 tools/patch_legacy_unity_vulkan.py --unity-project "$PROJECT"
```

Set the usual paths:

```bash
export UNITY=/path/to/Unity/Editor/Unity
export PROJECT=/path/to/TriBenchUnity
```

`single_unity_eval.sh` applies the Vulkan compatibility patch automatically on
Linux. It also builds or reuses a Linux standalone Development Player at:

```text
outputs/unity_player/linux/TriBenchProfilePlayer.x86_64
```

### Export Unity assets only

```bash
./single_export_unity.sh 2dts mipnerf360/all cpu
./single_export_unity.sh triangle-splatting mipnerf360/garden cpu
```

Each scene writes:

```text
outputs/{method}/{dataset}/{scene}/unity_native/{method}.triasset/
```

Existing valid packages are skipped; use `--force` to re-export.

### Method-aware Unity benchmark

```bash
./single_unity_eval.sh 2dts mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT"
```

For all methods:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project "$PROJECT"
done
```

The default condition is `method-aware`; outputs are written under each scene's
`unity_method_aware/` directory and summarized in:

```text
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.csv
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.json
outputs/unity_reports/{method}/unity_method_aware/unity_metrics_report.md
```

The terminal summary reports:

```text
PSNR | SSIM | LPIPS | GPU FPS | GPU P50 ms | GPU P95 ms | Render MiB | Asset MiB
```

`GPU FPS` is computed as `1000 / GPU P50 ms`. CPU/engine FPS is deliberately
not used for paper tables.

### General-purpose Unity renderer

Use `--general-purpose` and a separate output name:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project "$PROJECT" \
    --general-purpose \
    --output-name unity_general_purpose
done
```

Methods whose exported manifest does not declare a comparable general-purpose
appearance are rejected instead of producing misleading numbers.

### MeshSplatting topology ablation

To isolate the value of MeshSplatting's shared-vertex topology, use:

```bash
./single_mesh_topology_unity_eval.sh mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT"
```

This runs `mesh-splatting` with three deployment layouts:

- `mesh`: preserves the exported indexed mesh and renders it through a real
  Unity `MeshFilter`/`MeshRenderer` with `Mesh.SetIndices`.  For
  MeshSplatting, `single_mesh_topology_unity_eval.sh` enables
  `--indexed-mesh-method-aware`, so the Unity indexed mesh path still evaluates
  the learned SH appearance instead of falling back to the older DC-only
  vertex-color baseline.
- `shader-soup`: loads the same indexed `.triasset`, but uses the procedural
  shader path as a corner-level triangle-soup intervention without duplicating
  buffers.
- `materialized-soup`: exports a separate `.triasset` with per-triangle
  duplicated positions/SH/opacity attributes and sequential soup indices. This
  is intended for asset-memory, load-feasibility, and CG compatibility evidence.

Reports are written under:

```text
outputs/unity_reports/mesh-splatting/unity_mesh_topology/{indexed_mesh,shader_soup,materialized_soup}/
```

The lower-level wrapper also accepts `--topology indexed|soup` directly when a
single shader-level condition is needed:

```bash
./single_unity_eval.sh mesh-splatting mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --topology soup \
  --output-name unity_method_aware_soup
```

For a single true Unity indexed MeshRenderer run, use:

```bash
./single_unity_eval.sh mesh-splatting mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --general-purpose \
  --topology indexed \
  --indexed-mesh-method-aware \
  --output-name unity_indexed_mesh_method_aware
```

For true exported soup assets, use a separate asset subdirectory:

```bash
./single_unity_eval.sh mesh-splatting mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --asset-subdir unity_native_materialized_soup \
  --export-topology soup \
  --output-name unity_method_aware_materialized_soup
```

### Runtime profile details

By default, GPU speed is measured with the standalone Player:

```text
--profile-runtime player
--profile-runs 3
--profile-views 3
--profile-warmup 60
--profile-frames 180
--gpu-timing-min-fraction 0.5
```

This means each scene profiles three independent Player runs, three held-out
views per run, and 180 timed frames per view after warmup. Unity's
`FrameTimingManager` may not return GPU timing on every frame, so
`--gpu-timing-min-fraction` controls the minimum valid GPU sample coverage. A
profile with no valid GPU timing fails rather than falling back to CPU time.

For fast smoke tests:

```bash
./single_unity_eval.sh diffsoup mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --profile-runs 1 \
  --profile-views 1 \
  --profile-warmup 20 \
  --profile-frames 60 \
  --fps-warmup 3 \
  --fps-frames 12
```

If quality images and metrics already exist and only GPU profiles need to be
rebuilt:

```bash
./single_unity_eval.sh diffsoup mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --skip-capture \
  --skip-metrics \
  --force-profile
```

### Resolution and dataset rules

The wrapper reads each scene config. For Mip-NeRF 360, `image_dir: images` plus
`resolution: 4` resolves to `images_4`; `resolution: 2` resolves to `images_2`.
This keeps Unity quality/profile resolution aligned with native evaluation.
The Unity camera replay currently requires COLMAP `sparse/0/cameras.bin` and
`sparse/0/images.bin`.

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

## Citation

If you find this benchmark, codebase, or results helpful in your research, please cite our paper:

```bibtex
@misc{zhang2026meshsplatbenchunifiedbenchmarktrianglebased,
      title={MeshSplatBench: A Unified Benchmark for Triangle-Based Neural Rendering}, 
      author={Kaixuan Zhang and Minxian Li and Mingwu Ren and Xiatian Zhu},
      year={2026},
      eprint={2609.01306},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2609.01306}, 
}
```

## License

This project is licensed under the [MIT License](LICENSE). Third-party submodules and vendor components under `tribench/vendor/` and `submodules/` are subject to their respective original licenses.
