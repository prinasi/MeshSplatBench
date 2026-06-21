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

| Method                       | Adapter                    | Status  |
| ---------------------------- | -------------------------- | ------- |
| 2D Triangle Splatting (2DTS) | `D2TSAdapter`              | Initial config/train/render/eval support |
| Triangle Splatting           | `TriangleSplattingAdapter` | Planned |
| MeshSplatting                | `MeshSplattingAdapter`     | Planned |
| DiffSoup                     | `DiffSoupAdapter`          | Planned |

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
3. **Phase 3 -- Remaining Adapters**: Triangle Splatting, MeshSplatting, and DiffSoup adapters with parity validation.
4. **Phase 4 -- Polish & Release**: CLI refinement, experiment configs, documentation, and public release.

### Next Development Steps

The immediate goal is to make Triangle Splatting the first complete, trusted
reference pipeline before broadening adapter coverage. This gives the remaining
methods a concrete standard for training, rendering, evaluation, profiling, and
configuration behavior.

1. **Stabilize Triangle Splatting as the canonical baseline**
   - Ensure `tribench train/render/eval/inspect/profile --config ...` works end-to-end for one COLMAP scene.
   - Keep the single experiment YAML as the source of truth for train, render, eval, profile, mesh export, and inspect outputs.

2. **Formalize loss configuration**
   - Add an explicit `loss:` section to experiment configs instead of relying only on trainer extra arguments.
   - Record pixel loss type and weights such as `lambda_dssim`, `lambda_opacity`, `lambda_size`, `lambda_normals`, and `lambda_dist`.
   - Wire `TriangleSplattingTrainingMethod` to read the normalized loss config while preserving existing defaults.

3. **Make profiling functional**
   - Extend `tribench profile --config ...` beyond metadata output.
   - Report forward latency, FPS, peak CUDA memory, primitive count, checkpoint size, and optional backward latency.
   - Start with Triangle Splatting, then reuse the same profile schema for other adapters.

4. **Add parity validation**
   - Compare TriBench rendering against the upstream Triangle Splatting renderer on the same scene and checkpoint.
   - Track camera consistency, image PSNR, max pixel difference, primitive counts, and checkpoint loading behavior.
   - Add a small train/render/eval smoke test, such as a 100-step run, to catch pipeline regressions.

5. **Clean up known test debt**
   - Fix existing renderer test mismatches around DiffSoup no-model errors and Triangle Splatting missing-checkpoint exceptions.
   - Keep the full `tribench/tests` suite green before expanding adapter work.

6. **Then expand adapter coverage**
   - Bring 2DTS, MeshSplatting, and DiffSoup up to the same adapter contract.
   - For each method, add a unified config, render/eval/profile smoke path, and parity notes.

## License

MIT
