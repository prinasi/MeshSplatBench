# TriBench

A Python-first developer toolkit for unified benchmarking of triangle-splatting-style radiance field methods.

## Overview

TriBench provides a single, consistent evaluation framework for comparing different triangle-based radiance field reconstruction methods. Instead of wrangling per-repo scripts and ad-hoc metrics, you get:

- **Standardized metrics** -- PSNR, SSIM, LPIPS, and geometry quality, computed the same way for every method.
- **Adapter pattern** -- plug any triangle-splatting method into the same API.
- **Reproducible experiments** -- YAML configs lock down every hyperparameter.
- **CLI-first workflow** -- inspect, profile, evaluate, and compare from the terminal.

## Supported Methods

| Method | Adapter | Status |
|---|---|---|
| 2D Triangle Splatting (2DTS) | `D2TSAdapter` | Planned |
| Triangle Splatting | `TriangleSplattingAdapter` | Planned |
| MeshSplatting | `MeshSplattingAdapter` | Planned |
| DiffSoup | `DiffSoupAdapter` | Planned |

## Installation

```bash
git clone https://github.com/your-org/TriBench.git
cd TriBench
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train, render, evaluate, inspect, and profile from one experiment config
tribench train --config configs/triangle_splatting_colmap.yaml
tribench render images --config configs/triangle_splatting_colmap.yaml
tribench eval images --config configs/triangle_splatting_colmap.yaml
tribench inspect --config configs/triangle_splatting_colmap.yaml
tribench profile --config configs/triangle_splatting_colmap.yaml

# Config values can still be overridden from the CLI
tribench render video --config configs/triangle_splatting_colmap.yaml --output-dir outputs/demo-video
```

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

## License

MIT
