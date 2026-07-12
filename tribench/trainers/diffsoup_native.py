"""Native DiffSoup training integration for TriBench.

The implementation delegates the optimisation loop to DiffSoup example
scripts vendored inside TriBench, so training does not require a separate
DiffSoup checkout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

import torch

from tribench.trainers.hooks import TrainingMethod
from tribench.trainers.registry import register_training_method


DEFAULT_DIFFSOUP_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "diffsoup"
DEFAULT_DIFFSOUP_EXAMPLES_DIR = DEFAULT_DIFFSOUP_ROOT / "examples"


def run_diffsoup_native_config(
    *,
    trainer_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run DiffSoup's original training script for one TriBench config."""
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = output_dir / "final_params.pt"
    if checkpoint.is_file() and bool(trainer_cfg.get("skip_completed", True)):
        steps_done = _checkpoint_steps(checkpoint)
        if steps_done >= max_steps:
            if not quiet:
                print(
                    f"Found DiffSoup checkpoint at {checkpoint}; "
                    f"requested max_steps={max_steps} is already reached."
                )
            return {
                "total_steps": max_steps,
                "start_step": steps_done,
                "trained_steps": 0,
                "total_time_s": 0.0,
                "avg_step_time_ms": 0.0,
                "final_losses": {},
            }

    examples_dir = _resolve_examples_dir(trainer_cfg)

    dataset_kind = _resolve_dataset_kind(trainer_cfg, dataset_cfg, dataset_root)
    start = time.time()
    if dataset_kind in {"auto", "colmap", "dtu"}:
        _run_mip360_script(
            examples_dir=examples_dir,
            trainer_cfg=trainer_cfg,
            dataset_root=dataset_root,
            output_dir=output_dir,
            max_steps=max_steps,
            dataset_kind=dataset_kind,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_kind in {"synthetic", "shelly"}:
        _run_synthetic_script(
            examples_dir=examples_dir,
            trainer_cfg=trainer_cfg,
            dataset_root=dataset_root,
            output_dir=output_dir,
            max_steps=max_steps,
            dataset_kind=dataset_kind,
            dataset_cfg=dataset_cfg,
        )
    else:
        raise ValueError(
            "DiffSoup dataset_type must resolve to one of: auto, colmap, dtu, "
            f"synthetic, shelly; got {dataset_kind!r}."
        )

    total_time = time.time() - start
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DiffSoup training did not produce {checkpoint}")
    trained_steps = max_steps
    return {
        "total_steps": max_steps,
        "start_step": 0,
        "trained_steps": trained_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": total_time * 1000.0 / max(trained_steps, 1),
        "final_losses": _final_loss(checkpoint),
    }


def _run_mip360_script(
    *,
    examples_dir: Path,
    trainer_cfg: dict[str, Any],
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    dataset_kind: str,
    dataset_cfg: dict[str, Any],
) -> None:
    module = _load_reference_script(examples_dir, "01_mip360.py")
    downscale = _coerce_downscale(
        trainer_cfg.get("downscale", dataset_cfg.get("downscale", dataset_cfg.get("resolution"))),
        default=2 if dataset_kind == "dtu" else 4,
        allowed={0, 1, 2, 4, 8},
    )
    white_background = trainer_cfg.get("white_background")
    if white_background is None and dataset_kind == "dtu":
        white_background = False
    dtu_eval_mode = str(dataset_cfg.get("dtu_eval_mode", trainer_cfg.get("dtu_eval_mode", "full"))).lower()
    foreground_training = bool(trainer_cfg.get("foreground_training", False))
    target_prims = _primitive_target(trainer_cfg, "n_points", "target_prims")
    with _reference_import_context(examples_dir):
        module.main(
            scene_root=str(Path(dataset_root).expanduser()),
            batch_size=int(trainer_cfg.get("batch_size", 4)),
            steps=max_steps,
            n_points=target_prims,
            downscale=downscale,
            flip_z=bool(trainer_cfg.get("flip_z", True)),
            dataset_type=dataset_kind,
            out_dir=str(output_dir),
            white_background=None if white_background is None else bool(white_background),
            foreground_training=foreground_training,
            dtu_eval_mode=dtu_eval_mode,
        )


def _run_synthetic_script(
    *,
    examples_dir: Path,
    trainer_cfg: dict[str, Any],
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    dataset_kind: str,
    dataset_cfg: dict[str, Any],
) -> None:
    point_cloud_init = bool(trainer_cfg.get("point_cloud_init", False))
    point_cloud_cfg = trainer_cfg.get("point_cloud", trainer_cfg.get("point_cloud_path"))
    use_point_cloud = point_cloud_init or point_cloud_cfg is not None
    script_name = "03_random_init.py" if bool(trainer_cfg.get("random_init", False)) or use_point_cloud else "02_synthetic.py"
    scene = str(dataset_cfg.get("scene") or Path(dataset_root).name)
    datasets_root = str(Path(dataset_root).expanduser().parent)
    downscale = _coerce_downscale(
        trainer_cfg.get("downscale", dataset_cfg.get("downscale", dataset_cfg.get("resolution"))),
        default=2 if dataset_kind == "shelly" else 1,
        allowed={1, 2, 4},
    )
    argv = [
        script_name,
        "--scene",
        scene,
        "--datasets_root",
        datasets_root,
        "--steps",
        str(max_steps),
        "--batch_size",
        str(int(trainer_cfg.get("batch_size", 4))),
        "--target_prims",
        str(_primitive_target(trainer_cfg, "target_prims", "n_points")),
        "--downscale",
        str(downscale),
        "--out_dir",
        str(output_dir),
    ]
    if script_name == "02_synthetic.py":
        argv.extend([
            "--mobilenerf_root",
            str(Path(trainer_cfg.get("mobilenerf_root", "./datasets/mobilenerf_results")).expanduser()),
        ])
    else:
        if use_point_cloud:
            point_cloud_path = Path(str(point_cloud_cfg or "points3d.ply")).expanduser()
            if not point_cloud_path.is_absolute():
                point_cloud_path = Path(dataset_root).expanduser() / point_cloud_path
            argv.extend(["--point_cloud", str(point_cloud_path)])
        argv.extend(["--n_points", str(int(trainer_cfg.get("seed_points", 100_000)))])
    if dataset_kind == "shelly" or bool(trainer_cfg.get("no_png_suffix", False)):
        argv.append("--no_png_suffix")

    module = _load_reference_script(examples_dir, script_name)
    with _reference_import_context(examples_dir):
        with _temporary_argv(argv):
            module.main()


def _resolve_examples_dir(trainer_cfg: dict[str, Any]) -> Path:
    examples_dir = trainer_cfg.get("examples_dir")
    if examples_dir is not None:
        resolved = Path(examples_dir).expanduser()
    elif trainer_cfg.get("repo_root") is not None:
        resolved = Path(trainer_cfg["repo_root"]).expanduser() / "examples"
    else:
        resolved = DEFAULT_DIFFSOUP_EXAMPLES_DIR

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Vendored DiffSoup examples directory not found: {resolved}. "
            "Reinstall TriBench or pass trainer.examples_dir for an explicit override."
        )
    return resolved


def _load_reference_script(examples_dir: Path, script_name: str) -> ModuleType:
    script_path = examples_dir / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"DiffSoup reference script not found: {script_path}")

    with _reference_import_context(examples_dir):
        module_name = f"_tribench_diffsoup_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DiffSoup reference script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


@contextmanager
def _reference_import_context(examples_dir: Path) -> Iterator[None]:
    import tribench.vendor.diffsoup as vendored_diffsoup

    old_diffsoup = sys.modules.get("diffsoup")
    old_utils = sys.modules.get("utils")
    old_path = list(sys.path)
    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    sys.modules["diffsoup"] = vendored_diffsoup
    sys.modules.pop("utils", None)
    sys.path.insert(0, str(examples_dir))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tribench")
    try:
        yield
    finally:
        sys.path[:] = old_path
        if old_diffsoup is None:
            sys.modules.pop("diffsoup", None)
        else:
            sys.modules["diffsoup"] = old_diffsoup
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
        if old_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mplconfig


@contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    old_argv = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _resolve_dataset_kind(
    trainer_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_root: str,
) -> str:
    explicit = trainer_cfg.get("dataset_type")
    if explicit is not None:
        return str(explicit).replace("_", "-").lower()
    dataset_type = str(dataset_cfg.get("type", "auto")).lower()
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    if dataset_type in {"dtu"} or dataset_name == "dtu":
        return "dtu"
    if dataset_type in {"synthetic", "blender", "nerf-synthetic", "nerf_synthetic"}:
        return "synthetic"
    if dataset_type == "shelly" or dataset_name == "shelly":
        return "shelly"
    if dataset_type in {"colmap", "tanks", "tandt", "mipnerf360", "auto"}:
        return "colmap"
    root = Path(dataset_root)
    if (root / "cameras.npz").is_file():
        return "dtu"
    if (root / "transforms_train.json").is_file():
        return "synthetic"
    return "auto"


def _coerce_downscale(value: Any, *, default: int, allowed: set[int]) -> int:
    try:
        downscale = int(value)
    except (TypeError, ValueError):
        downscale = default
    if downscale not in allowed:
        downscale = default
    return downscale


def _primitive_target(trainer_cfg: dict[str, Any], primary: str, fallback: str) -> int:
    value = trainer_cfg.get("max_primitives")
    if value is None:
        value = trainer_cfg.get(primary, trainer_cfg.get(fallback, 15_000))
    target = int(value)
    if target <= 0:
        raise ValueError(f"Primitive target must be positive, got {target!r}")
    return target


def _checkpoint_steps(path: Path) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return 0
    try:
        return int(ckpt.get("steps", 0))
    except Exception:
        return 0


def _final_loss(path: Path) -> dict[str, float]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    losses = ckpt.get("losses")
    if isinstance(losses, list) and losses:
        return {"total": float(losses[-1])}
    return {}


@register_training_method("diffsoup")
class DiffSoupNativeTrainingMethod(TrainingMethod):
    """Placeholder registry entry; DiffSoup trains through its native script."""

    def _native_only(self) -> None:
        raise RuntimeError(
            "DiffSoup uses its native training script. Use `tribench train "
            "--config <diffsoup config>` or `tribench train -m diffsoup -d <dataset>`."
        )

    def sample_batch(self):
        self._native_only()

    def render_train(self, cameras):
        self._native_only()

    def compute_loss(self, output, gt_images):
        self._native_only()

    def optimizer_step(self) -> None:
        self._native_only()

    def update_structure(self, step: int):
        self._native_only()
