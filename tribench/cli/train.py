"""Generic train command for TriBench.

Dispatches to method-specific ``TrainingMethod`` implementations via the
training-method registry.  Every registered method is driven by the same
``TrainingLoop`` orchestrator, so logging, timing, and checkpoint
scheduling are consistent across backends.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from tribench.core.builder import build_training_loop, build_training_method
from tribench.core.config import Config, load_config, resolve_dataset_config, save_config_snapshot
from tribench.core.runtime_stats import run_with_training_stats
from tribench.trainers.checkpoints import find_latest_point_cloud_checkpoint
from tribench.trainers.loop import TrainingConfig, TrainingLoop
from tribench.trainers.registry import get_training_method

TRAIN_CONTEXT_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}


def _yaml_to_dict(config: Path | None) -> dict:
    """Load a YAML config file into a dict, or return {} if None."""
    if config is None:
        return {}
    raw = load_config(config)
    if not isinstance(raw, dict):
        raise typer.BadParameter("Training config must be a YAML mapping.")
    return raw


def _coerce_value(value: str) -> object:
    """Convert simple CLI strings to bool/int/float where possible."""
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_extra_args(raw_args: list[str]) -> dict[str, object]:
    """Parse ``--key value`` pairs from ``typer.Context.args`` into a dict."""
    result: dict[str, object] = {}
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("--"):
                result[key] = _coerce_value(raw_args[i + 1])
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            i += 1
    return result


def train(
    ctx: typer.Context,
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Method name"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to dataset directory"),
    output_dir: Path = typer.Option("./outputs", "--output-dir", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional training config YAML"),
    max_steps: int = typer.Option(30_000, "--max-steps", help="Maximum training steps"),
    log_interval: int = typer.Option(100, "--log-interval", help="Steps between log output"),
    eval_interval: int = typer.Option(1_000, "--eval-interval", help="Steps between evaluation"),
    save_interval: int = typer.Option(5_000, "--save-interval", help="Steps between checkpoints"),
    images: str = typer.Option("images", "--images", help="Image folder inside the dataset root"),
    resolution: int = typer.Option(
        -1,
        "--resolution",
        "-r",
        help="1/2/4/8 downscale factor; other positive values are target width; -1 caps large images",
    ),
    eval_split: bool = typer.Option(True, "--eval/--no-eval", help="Hold out every 8th image for test"),
    white_background: bool = typer.Option(False, "--white-background", help="Use white background"),
    outdoor: bool = typer.Option(False, "--outdoor", help="Use outdoor triangle-splatting pruning rules"),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce trainer output"),
):
    """Train a registered method using the generic TrainingLoop.

    Method-specific options can be passed after ``--`` as ``--key value``
    pairs, or via a YAML config file with ``--config``.  For example::

        tribench train -m triangle-splatting -d data/bicycle \\
            --max-steps 30000 -- densify-until-iter 15000
    """
    if config is not None:
        cfg = Config.fromfile(config)
        if "trainer" in cfg or "loop" in cfg or "dataset" in cfg:
            summary = _train_from_structured_config(cfg, output_dir=output_dir, quiet=quiet)
            if not quiet:
                typer.echo(
                    f"Training complete: {summary['total_steps']} steps "
                    f"in {summary['total_time_s']:.1f}s"
                )
            return

    if method is None or dataset is None:
        raise typer.BadParameter(
            "Use --config with a structured TriBench config, or provide both "
            "--method and --dataset."
        )

    # -- Resolve the training method class ----------------------------
    method_cls = get_training_method(method)

    # -- Build output directory ---------------------------------------
    output_dir = output_dir.expanduser()
    if output_dir.name == "outputs":
        canonical = method.lower().replace("_", "-")
        output_dir = output_dir / canonical / dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Merge config sources -----------------------------------------
    # Priority: CLI flags > extra args > YAML config > defaults
    yaml_dict = _yaml_to_dict(config)
    extra_dict = _parse_extra_args(ctx.args)

    # Start with YAML, override with extra args
    kwargs: dict = {**yaml_dict, **extra_dict}

    # Standard CLI overrides
    kwargs.setdefault("images", images)
    kwargs.setdefault("resolution", resolution)
    kwargs.setdefault("eval_split", eval_split)
    kwargs.setdefault("white_background", white_background)
    if outdoor:
        kwargs["outdoor"] = True
    else:
        kwargs.setdefault("outdoor", outdoor)

    # Separate "extra" args that the TrainingMethod may not know about
    import inspect

    sig = inspect.signature(method_cls.__init__)
    known_params = set(sig.parameters.keys()) - {"self"}
    init_kwargs: dict = {}
    extra_kwargs: dict = {}
    for k, v in kwargs.items():
        if k in known_params:
            init_kwargs[k] = v
        else:
            extra_kwargs[k] = v

    # Pack unknown keys into extra_args if the method accepts it
    if "extra_args" in known_params and extra_kwargs:
        init_kwargs["extra_args"] = extra_kwargs

    # -- Instantiate the training method ------------------------------
    # Always pass the required positional-ish args
    if "dataset" in known_params:
        init_kwargs["dataset"] = str(dataset.expanduser())
    if "output_dir" in known_params:
        init_kwargs["output_dir"] = str(output_dir)
    if "max_steps" in known_params:
        init_kwargs["max_steps"] = max_steps

    training_method = method_cls(**init_kwargs)

    # -- Run the training loop ----------------------------------------
    loop_config = TrainingConfig(
        max_steps=max_steps,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        output_dir=str(output_dir),
    )
    loop = TrainingLoop(training_method, loop_config)
    summary = run_with_training_stats(loop.train, output_dir)

    if not quiet:
        typer.echo(f"Training complete: {summary['total_steps']} steps "
                    f"in {summary['total_time_s']:.1f}s")


def _train_from_structured_config(
    cfg: Config,
    *,
    output_dir: Path,
    quiet: bool,
) -> dict:
    """Run training from a TriBench structured config."""
    if "trainer" not in cfg:
        raise typer.BadParameter("Structured training config must include 'trainer'.")
    if "dataset" not in cfg:
        raise typer.BadParameter("Structured training config must include 'dataset'.")

    output_cfg = Config(cfg.get("output", {})).to_dict()
    run_output_dir = Path(output_cfg.get("dir") or output_dir).expanduser()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, run_output_dir)

    dataset_cfg = resolve_dataset_config(Config(cfg.dataset).to_dict())
    dataset_root = dataset_cfg.pop("root", dataset_cfg.pop("dataset_path", None))
    if dataset_root is None:
        raise typer.BadParameter("dataset.root is required for training configs.")

    trainer_cfg = Config(cfg.trainer).to_dict()
    loop_cfg = Config(cfg.get("loop", {})).to_dict()
    max_steps = int(trainer_cfg.pop("max_steps", loop_cfg.get("max_steps", 30_000)))
    loop_cfg.setdefault("max_steps", max_steps)

    if "image_dir" in dataset_cfg and "images" not in trainer_cfg:
        trainer_cfg["images"] = dataset_cfg["image_dir"]
    for key in ("resolution", "eval_split", "llffhold"):
        if key in dataset_cfg and key not in trainer_cfg:
            trainer_cfg[key] = dataset_cfg[key]

    method_name = str(trainer_cfg.get("type", trainer_cfg.get("name", ""))).replace("_", "-")
    native_loop = bool(
        trainer_cfg.pop(
            "native_loop",
            method_name in {"triangle-splatting", "mesh-splatting"},
        )
    )
    if method_name == "triangle-splatting" and native_loop:
        summary = run_with_training_stats(
            lambda: _run_triangle_splatting_native_config(
                trainer_cfg=trainer_cfg,
                dataset_root=str(dataset_root),
                output_dir=run_output_dir,
                max_steps=max_steps,
                quiet=quiet,
            ),
            run_output_dir,
        )
        if not quiet:
            typer.echo(f"Config snapshot saved to {run_output_dir / 'config.yaml'}")
        return summary

    if method_name == "mesh-splatting" and native_loop:
        summary = run_with_training_stats(
            lambda: _run_mesh_splatting_native_config(
                trainer_cfg=trainer_cfg,
                dataset_root=str(dataset_root),
                output_dir=run_output_dir,
                max_steps=max_steps,
                quiet=quiet,
            ),
            run_output_dir,
        )
        if not quiet:
            typer.echo(f"Config snapshot saved to {run_output_dir / 'config.yaml'}")
        return summary

    if method_name == "2dts":
        from tribench.trainers.d2ts_native import run_d2ts_native_config

        summary = run_with_training_stats(
            lambda: run_d2ts_native_config(
                cfg=cfg,
                dataset_root=str(dataset_root),
                output_dir=run_output_dir,
                max_steps=max_steps,
                quiet=quiet,
            ),
            run_output_dir,
        )
        if not quiet:
            typer.echo(f"Config snapshot saved to {run_output_dir / 'config.yaml'}")
        return summary

    training_method = build_training_method(
        trainer_cfg,
        default_args={
            "dataset": dataset_root,
            "output_dir": str(run_output_dir),
            "max_steps": max_steps,
        },
    )
    loop = build_training_loop(loop_cfg, training_method, output_dir=run_output_dir)
    summary = run_with_training_stats(loop.train, run_output_dir)
    if not quiet:
        typer.echo(f"Config snapshot saved to {run_output_dir / 'config.yaml'}")
    return summary


def _run_triangle_splatting_native_config(
    *,
    trainer_cfg: dict,
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool,
) -> dict:
    """Run triangle-splatting with its native training loop."""
    from tribench.vendor.triangle_splatting.train import run_training

    start = time.time()
    argv = _triangle_splatting_native_argv(
        trainer_cfg=trainer_cfg,
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_steps=max_steps,
        quiet=quiet,
    )
    run_training(argv)
    total_time = time.time() - start
    return {
        "total_steps": max_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": total_time * 1000.0 / max(max_steps, 1),
        "final_losses": {},
    }


def _run_mesh_splatting_native_config(
    *,
    trainer_cfg: dict,
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool,
) -> dict:
    """Run MeshSplatting with its native training loop."""
    from tribench.vendor.mesh_splatting.train import run_training

    latest = find_latest_point_cloud_checkpoint(output_dir)
    start_step = 0
    if latest is not None:
        start_step, ckpt_dir = latest
        if start_step >= max_steps:
            typer.echo(
                f"Found checkpoint at {ckpt_dir}; requested max_steps={max_steps} "
                "is already reached."
            )
            return {
                "total_steps": max_steps,
                "start_step": start_step,
                "trained_steps": 0,
                "total_time_s": 0.0,
                "avg_step_time_ms": 0.0,
                "final_losses": {},
            }

    start = time.time()
    argv = _mesh_splatting_native_argv(
        trainer_cfg=trainer_cfg,
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_steps=max_steps,
        quiet=quiet,
        load_iteration=start_step if start_step > 0 else None,
    )
    run_training(argv)
    total_time = time.time() - start
    trained_steps = max(max_steps - start_step, 0)
    return {
        "total_steps": max_steps,
        "start_step": start_step,
        "trained_steps": trained_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": total_time * 1000.0 / max(trained_steps, 1),
        "final_losses": {},
    }


def _triangle_splatting_native_argv(
    *,
    trainer_cfg: dict,
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool,
) -> list[str]:
    argv = [
        "-s",
        str(Path(dataset_root).expanduser()),
        "-m",
        str(output_dir),
        "--iterations",
        str(max_steps),
        "--test_iterations",
        "-1",
        "--save_iterations",
        "7000",
        str(max_steps),
    ]

    images = trainer_cfg.get("images")
    if images is not None:
        argv.extend(["-i", str(images)])
    resolution = trainer_cfg.get("resolution")
    if resolution is not None:
        argv.extend(["-r", str(resolution)])
    if bool(trainer_cfg.get("white_background", False)):
        argv.append("--white_background")
    if bool(trainer_cfg.get("eval_split", True)):
        argv.append("--eval")
    if bool(trainer_cfg.get("no_dome", False)):
        argv.append("--no_dome")
    if bool(trainer_cfg.get("outdoor", False)):
        argv.append("--outdoor")
    if quiet:
        argv.append("--quiet")
    latest = find_latest_point_cloud_checkpoint(output_dir)
    if latest is not None:
        step, ckpt_dir = latest
        argv.extend(["--load_iteration", str(step)])
        print(f"Loaded checkpoint: {ckpt_dir}")

    passthrough = {
        "depth_ratio",
        "lambda_normals",
        "lambda_dist",
        "iteration_mesh",
        "densify_until_iter",
        "lambda_opacity",
        "importance_threshold",
        "lr_triangles_points_init",
        "lambda_size",
        "max_shapes",
        "opacity_dead",
        "opacity_lr",
        "feature_lr",
        "lr_sigma",
        "proba_distr",
        "split_size",
        "add_shape",
        "position_lr_delay_mult",
        "position_lr_max_steps",
        "densification_interval",
        "densify_from_iter",
    }
    for key in sorted(passthrough):
        if key in trainer_cfg:
            argv.extend([f"--{key}", str(trainer_cfg[key])])
    if bool(trainer_cfg.get("random_background", False)):
        argv.append("--random_background")
    return argv


def _mesh_splatting_native_argv(
    *,
    trainer_cfg: dict,
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool,
    load_iteration: int | None = None,
) -> list[str]:
    argv = [
        "-s",
        str(Path(dataset_root).expanduser()),
        "-m",
        str(output_dir),
        "--iterations",
        str(max_steps),
        "--test_iterations",
        "-1",
    ]

    images = trainer_cfg.get("images")
    if images is not None:
        argv.extend(["-i", str(images)])
    resolution = trainer_cfg.get("resolution")
    if resolution is not None:
        argv.extend(["-r", str(resolution)])
    if bool(trainer_cfg.get("white_background", False)):
        argv.append("--white_background")
    if bool(trainer_cfg.get("eval_split", True)):
        argv.append("--eval")
    if bool(trainer_cfg.get("indoor", False)):
        argv.append("--indoor")
    if quiet:
        argv.append("--quiet")
    if load_iteration is not None:
        argv.extend(["--load_iteration", str(load_iteration)])

    passthrough = {
        "add_percentage",
        "data_device",
        "densification_interval",
        "densify_from_iter",
        "densify_until_iter",
        "depth_lambda_final",
        "depth_lambda_init",
        "depth_ratio",
        "feature_lr",
        "final_opacity_iter",
        "intervall_add_triangles",
        "iteration_mesh",
        "lamba_depth",
        "lambda_dssim",
        "lambda_normals",
        "lambda_normals_super",
        "lambda_vertex",
        "lambda_weight",
        "lr_triangles_points_init",
        "max_diff_threshold",
        "max_points",
        "position_lr_delay_mult",
        "position_lr_max_steps",
        "prune_size",
        "prune_triangles_threshold",
        "set_sigma",
        "set_weight",
        "sigma_start",
        "sigma_until",
        "size_probs_zero",
        "size_probs_zero_image_space",
        "splitt_large_triangles",
        "start_opacity_floor",
        "start_pruning",
        "start_upsampling",
        "start_vertex_opt",
        "upscaling_factor",
        "weight_lr",
    }
    for key in sorted(passthrough):
        if key in trainer_cfg:
            argv.extend([f"--{key}", str(trainer_cfg[key])])
    if bool(trainer_cfg.get("random_background", False)):
        argv.append("--random_background")
    return argv
