"""Profile command: measure renderer performance."""

from __future__ import annotations

import json
from pathlib import Path

import typer

profile_app = typer.Typer(no_args_is_help=True)


@profile_app.callback(invoke_without_command=True)
def profile(
    ctx: typer.Context,
    method: str | None = typer.Option(None, "--method", "-m", help="Method name"),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to dataset directory",
    ),
    scene: str | None = typer.Option(None, "--scene", help="Scene name or dataset directory"),
    config: Path | None = typer.Option(  # noqa: B008
        None, "--config", help="MeshSplatBench profile config YAML"
    ),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    repeats: int = typer.Option(100, "--repeats", "-r", help="Number of profiling repetitions"),
    warmup: int = typer.Option(5, "--warmup", "-w", help="Number of untimed warmup renders"),
    backward: bool = typer.Option(
        False,
        "--backward/--no-backward",
        help="Also measure differentiable backward-pass latency",
    ),
    output: str = typer.Option("profile.json", "--output", "-o", help="Output JSON file path"),
):
    """Profile renderer performance.
    
    Measures:
    - Forward pass latency (mean, std, p50, p95)
    - Backward pass latency (where available)
    - Peak CUDA memory usage
    - Rendering FPS
    """
    profile_cfg: dict = {}
    dataset_cfg: dict = {}
    if config is not None:
        from msbench.core.config import Config

        cfg = Config.fromfile(config)
        profile_cfg = Config(cfg.get("profile", {})).to_dict()
        dataset_cfg = Config(cfg.get("dataset", {})).to_dict()
        adapter_cfg = Config(cfg.get("adapter", {})).to_dict()
        trainer_cfg = Config(cfg.get("trainer", {})).to_dict()
        output_cfg = Config(cfg.get("output", {})).to_dict()
        output_dir = output_cfg.get("dir")
        method = method or adapter_cfg.get("type") or trainer_cfg.get("type")
        checkpoint = checkpoint or adapter_cfg.get("checkpoint") or output_dir
        dataset = dataset or dataset_cfg.get("root") or dataset_cfg.get("dataset_path")
        if not _provided_on_command_line(ctx, "split"):
            split = str(profile_cfg.get("split", dataset_cfg.get("split", split)))
        if not _provided_on_command_line(ctx, "repeats"):
            repeats = int(profile_cfg.get("repeats", repeats))
        if not _provided_on_command_line(ctx, "warmup"):
            warmup = int(profile_cfg.get("warmup", warmup))
        if not _provided_on_command_line(ctx, "backward"):
            backward = bool(profile_cfg.get("backward", backward))
        if not _provided_on_command_line(ctx, "output"):
            if "output" in profile_cfg:
                output = str(profile_cfg["output"])
            elif output_dir is not None:
                output = str(Path(output_dir) / "profile.json")

    scene_name = _scene_name(scene)
    if scene is not None:
        dataset = _resolve_scene_dataset(scene, dataset)
    if scene_name is not None:
        checkpoint = _resolve_scene_path(checkpoint, scene_name)
        output = _resolve_scene_path(output, scene_name)

    if method is None:
        raise typer.BadParameter("Use --method, or set adapter.type/trainer.type in --config.")
    if dataset is None:
        raise typer.BadParameter("Use --scene/--dataset, or set dataset.root in --config.")
    if checkpoint is None:
        raise typer.BadParameter("Use --checkpoint, or set adapter.checkpoint in --config.")
    if repeats <= 0:
        raise typer.BadParameter("--repeats must be positive.")
    if warmup < 0:
        raise typer.BadParameter("--warmup must be non-negative.")

    adapter_cfg = dict(adapter_cfg) if config is not None else {}
    adapter_cfg["type"] = method
    adapter_cfg["checkpoint"] = checkpoint
    resolved_dataset_cfg = dict(dataset_cfg)
    resolved_dataset_cfg["root"] = dataset
    resolved_dataset_cfg["split"] = split

    from msbench.core.builder import build_adapter, build_dataset

    typer.echo(f"Profiling method: {method}")
    typer.echo(f"Checkpoint: {checkpoint}")
    typer.echo(f"Dataset: {dataset} (split: {split})")
    typer.echo(f"Repeats: {repeats} (warmup: {warmup})")
    typer.echo(f"Backward: {'enabled' if backward else 'disabled'}")

    try:
        adapter = build_adapter(adapter_cfg)
        profile_dataset = build_dataset(resolved_dataset_cfg)
    except (FileNotFoundError, ImportError, KeyError, NotImplementedError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if len(profile_dataset) == 0:
        raise typer.BadParameter(f"Dataset split {split!r} contains no cameras.")
    sample = profile_dataset.sample(0)
    camera = sample.camera.to(adapter.device)

    from msbench.core.profiler import profile_renderer

    result = {
        "method": method,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "split": split,
        "repeats": repeats,
        "warmup": warmup,
        "status": "complete",
        "device": str(adapter.device),
        "camera_name": sample.name,
    }
    result.update(
        profile_renderer(
            adapter,
            camera,
            repeats=repeats,
            warmup=warmup,
            include_backward=backward,
        )
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    typer.echo(f"Profile saved to {output_path}")
    typer.echo(
        f"Forward: mean={result['forward_latency_ms']:.3f} ms, "
        f"p50={result['forward']['p50_ms']:.3f} ms, "
        f"p95={result['forward']['p95_ms']:.3f} ms, fps={result['fps']:.2f}"
    )
    typer.echo(
        f"Peak CUDA memory: {result['peak_cuda_memory_gb']:.3f} GB; "
        f"primitives: {result['primitive_count']}; "
        f"checkpoint: {result['checkpoint_size_mb']} MB"
    )
    if backward:
        typer.echo(f"Backward: mean={result['backward_latency_ms']:.3f} ms")


def _provided_on_command_line(ctx: typer.Context, name: str) -> bool:
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == "COMMANDLINE"


def _scene_name(scene: str | None) -> str | None:
    if scene is None:
        return None
    return Path(scene).expanduser().name


def _resolve_scene_dataset(scene: str, dataset_root: str | None) -> str:
    scene_path = Path(scene).expanduser()
    if scene_path.is_absolute() or scene_path.exists():
        return str(scene_path)
    if scene_path.parent != Path("."):
        return str(scene_path)
    if dataset_root is None:
        return scene

    root = Path(str(dataset_root).format(scene=scene_path.name)).expanduser()
    if root.name == scene_path.name:
        return str(root)
    if _looks_like_scene_path(root):
        return str(root.parent / scene_path.name)
    return str(root / scene_path)


def _resolve_scene_path(value: str | None, scene: str) -> str | None:
    if value is None:
        return None
    raw_value = str(value)
    resolved = Path(raw_value.format(scene=scene)).expanduser()
    if "{scene}" in raw_value:
        return str(resolved)
    if _looks_like_scene_path(resolved):
        return str(resolved.parent / scene)
    replaced = _replace_scene_component(resolved, scene)
    if replaced != resolved:
        return str(replaced)
    return str(resolved)


def _replace_scene_component(path: Path, scene: str) -> Path:
    parts = list(path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if _looks_like_scene_path(Path(parts[idx])):
            parts[idx] = scene
            return Path(*parts)
    return path


def _looks_like_scene_path(path: Path) -> bool:
    known_scenes = {
        "bicycle",
        "bonsai",
        "counter",
        "garden",
        "kitchen",
        "room",
        "stump",
        "treehill",
        "flowers",
        "truck",
        "train",
    }
    return path.name in known_scenes or path.name.startswith("scan")
