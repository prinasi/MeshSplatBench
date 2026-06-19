"""Inspect command: load a model and report statistics."""

import json
from pathlib import Path
from typing import Optional

import typer

from tribench.cli.config import adapter_config, load_cli_config, output_dir as config_output_dir, section

inspect_app = typer.Typer(no_args_is_help=True)


@inspect_app.callback(invoke_without_command=True)
def inspect(
    scene: Optional[Path] = typer.Option(
        None,
        "--scene",
        "-s",
        help="Path to a dataset/scene directory",
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        "-m",
        help="Method name (e.g., 2dts, triangle-splatting)",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench config YAML"),
    output: str = typer.Option("stats.json", "--output", "-o", help="Output JSON file path"),
):
    """Inspect a dataset scene or trained model checkpoint.
    
    Scene inspection reports dataset layout and counts. Checkpoint inspection
    loads a model checkpoint and reports:
    - Primitive counts (triangles, vertices, faces)
    - Parameter counts
    - Geometry and opacity summaries
    - Backend-specific fields
    """
    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        inspect_cfg = section(cfg, "inspect")
        dataset_cfg = section(cfg, "dataset")
        if scene is None and inspect_cfg.get("scene") is not None:
            scene = Path(inspect_cfg["scene"])
        if scene is None and inspect_cfg.get("dataset", False):
            scene_root = dataset_cfg.get("root") or dataset_cfg.get("dataset_path")
            if scene_root is not None:
                scene = Path(scene_root)
        if output == "stats.json":
            run_dir = config_output_dir(cfg)
            output = str(
                inspect_cfg.get("output")
                or section(cfg, "output").get("stats_file")
                or (Path(run_dir) / "stats.json" if run_dir else output)
            )
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        method = adapter_cfg.get("type") or adapter_cfg.get("name")
        checkpoint = adapter_cfg.get("checkpoint")

    if scene is not None:
        _inspect_scene(scene, output)
        return

    if method is None or checkpoint is None:
        raise typer.BadParameter("Use --scene, or provide both --method and --checkpoint.")

    from tribench.core.registry import get_adapter

    typer.echo(f"Inspecting method: {method}")
    typer.echo(f"Checkpoint: {checkpoint}")

    try:
        adapter_cls = get_adapter(method)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    adapter = adapter_cls()

    try:
        adapter.load_checkpoint(checkpoint)
    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        stats = adapter.model_stats()
    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    typer.echo(f"Statistics saved to {output_path}")

    # Print summary
    typer.echo("\n--- Model Statistics ---")
    for key, value in stats.items():
        typer.echo(f"  {key}: {value}")


def _inspect_scene(scene: Path, output: str) -> None:
    scene_path = scene.expanduser()
    info = _scene_info(scene_path)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2))

    typer.echo(f"Inspecting scene: {scene_path}")
    typer.echo(f"Scene inspection saved to {output_path}")
    typer.echo("\n--- Scene Inspection ---")
    for key, value in info.items():
        typer.echo(f"  {key}: {value}")


def _scene_info(scene_path: Path) -> dict[str, object]:
    exists = scene_path.exists()
    is_dir = scene_path.is_dir()
    info: dict[str, object] = {
        "path": str(scene_path),
        "exists": exists,
        "is_dir": is_dir,
    }
    if not exists:
        info["dataset_type"] = "missing"
        return info
    if not is_dir:
        info["dataset_type"] = "file"
        return info

    image_dirs = [scene_path / "images", scene_path / "image"]
    image_files = []
    for image_dir in image_dirs:
        if image_dir.exists():
            image_files = _image_files(image_dir)
            break

    sparse_dir = scene_path / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_path / "sparse"

    transforms = sorted(path.name for path in scene_path.glob("transforms_*.json"))
    has_dtu_cameras = (scene_path / "cameras.npz").exists()
    has_colmap = sparse_dir.exists()

    if has_dtu_cameras:
        dataset_type = "dtu"
    elif transforms:
        dataset_type = "blender"
    elif has_colmap:
        dataset_type = "colmap"
    else:
        dataset_type = "unknown"

    info.update(
        {
            "dataset_type": dataset_type,
            "image_count": len(image_files),
            "image_dir": str(image_files[0].parent) if image_files else None,
            "has_sparse": has_colmap,
            "sparse_dir": str(sparse_dir) if has_colmap else None,
            "has_cameras_npz": has_dtu_cameras,
            "transforms": transforms,
        }
    )
    return info


def _image_files(image_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix in suffixes
    )
