"""Eval commands: image metrics and mesh/DTU metrics."""

import json
from pathlib import Path
from typing import Optional

import typer

from tribench.cli.config import (
    adapter_config,
    dataset_config,
    load_cli_config,
    merged_section,
    metrics_file,
    output_dir,
    section,
)
from tribench.core.config import merge_dicts

eval_app = typer.Typer(no_args_is_help=True)


@eval_app.callback()
def evaluate():
    """Evaluate image renders and mesh/DTU geometry."""


@eval_app.command("images")
def evaluate_images(
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Method name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Path to dataset directory"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench evaluation config YAML"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    output: str = typer.Option("metrics.json", "--output", "-o", help="Output JSON file path"),
    save_renders: bool = typer.Option(False, "--save-renders", help="Save rendered images"),
    render_dir: str = typer.Option("renders/", "--render-dir", help="Directory for saved renders"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(
        1,
        "--resolution",
        "-r",
        help="1/2/4/8 downscale factor; other positive values are target width; -1 caps large images",
    ),
    eval_every: int = typer.Option(8, "--eval-every"),
):
    """Evaluate rendering quality on a dataset split."""
    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.config import save_config_snapshot
    from tribench.core.datasets import load_dataset
    from tribench.core.rendering import load_adapter, render_dataset_split

    method_label = method
    checkpoint_label = checkpoint
    dataset_label = dataset
    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
            raise typer.BadParameter(
                "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
            )
        dataset_cfg = dataset_config(
            cfg,
            dataset=dataset,
            dataset_type=dataset_type if dataset_type != "auto" else None,
            split=split if split != "test" else None,
            image_dir=image_dir if image_dir != "images" else None,
            resolution=resolution if resolution != 1 else None,
            eval_every=eval_every if eval_every != 8 else None,
            stage="eval",
        )
        adapter = build_adapter(adapter_cfg)
        ds = build_dataset(dataset_cfg)
        split = str(dataset_cfg.get("split", split))
        method_label = adapter_cfg.get("type")
        checkpoint_label = adapter_cfg.get("checkpoint")
        dataset_label = dataset_cfg.get("root", dataset_cfg.get("dataset_path"))
        eval_cfg = section(cfg, "eval")
        render_cfg = section(cfg, "render")
        if output == "metrics.json":
            output = metrics_file(cfg, default=output)
        render_dir = str(
            eval_cfg.get("render_dir")
            or render_cfg.get("output_dir")
            or render_cfg.get("dir")
            or output_dir(cfg, render_dir)
        )
        save_renders = bool(eval_cfg.get("save_renders", render_cfg.get("save_renders", save_renders)))
        save_config_snapshot(cfg, Path(output).parent)
    else:
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        adapter = load_adapter(method, checkpoint)
        ds = load_dataset(
            dataset,
            dataset_type=dataset_type,
            split=split,
            eval_every=eval_every,
            image_dir=image_dir,
            resolution=resolution,
        )
    manifest = render_dataset_split(
        adapter,
        ds,
        render_dir,
        save_gt=save_renders,
        save_aux=False,
        metrics=True,
    )
    metrics = {
        "method": method_label,
        "checkpoint": checkpoint_label,
        "dataset": dataset_label,
        "split": split,
        "num_views": manifest["num_frames"],
        "aggregate": manifest["aggregate"],
        "per_view": [
            {"name": frame["name"], **(frame["metrics"] or {})}
            for frame in manifest["frames"]
            if frame["metrics"] is not None
        ],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(metrics, indent=2))
    typer.echo(f"Metrics saved to {output}")


@eval_app.command("mesh")
def evaluate_mesh(
    pred: Optional[str] = typer.Option(None, "--pred", help="Predicted mesh/point cloud"),
    dtu_root: Optional[str] = typer.Option(None, "--dtu-root", help="Official DTU dataset root"),
    scan_id: Optional[str] = typer.Option(None, "--scan-id", help="DTU scan id, e.g. 24 or scan24"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench evaluation config YAML"),
    output: str = typer.Option("mesh_metrics.json", "--output", "-o"),
    samples: Optional[int] = typer.Option(None, "--samples"),
    downsample_density: Optional[float] = typer.Option(None, "--downsample-density"),
    max_dist: Optional[float] = typer.Option(None, "--max-dist"),
    cull_masks: Optional[bool] = typer.Option(None, "--cull-masks/--no-cull-masks"),
):
    """Evaluate a reconstructed mesh. DTU configs use the DTU mesh protocol."""
    from tribench.core.mesh_eval import dtu_mesh_metrics, write_mesh_metrics

    scene_root = None
    if config is not None:
        cfg = load_cli_config(config)
        legacy_cfg = merge_dicts(section(cfg, "dtu_mesh"), merged_section(cfg, "eval", nested="dtu_mesh"))
        mesh_cfg = merge_dicts(section(cfg, "mesh_eval"), merged_section(cfg, "eval", nested="mesh"))
        eval_cfg = merge_dicts(legacy_cfg, mesh_cfg)
        inferred_dtu_root, inferred_scan_id = _infer_dtu_eval_target(cfg)
        scene_root = _infer_dtu_scene_root(cfg)
        pred = pred or eval_cfg.get("pred") or _infer_pred_mesh(cfg)
        dtu_root = dtu_root or eval_cfg.get("dtu_root") or inferred_dtu_root
        scan_id = scan_id or eval_cfg.get("scan_id") or inferred_scan_id
        if output == "mesh_metrics.json":
            output_cfg = section(cfg, "output")
            output = str(
                eval_cfg.get("output")
                or output_cfg.get("mesh_metrics_file")
                or output_cfg.get("dtu_mesh_metrics_file")
                or output
            )
        samples = int(samples if samples is not None else eval_cfg.get("samples", 500_000))
        downsample_density = float(
            downsample_density
            if downsample_density is not None
            else eval_cfg.get("downsample_density", 0.2)
        )
        max_dist = float(max_dist if max_dist is not None else eval_cfg.get("max_dist", 20.0))
        cull_masks = bool(cull_masks if cull_masks is not None else eval_cfg.get("cull_masks", True))
    if pred is None or dtu_root is None or scan_id is None:
        raise typer.BadParameter("Use --config, or provide --pred, --dtu-root, and --scan-id.")

    samples = 500_000 if samples is None else samples
    downsample_density = 0.2 if downsample_density is None else downsample_density
    max_dist = 20.0 if max_dist is None else max_dist
    cull_masks = True if cull_masks is None else cull_masks
    metrics = dtu_mesh_metrics(
        pred,
        dtu_root,
        scan_id,
        scene_root=scene_root,
        num_samples=samples,
        downsample_density=downsample_density,
        max_dist=max_dist,
        cull_masks=cull_masks,
    )
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))


def _infer_pred_mesh(cfg) -> str | None:
    mesh_cfg = section(cfg, "mesh")
    output_cfg = section(cfg, "output")
    value = mesh_cfg.get("output") or output_cfg.get("mesh_file")
    if value is None and output_cfg.get("dir") is not None:
        value = Path(output_cfg["dir"]) / "mesh.ply"
    return str(value) if value is not None else None


def _infer_dtu_eval_target(cfg) -> tuple[str | None, str | None]:
    dataset_cfg = section(cfg, "dataset")
    scene = dataset_cfg.get("scene")
    root = dataset_cfg.get("root") or dataset_cfg.get("dataset_path")
    dataset_type = str(dataset_cfg.get("type", dataset_cfg.get("dataset_type", ""))).lower()

    scan_id = str(scene) if scene is not None else None
    dtu_root = None
    if root is not None:
        root_path = Path(str(root)).expanduser()
        if scan_id is not None and root_path.name == Path(scan_id).name:
            dtu_root = str(root_path.parent)
        elif dataset_type == "dtu":
            dtu_root = str(root_path)

    if scan_id is None and root is not None and Path(str(root)).name.startswith("scan"):
        scan_id = Path(str(root)).name
    return dtu_root, scan_id


def _infer_dtu_scene_root(cfg) -> str | None:
    dataset_cfg = section(cfg, "dataset")
    root = dataset_cfg.get("root") or dataset_cfg.get("dataset_path")
    if root is None:
        return None
    root_path = Path(str(root)).expanduser()
    if root_path.name.startswith("scan") or (root_path / "cameras.npz").exists():
        return str(root_path)
    scene = dataset_cfg.get("scene")
    if scene is not None:
        candidate = root_path / str(scene)
        return str(candidate)
    return None
