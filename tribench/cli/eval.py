"""Eval commands: image metrics and mesh Chamfer/DTU metrics."""

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


@eval_app.command("chamfer")
def chamfer(
    pred: Optional[str] = typer.Option(None, "--pred", help="Predicted mesh/point cloud"),
    gt: Optional[str] = typer.Option(None, "--gt", help="Ground-truth mesh/point cloud"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench evaluation config YAML"),
    output: str = typer.Option("mesh_metrics.json", "--output", "-o"),
    samples: int = typer.Option(500_000, "--samples"),
):
    """Compute symmetric Chamfer distance between mesh/point-cloud files."""
    from tribench.core.mesh_eval import chamfer_distance, write_mesh_metrics

    if config is not None:
        cfg = load_cli_config(config)
        chamfer_cfg = merge_dicts(section(cfg, "chamfer"), merged_section(cfg, "eval", nested="chamfer"))
        pred = pred or chamfer_cfg.get("pred")
        gt = gt or chamfer_cfg.get("gt")
        if output == "mesh_metrics.json":
            output = str(chamfer_cfg.get("output") or section(cfg, "output").get("mesh_metrics_file", output))
        samples = int(chamfer_cfg.get("samples", samples))
    if pred is None or gt is None:
        raise typer.BadParameter("Use --config, or provide both --pred and --gt.")

    metrics = chamfer_distance(pred, gt, num_samples=samples)
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))


@eval_app.command("dtu-mesh")
def dtu_mesh(
    pred: Optional[str] = typer.Option(None, "--pred", help="Predicted mesh/point cloud"),
    dtu_root: Optional[str] = typer.Option(None, "--dtu-root", help="Official DTU dataset root"),
    scan_id: Optional[str] = typer.Option(None, "--scan-id", help="DTU scan id, e.g. 24 or scan24"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench evaluation config YAML"),
    output: str = typer.Option("dtu_mesh_metrics.json", "--output", "-o"),
    samples: int = typer.Option(500_000, "--samples"),
):
    """Find DTU official GT for a scan and compute Chamfer metrics."""
    from tribench.core.mesh_eval import chamfer_distance, find_dtu_ground_truth, write_mesh_metrics

    if config is not None:
        cfg = load_cli_config(config)
        dtu_cfg = merge_dicts(section(cfg, "dtu_mesh"), merged_section(cfg, "eval", nested="dtu_mesh"))
        pred = pred or dtu_cfg.get("pred")
        dtu_root = dtu_root or dtu_cfg.get("dtu_root")
        scan_id = scan_id or dtu_cfg.get("scan_id")
        if output == "dtu_mesh_metrics.json":
            output = str(dtu_cfg.get("output") or section(cfg, "output").get("dtu_mesh_metrics_file", output))
        samples = int(dtu_cfg.get("samples", samples))
    if pred is None or dtu_root is None or scan_id is None:
        raise typer.BadParameter("Use --config, or provide --pred, --dtu-root, and --scan-id.")

    gt = find_dtu_ground_truth(dtu_root, scan_id)
    metrics = chamfer_distance(pred, gt, num_samples=samples)
    metrics["gt_path"] = str(gt)
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))
