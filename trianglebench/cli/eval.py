"""Eval commands: image metrics and mesh Chamfer/DTU metrics."""

import json

import typer

eval_app = typer.Typer(no_args_is_help=True)


@eval_app.callback()
def evaluate():
    """Evaluate image renders and mesh/DTU geometry."""


@eval_app.command("images")
def evaluate_images(
    method: str = typer.Option(..., "--method", "-m", help="Method name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
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
    from pathlib import Path

    from trianglebench.core.datasets import load_dataset
    from trianglebench.core.rendering import load_adapter, render_dataset_split

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
        "method": method,
        "checkpoint": checkpoint,
        "dataset": dataset,
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
    pred: str = typer.Option(..., "--pred", help="Predicted mesh/point cloud"),
    gt: str = typer.Option(..., "--gt", help="Ground-truth mesh/point cloud"),
    output: str = typer.Option("mesh_metrics.json", "--output", "-o"),
    samples: int = typer.Option(500_000, "--samples"),
):
    """Compute symmetric Chamfer distance between mesh/point-cloud files."""
    from trianglebench.core.mesh_eval import chamfer_distance, write_mesh_metrics

    metrics = chamfer_distance(pred, gt, num_samples=samples)
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))


@eval_app.command("dtu-mesh")
def dtu_mesh(
    pred: str = typer.Option(..., "--pred", help="Predicted mesh/point cloud"),
    dtu_root: str = typer.Option(..., "--dtu-root", help="Official DTU dataset root"),
    scan_id: str = typer.Option(..., "--scan-id", help="DTU scan id, e.g. 24 or scan24"),
    output: str = typer.Option("dtu_mesh_metrics.json", "--output", "-o"),
    samples: int = typer.Option(500_000, "--samples"),
):
    """Find DTU official GT for a scan and compute Chamfer metrics."""
    from trianglebench.core.mesh_eval import chamfer_distance, find_dtu_ground_truth, write_mesh_metrics

    gt = find_dtu_ground_truth(dtu_root, scan_id)
    metrics = chamfer_distance(pred, gt, num_samples=samples)
    metrics["gt_path"] = str(gt)
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))
