"""Eval commands: image metrics and mesh/DTU metrics."""

import json
from pathlib import Path
from typing import Mapping, Optional

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
    ste_threshold: Optional[float] = typer.Option(
        None, "--ste-threshold",
        help="Opacity STE binarisation threshold (e.g. 0.3 for DTU).",
    ),
    sort_level: Optional[int] = typer.Option(
        None, "--sort-level",
        help="Depth-sort granularity: 0=per-triangle, 1=per-tile, 2=per-pixel.",
    ),
    bg_color: Optional[str] = typer.Option(
        None, "--bg-color",
        help="Background colour for rendering: white or black.",
    ),
):
    """Evaluate rendering quality on a dataset split.

    Metrics are computed from the images already produced by ``render images``
    (``<run_dir>/renders/<split>``) so no images are re-rendered or duplicated.
    When those renders are missing, the split is rendered once into that same
    folder as a fallback.
    """
    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.config import save_config_snapshot
    from tribench.core.datasets import load_dataset
    from tribench.core.rendering import (
        compute_metrics_from_render_dir,
        load_adapter,
        render_dataset_split,
    )
    from tribench.core.runtime_stats import load_training_stats_for_checkpoint

    method_label = method
    checkpoint_label = checkpoint
    dataset_label = dataset
    adapter = None
    ds = None
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
        split = str(dataset_cfg.get("split", split))
        dataset_kind = str(dataset_cfg.get("type", dataset_cfg.get("dataset_type", ""))).lower()
        metrics_mode = str(dataset_cfg.get("dtu_eval_mode", "full")) if dataset_kind == "dtu" else None
        method_label = adapter_cfg.get("type")
        checkpoint_label = adapter_cfg.get("checkpoint")
        dataset_label = dataset_cfg.get("root", dataset_cfg.get("dataset_path"))
        eval_cfg = section(cfg, "eval")
        render_cfg = section(cfg, "render")
        if output == "metrics.json":
            output = metrics_file(cfg, default=output)
        run_dir = (
            render_cfg.get("output_dir")
            or render_cfg.get("dir")
            or output_dir(cfg)
        )
        render_dir = str(
            eval_cfg.get("render_dir")
            or (Path(run_dir) / "renders" / split if run_dir else Path(render_dir) / split)
        )
        save_config_snapshot(cfg, output_dir(cfg) or Path(output).parent)
    else:
        is_dtu_path = dataset is not None and (Path(dataset).expanduser() / "cameras.npz").is_file()
        metrics_mode = "full" if dataset_type == "dtu" or is_dtu_path else None
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        render_dir = str(Path(render_dir) / split)

    # Prefer metrics computed from existing renders; only render when missing.
    manifest = compute_metrics_from_render_dir(
        render_dir,
        expected_method=str(method_label) if method_label is not None else None,
        expected_checkpoint=checkpoint_label,
        expected_metrics_mode=metrics_mode,
    )
    if manifest is None:
        if config is not None:
            adapter = build_adapter(adapter_cfg)
            _apply_render_overrides(adapter, ste_threshold, sort_level, bg_color)
            ds = build_dataset(dataset_cfg)
        else:
            adapter = load_adapter(method, checkpoint)
            _apply_render_overrides(adapter, ste_threshold, sort_level, bg_color)
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
            device=adapter.device,
            save_gt=True,
            save_aux=False,
            metrics=True,
        )
    inference = manifest.get("timing", {})
    benchmark_inference = _benchmark_diffsoup_inference(
        method_label,
        adapter,
        ds,
        adapter_cfg if config is not None else None,
        dataset_cfg if config is not None else None,
        method=method,
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_type=dataset_type,
        image_dir=image_dir,
        resolution=resolution,
        eval_every=eval_every,
        bg_color=bg_color,
    )
    render_inference = inference
    if benchmark_inference is not None:
        inference = benchmark_inference
    inference_fps = inference.get("gpu_fps") or inference.get("fps")

    training = load_training_stats_for_checkpoint(checkpoint_label, output)
    metrics = {
        "method": method_label,
        "checkpoint": checkpoint_label,
        "dataset": dataset_label,
        "split": split,
        "num_views": manifest["num_frames"],
        "inference": inference,
        "inference_fps": inference_fps,
        "inference_time_s": inference.get("gpu_time_s") or inference.get("time_s"),
        "render_inference": render_inference if benchmark_inference is not None else None,
        "training": training,
        "training_time_s": training.get("total_time_s") if training else None,
        "training_peak_gpu_memory_mib": training.get("peak_gpu_memory_mib") if training else None,
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


def _benchmark_diffsoup_inference(
    method_label,
    adapter,
    dataset_obj,
    adapter_cfg: Mapping | None,
    dataset_cfg: Mapping | None,
    *,
    method: str | None,
    checkpoint: str | None,
    dataset: str | None,
    dataset_type: str,
    image_dir: str,
    resolution: int,
    eval_every: int,
    bg_color: str | None,
) -> dict | None:
    method_name = str(method_label or method or "").replace("_", "-").lower()
    if method_name != "diffsoup":
        return None

    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.datasets import load_dataset

    benchmark_adapter = adapter
    if benchmark_adapter is None:
        if adapter_cfg is not None:
            benchmark_adapter = build_adapter(adapter_cfg)
        elif method is not None and checkpoint is not None:
            benchmark_adapter = build_adapter({"type": method, "checkpoint": checkpoint})
        else:
            return None
        _apply_render_overrides(benchmark_adapter, None, None, bg_color)

    if not hasattr(benchmark_adapter, "benchmark_cameras"):
        return None

    benchmark_cameras = None
    benchmark_split = "all"
    if dataset_cfg is not None:
        dtype = str(dataset_cfg.get("type", dataset_cfg.get("dataset_type", "auto"))).lower()
        if dtype in {"auto", "colmap", "mipnerf360", "mip360", "tanks", "tanksandtemples", "tankstemple"}:
            benchmark_cameras = []
            for split_name in ("train", "test"):
                split_cfg = dict(dataset_cfg)
                split_cfg["split"] = split_name
                split_dataset = build_dataset(split_cfg)
                benchmark_cameras.extend(
                    split_dataset.sample(i).camera for i in range(len(split_dataset))
                )
            benchmark_split = "train test"
        else:
            benchmark_cfg = dict(dataset_cfg)
            benchmark_cfg["split"] = "all"
            benchmark_dataset = build_dataset(benchmark_cfg)
            benchmark_cameras = [
                benchmark_dataset.sample(i).camera
                for i in range(len(benchmark_dataset))
            ]
    elif dataset is not None:
        dtype = str(dataset_type).lower()
        if dtype in {"auto", "colmap", "mipnerf360", "mip360", "tanks", "tanksandtemples", "tankstemple"}:
            benchmark_cameras = []
            for split_name in ("train", "test"):
                split_dataset = load_dataset(
                    dataset,
                    dataset_type=dataset_type,
                    split=split_name,
                    eval_every=eval_every,
                    image_dir=image_dir,
                    resolution=resolution,
                )
                benchmark_cameras.extend(
                    split_dataset.sample(i).camera for i in range(len(split_dataset))
                )
            benchmark_split = "train test"
        else:
            benchmark_dataset = load_dataset(
                dataset,
                dataset_type=dataset_type,
                split="all",
                eval_every=eval_every,
                image_dir=image_dir,
                resolution=resolution,
            )
            benchmark_cameras = [
                benchmark_dataset.sample(i).camera
                for i in range(len(benchmark_dataset))
            ]
    elif dataset_obj is not None:
        benchmark_cameras = [
            dataset_obj.sample(i).camera
            for i in range(len(dataset_obj))
        ]
        benchmark_split = str(getattr(dataset_obj, "split", "dataset"))
    else:
        return None

    benchmark = benchmark_adapter.benchmark_cameras(benchmark_cameras, warmup=10, trials=5)
    benchmark["benchmark_split"] = benchmark_split
    return benchmark


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
    geometry_mode: Optional[str] = typer.Option(
        None,
        "--geometry-mode",
        help="DTU predicted geometry interpretation: auto, mesh, or pcd.",
    ),
    force_export: Optional[bool] = typer.Option(
        None,
        "--force-export/--no-force-export",
        help="Re-export config meshes before evaluating. Defaults to auto when checkpoint is newer.",
    ),
):
    """Evaluate a reconstructed mesh. DTU configs use the DTU mesh protocol."""
    from tribench.core.mesh_eval import dtu_mesh_metrics, write_mesh_metrics

    cfg = None
    scene_root = None
    dtu_eval_mode = None
    if config is not None:
        cfg = load_cli_config(config)
        dataset_cfg_for_mode = section(cfg, "dataset")
        dtu_eval_mode = dataset_cfg_for_mode.get("dtu_eval_mode")
        legacy_cfg = merge_dicts(section(cfg, "dtu_mesh"), merged_section(cfg, "eval", nested="dtu_mesh"))
        metric_mesh_cfg = merge_dicts(section(cfg, "mesh_eval"), merged_section(cfg, "eval", nested="mesh"))
        eval_cfg = merge_dicts(legacy_cfg, metric_mesh_cfg)
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
        geometry_mode = geometry_mode or eval_cfg.get("geometry_mode") or eval_cfg.get("mode")
    if pred is None or dtu_root is None or scan_id is None:
        raise typer.BadParameter("Use --config, or provide --pred, --dtu-root, and --scan-id.")

    samples = 500_000 if samples is None else samples
    downsample_density = 0.2 if downsample_density is None else downsample_density
    max_dist = 20.0 if max_dist is None else max_dist
    cull_masks = True if cull_masks is None else cull_masks
    geometry_mode = geometry_mode or "auto"
    if cfg is not None:
        pred = _ensure_pred_mesh_exists(
            cfg,
            pred,
            _mesh_export_config(cfg),
            force_export=force_export,
        )
    metrics = dtu_mesh_metrics(
        pred,
        dtu_root,
        scan_id,
        scene_root=scene_root,
        num_samples=samples,
        downsample_density=downsample_density,
        max_dist=max_dist,
        cull_masks=cull_masks,
        geometry_mode=geometry_mode,
    )
    if dtu_eval_mode is not None:
        metrics["dtu_eval_mode"] = str(dtu_eval_mode)
    write_mesh_metrics(metrics, output)
    typer.echo(json.dumps(metrics, indent=2))


def _ensure_pred_mesh_exists(
    cfg,
    pred: str,
    mesh_cfg: dict | None = None,
    *,
    force_export: bool | None = None,
) -> str:
    """Export the configured adapter mesh when absent or stale."""
    pred_path = Path(pred).expanduser()
    adapter_cfg = adapter_config(cfg)
    checkpoint = adapter_cfg.get("checkpoint")
    method = str(adapter_cfg.get("type", "")).replace("_", "-")
    should_export = force_export is True or not pred_path.exists()
    if pred_path.exists() and force_export is None and checkpoint is not None:
        checkpoint_mtime = _checkpoint_mtime(Path(str(checkpoint)).expanduser())
        should_export = checkpoint_mtime is not None and checkpoint_mtime > pred_path.stat().st_mtime
    if (
        pred_path.exists()
        and force_export is None
        and method == "mesh-splatting"
        and not _has_current_mesh_splatting_export_metadata(pred_path)
    ):
        should_export = True
    if pred_path.exists() and not should_export:
        return str(pred_path)

    if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
        return str(pred_path)

    from tribench.core.builder import build_adapter
    from tribench.core.mesh_eval import export_adapter_mesh
    from tribench.cli.render import _mesh_export_kwargs

    if pred_path.exists():
        typer.echo(f"Predicted mesh at {pred_path} is stale or from an older pipeline; re-exporting mesh.")
    else:
        typer.echo(f"Predicted mesh not found at {pred_path}; exporting mesh first.")
    adapter = build_adapter(adapter_cfg)
    export_kwargs = _mesh_export_kwargs(
        cfg,
        mesh_cfg or section(cfg, "mesh"),
        method=str(adapter_cfg.get("type", "")),
    )
    return str(export_adapter_mesh(adapter, pred_path, **export_kwargs))


def _has_current_mesh_splatting_export_metadata(pred_path: Path) -> bool:
    metadata_path = pred_path.with_suffix(pred_path.suffix + ".tribench.json")
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    export = metadata.get("tribench_mesh_export", metadata)
    return (
        export.get("method") == "mesh-splatting"
        and export.get("pipeline") == "native_tsdf"
        and int(export.get("version", 0)) >= 1
    )


def _checkpoint_mtime(path: Path) -> float | None:
    if path.is_dir():
        state_path = path / "point_cloud_state_dict.pt"
        if state_path.exists():
            return state_path.stat().st_mtime
    if path.exists():
        return path.stat().st_mtime
    return None


def _mesh_export_config(cfg) -> dict:
    """Return mesh-export options without inheriting top-level eval split."""
    mesh_cfg = section(cfg, "mesh")
    eval_cfg = section(cfg, "eval")
    eval_mesh_cfg = eval_cfg.get("mesh", {}) or {}
    if not isinstance(eval_mesh_cfg, Mapping):
        return mesh_cfg
    return merge_dicts(mesh_cfg, eval_mesh_cfg)


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


def _apply_render_overrides(
    adapter,
    ste_threshold: float | None,
    sort_level: int | None,
    bg_color: str | None,
) -> None:
    """Forward CLI render-param overrides to the adapter if it supports them."""
    if not hasattr(adapter, "configure"):
        return
    overrides = {}
    if ste_threshold is not None:
        overrides["ste_threshold"] = ste_threshold
    if sort_level is not None:
        overrides["sort_level"] = sort_level
    if bg_color is not None:
        overrides["bg_color"] = bg_color
    if overrides:
        adapter.configure(**overrides)
