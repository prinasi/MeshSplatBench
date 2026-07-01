"""Render commands: split renders, videos, viewer, and mesh export."""

from pathlib import Path
from typing import Optional

import typer

from tribench.cli.config import (
    adapter_config,
    dataset_config,
    load_cli_config,
    merged_section,
    output_dir as config_output_dir,
    section,
)

render_app = typer.Typer(no_args_is_help=True)
RESOLUTION_HELP = (
    "1/2/4/8 downscale factor; other positive values are target width; "
    "-1 caps large images"
)


def _load(method: str, checkpoint: str):
    from tribench.core.rendering import load_adapter

    return load_adapter(method, checkpoint)


def _load_dataset(
    dataset: str,
    dataset_type: str,
    split: str,
    eval_every: int,
    image_dir: str,
    resolution: int,
):
    from tribench.core.datasets import load_dataset

    return load_dataset(
        dataset,
        dataset_type=dataset_type,
        split=split,
        eval_every=eval_every,
        image_dir=image_dir,
        resolution=resolution,
    )


@render_app.callback()
def render():
    """Render images, videos, meshes, and online viewer previews."""


def _split_render_dir(run_dir: str | None, fallback: str, split: str) -> str:
    """Resolve the per-split render directory under ``<run_dir>/renders/<split>``."""
    if run_dir:
        return str(Path(run_dir) / "renders" / split)
    base = Path(fallback)
    # Keep a split subfolder so train/test renders never collide.
    if base.name == split:
        return str(base)
    return str(base / split)



@render_app.command("images")
def render_images(
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Method name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Path to dataset directory"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench render config YAML"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split: train/test/all"),
    output_dir: str = typer.Option("renders/", "--output-dir", "-o", help="Output directory"),
    dataset_type: str = typer.Option("auto", "--dataset-type", help="auto/colmap/mipnerf360/tanks/dtu/blender"),
    image_dir: str = typer.Option("images", "--image-dir", help="COLMAP image directory"),
    resolution: int = typer.Option(
        1,
        "--resolution",
        "-r",
        help=RESOLUTION_HELP,
    ),
    eval_every: int = typer.Option(8, "--eval-every", help="COLMAP/DTU holdout stride"),
    save_gt: bool = typer.Option(False, "--save-gt", help="Also save ground-truth images"),
    save_aux: bool = typer.Option(False, "--save-aux", help="Save depth/alpha when available"),
):
    """Render images from a trained model for one dataset split."""
    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.config import save_config_snapshot
    from tribench.core.rendering import render_dataset_split

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        render_cfg = merged_section(cfg, "render", nested="images")
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
            stage="render",
        )
        if "split" in render_cfg:
            dataset_cfg["split"] = render_cfg["split"]
        resolved_split = str(dataset_cfg.get("split", split))
        adapter = build_adapter(adapter_cfg)
        ds = build_dataset(dataset_cfg)
        run_dir = render_cfg.get("output_dir") or render_cfg.get("dir") or config_output_dir(cfg)
        output_dir = _split_render_dir(run_dir, output_dir, resolved_split)
        save_gt = bool(render_cfg.get("save_gt", save_gt))
        save_aux = bool(render_cfg.get("save_aux", save_aux))
        save_config_snapshot(cfg, config_output_dir(cfg, output_dir))
    else:
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        adapter = _load(method, checkpoint)
        ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
        output_dir = _split_render_dir(None, output_dir, split)
    manifest = render_dataset_split(adapter, ds, output_dir, device=adapter.device, save_gt=save_gt, save_aux=save_aux)
    typer.echo(f"Rendered {manifest['num_frames']} frames to {output_dir}")



@render_app.command("split")
def render_split(
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench render config YAML"),
    split: str = typer.Option("test", "--split", "-s"),
    output_dir: str = typer.Option("renders/", "--output-dir", "-o"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    save_gt: bool = typer.Option(True, "--save-gt/--no-save-gt"),
    save_aux: bool = typer.Option(False, "--save-aux"),
    metrics: bool = typer.Option(False, "--metrics"),
):
    """Render train/test/all images, optionally computing image metrics."""
    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.config import save_config_snapshot
    from tribench.core.rendering import render_dataset_split

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        render_cfg = merged_section(cfg, "render", nested="split")
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
            stage="render",
        )
        if "split" in render_cfg:
            dataset_cfg["split"] = render_cfg["split"]
        adapter = build_adapter(adapter_cfg)
        ds = build_dataset(dataset_cfg)
        output_dir = str(render_cfg.get("output_dir") or render_cfg.get("dir") or config_output_dir(cfg, output_dir))
        save_gt = bool(render_cfg.get("save_gt", save_gt))
        save_aux = bool(render_cfg.get("save_aux", save_aux))
        metrics = bool(render_cfg.get("metrics", metrics))
        save_config_snapshot(cfg, output_dir)
    else:
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        adapter = _load(method, checkpoint)
        ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    manifest = render_dataset_split(
        adapter,
        ds,
        output_dir,
        device=adapter.device,
        save_gt=save_gt,
        save_aux=save_aux,
        metrics=metrics,
    )
    typer.echo(f"Rendered {manifest['num_frames']} frames to {output_dir}")


@render_app.command("video")
def render_video_cmd(
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench render config YAML"),
    split: str = typer.Option("train", "--split", "-s"),
    output_dir: str = typer.Option("video/", "--output-dir", "-o"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    frames: int = typer.Option(240, "--frames"),
    fps: int = typer.Option(30, "--fps"),
    write_frames: bool = typer.Option(
        False,
        "--write-frames/--no-write-frames",
        help="Also save every rendered video frame as a PNG sequence.",
    ),
    zoom: float = typer.Option(1.0, "--zoom"),
    z_variation: float = typer.Option(0.0, "--z-variation"),
    z_phase: float = typer.Option(0.0, "--z-phase", help="Phase offset for vertical oscillation [0, 1]"),
):
    """Render a PCA-aligned ellipse trajectory video from dataset cameras."""
    from tribench.core.builder import build_adapter, build_dataset
    from tribench.core.config import save_config_snapshot
    from tribench.core.rendering import generate_ellipse_cameras, render_video

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        video_cfg = merged_section(cfg, "render", nested="video")
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
            raise typer.BadParameter(
                "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
            )
        dataset_cfg = dataset_config(
            cfg,
            dataset=dataset,
            dataset_type=dataset_type if dataset_type != "auto" else None,
            split=split if split != "train" else None,
            image_dir=image_dir if image_dir != "images" else None,
            resolution=resolution if resolution != 1 else None,
            eval_every=eval_every if eval_every != 8 else None,
            stage="render",
        )
        if "split" in video_cfg:
            dataset_cfg["split"] = video_cfg["split"]
        frames = int(video_cfg.get("frames", frames))
        fps = int(video_cfg.get("fps", fps))
        write_frames = bool(video_cfg.get("write_frames", write_frames))
        zoom = float(video_cfg.get("zoom", zoom))
        z_variation = float(video_cfg.get("z_variation", z_variation))
        z_phase = float(video_cfg.get("z_phase", z_phase))
        run_dir = config_output_dir(cfg)
        output_dir = str(
            video_cfg.get("output_dir")
            or video_cfg.get("dir")
            or (Path(run_dir) / "video" if run_dir and output_dir == "video/" else output_dir)
        )
        adapter = build_adapter(adapter_cfg)
        ds = build_dataset(dataset_cfg)
        save_config_snapshot(cfg, output_dir)
    else:
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        adapter = _load(method, checkpoint)
        ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    cameras = generate_ellipse_cameras(
        ds.get_all_cameras(),
        n_frames=frames,
        zoom=zoom,
        z_variation=z_variation,
        z_phase=z_phase,
    )
    video_path = render_video(
        adapter,
        cameras,
        output_dir,
        fps=fps,
        write_frames=write_frames,
    )
    typer.echo(f"Video saved to {video_path}")


@render_app.command("viewer")
def viewer(
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench viewer config YAML"),
    split: str = typer.Option("test", "--split", "-s"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7007, "--port"),
    max_render_size: int = typer.Option(
        1280,
        "--max-render-size",
        help="Maximum long edge for interactive renders; use 0 for no cap.",
    ),
    jpeg_quality: int = typer.Option(90, "--jpeg-quality", help="JPEG quality for RGB renders."),
    viewer_splits: str = typer.Option("train,test", "--viewer-splits", help="Comma-separated splits to load."),
    geometry: Optional[str] = typer.Option(None, "--geometry", help="Path to viewer point cloud PLY."),
    auto_export_geometry: bool = typer.Option(
        True,
        "--auto-export-geometry/--no-auto-export-geometry",
        help="Auto-export viewer point cloud if it is not already available.",
    ),
):
    """Start an interactive nerfbaselines-compatible viewer."""
    from tribench.core.viewer import serve_viewer

    splits = [s.strip() for s in viewer_splits.split(",") if s.strip()]

    if config is not None:
        from tribench.core.builder import build_adapter, build_dataset
        cfg = load_cli_config(config)
        assert cfg is not None
        viewer_cfg = merged_section(cfg, "render", nested="viewer")
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
            raise typer.BadParameter(
                "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
            )
        # Build datasets for all requested splits
        train_ds = None
        test_ds = None
        for sp in splits:
            dataset_cfg = dataset_config(
                cfg,
                dataset=dataset,
                dataset_type=dataset_type if dataset_type != "auto" else None,
                split=sp,
                image_dir=image_dir if image_dir != "images" else None,
                resolution=resolution if resolution != 1 else None,
                eval_every=eval_every if eval_every != 8 else None,
                stage="render",
            )
            ds = build_dataset(dataset_cfg)
            if sp == "train":
                train_ds = ds
            else:
                test_ds = ds
        host = str(viewer_cfg.get("host", host))
        port = int(viewer_cfg.get("port", port))
        max_render_size = int(viewer_cfg.get("max_render_size", max_render_size))
        jpeg_quality = int(viewer_cfg.get("jpeg_quality", jpeg_quality))
        geometry = _optional_viewer_path(viewer_cfg, "geometry", geometry)
        auto_export_geometry = bool(
            viewer_cfg.get("auto_export_geometry", auto_export_geometry)
        )
        default_geometry, default_metadata = _default_viewer_pointcloud_paths(cfg)
        geometry = geometry or default_geometry
        # Try to build adapter, fallback to DummyRenderer if CUDA unavailable
        adapter = _try_build_adapter(adapter_cfg, method)
        run_dir = config_output_dir(cfg)
        geometry_output_dir = Path(run_dir) / "viewer" if run_dir else None
        method_name = str(adapter_cfg.get("type", method or "unknown"))
        checkpoint_path = str(adapter_cfg.get("checkpoint"))
    else:
        if method is None or checkpoint is None or dataset is None:
            raise typer.BadParameter(
                "Use --config, or provide --method, --checkpoint, and --dataset."
            )
        train_ds = None
        test_ds = None
        for sp in splits:
            ds = _load_dataset(dataset, dataset_type, sp, eval_every, image_dir, resolution)
            if sp == "train":
                train_ds = ds
            else:
                test_ds = ds
        adapter = _try_load_adapter(method, checkpoint)
        default_metadata = None
        geometry_output_dir = None
        method_name = method
        checkpoint_path = checkpoint

    metadata_path = default_metadata
    if geometry is None and auto_export_geometry:
        geometry, metadata_path = _maybe_auto_export_viewer_geometry(
            adapter,
            method_name,
            checkpoint_path,
            output_dir=geometry_output_dir,
        )

    serve_viewer(
        adapter,
        train_dataset=train_ds,
        test_dataset=test_ds,
        host=host,
        port=port,
        max_render_size=max_render_size,
        jpeg_quality=jpeg_quality,
        geometry_path=geometry,
        metadata_path=metadata_path,
    )


def _optional_viewer_path(viewer_cfg: dict, name: str, current: str | None) -> str | None:
    value = viewer_cfg.get(name, current)
    return str(value) if value else None


def _default_viewer_pointcloud_paths(cfg) -> tuple[str | None, str | None]:
    """Find an already-exported viewer point cloud under ``<output.dir>/viewer``.

    This never uses the CG export. It only wires in the lightweight viewer
    point cloud that belongs to ``tribench render viewer``.
    """
    run_dir = config_output_dir(cfg)
    if run_dir is None:
        return None, None
    viewer_dir = Path(run_dir) / "viewer"
    pointcloud = viewer_dir / "geometry_viewer_points.ply"
    metadata = viewer_dir / "geometry_viewer_metadata.json"
    return (
        str(pointcloud) if pointcloud.exists() else None,
        str(metadata) if metadata.exists() else None,
    )


def _maybe_auto_export_viewer_geometry(
    adapter,
    method: str,
    checkpoint: str | None,
    *,
    output_dir: Path | None = None,
) -> tuple[str | None, str | None]:
    """Auto-export only the lightweight point cloud needed by the viewer."""
    import tempfile

    try:
        primitive = adapter.to_primitive()
    except Exception:
        return None, None

    from tribench.core.viewer_geometry import export_viewer_point_cloud

    out_dir = output_dir or Path(tempfile.mkdtemp(prefix="tribench_viewer_geometry_"))
    try:
        result = export_viewer_point_cloud(
            primitive,
            out_dir,
            method_name=method,
            checkpoint_path=checkpoint,
        )
        typer.echo(f"[viewer] Auto-exported viewer point cloud: {result.viewer_ply}")
        return result.viewer_ply, result.metadata_path
    except Exception as e:
        typer.echo(f"[viewer] Viewer geometry export skipped: {e}")
        return None, None


def _try_build_adapter(adapter_cfg, method_hint: str | None = "unknown"):
    """Try to build an adapter; fall back to DummyRenderer on CUDA failure."""
    import torch
    from tribench.core.builder import build_adapter
    try:
        return build_adapter(adapter_cfg)
    except Exception as e:
        if not torch.cuda.is_available():
            typer.echo(f"[viewer] WARNING: CUDA not available, falling back to DummyRenderer. Error: {e}")
            from tribench.renderers.dummy import DummyRenderer
            dummy = DummyRenderer(method_name=method_hint or "unknown")
            cp = adapter_cfg.get("checkpoint") if isinstance(adapter_cfg, dict) else None
            if cp:
                dummy.load_checkpoint(str(cp))
            return dummy
        raise


def _try_load_adapter(method: str, checkpoint: str):
    """Try to load an adapter; fall back to DummyRenderer on CUDA failure."""
    import torch
    try:
        return _load(method, checkpoint)
    except Exception as e:
        if not torch.cuda.is_available():
            typer.echo(f"[viewer] WARNING: CUDA not available, falling back to DummyRenderer. Error: {e}")
            from tribench.renderers.dummy import DummyRenderer
            dummy = DummyRenderer(method_name=method)
            dummy.load_checkpoint(checkpoint)
            return dummy
        raise


@render_app.command("mesh")
def mesh(
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench render config YAML"),
    output: str = typer.Option("mesh.ply", "--output", "-o"),
):
    """Export the adapter primitive as a PLY mesh/triangle soup."""
    from tribench.core.builder import build_adapter
    from tribench.core.config import save_config_snapshot
    from tribench.core.mesh_eval import export_adapter_mesh

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        mesh_cfg = section(cfg, "mesh")
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
            raise typer.BadParameter(
                "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
            )
        if output == "mesh.ply":
            output = str(mesh_cfg.get("output") or section(cfg, "output").get("mesh_file", output))
        adapter = build_adapter(adapter_cfg)
        export_kwargs = _mesh_export_kwargs(
            cfg,
            mesh_cfg,
            method=str(adapter_cfg.get("type", "")),
        )
        save_config_snapshot(cfg, Path(output).parent)
    else:
        if method is None or checkpoint is None:
            raise typer.BadParameter("Use --config, or provide --method and --checkpoint.")
        adapter = _load(method, checkpoint)
        export_kwargs = {}
    path = export_adapter_mesh(adapter, output, **export_kwargs)
    typer.echo(f"Mesh saved to {path}")


@render_app.command("viewer-path")
def viewer_path(
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench render config YAML"),
    trajectory: Path = typer.Option(..., "--trajectory", "-t", help="Path to nerfbaselines-v1 trajectory JSON."),
    output_dir: str = typer.Option("trajectory_render/", "--output-dir", "-o"),
    output_types: str = typer.Option("color", "--output-types", help="Comma-separated: color,depth,alpha,normal"),
    device: Optional[str] = typer.Option(None, "--device", help="Override render device (e.g. cpu, cuda:0)."),
):
    """Render frames from a viewer-exported camera path (trajectory JSON).

    Loads a nerfbaselines-v1 trajectory JSON (exported from the viewer's
    ``save_trajectory()`` button), builds CameraBatch for each frame, and
    renders them offline. Produces per-frame images and an MP4 video.
    """
    from tribench.core.trajectory import (
        load_trajectory,
        trajectory_cameras,
        trajectory_get_fps,
        render_trajectory_frames,
    )

    traj = load_trajectory(trajectory)
    fps = trajectory_get_fps(traj)
    cameras = trajectory_cameras(traj, device=device)
    out_types = tuple(s.strip() for s in output_types.split(",") if s.strip())

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
        if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
            raise typer.BadParameter(
                "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
            )
        adapter = _try_build_adapter(adapter_cfg, method or "unknown")
    else:
        if method is None or checkpoint is None:
            raise typer.BadParameter("Use --config, or provide --method and --checkpoint.")
        adapter = _try_load_adapter(method, checkpoint)

    manifest = render_trajectory_frames(
        adapter,
        cameras,
        output_dir,
        fps=fps,
        output_types=out_types,
        device=device,
    )
    typer.echo(
        f"Rendered {manifest['num_frames']} frames at {fps} fps to {output_dir}"
    )
    if manifest.get("video_path"):
        typer.echo(f"Video: {manifest['video_path']}")


def _mesh_export_kwargs(cfg, mesh_cfg: dict, method: str | None = None) -> dict:
    try:
        dataset_cfg = dataset_config(
            cfg,
            split=str(mesh_cfg.get("split", "train")),
            stage=None,
        )
    except Exception:
        return {}

    dataset_path = dataset_cfg.get("root") or dataset_cfg.get("dataset_path")
    if dataset_path is None:
        return {}

    kwargs = {
        "dataset_path": str(dataset_path),
        "split": str(dataset_cfg.get("split", mesh_cfg.get("split", "train"))),
        "image_dir": str(dataset_cfg.get("image_dir", "images")),
        "resolution": int(dataset_cfg.get("resolution", 1)),
        "eval_every": int(dataset_cfg.get("eval_every", 8)),
        "voxel_size": float(mesh_cfg.get("voxel_size", 0.004)),
        "sdf_trunc": float(mesh_cfg.get("sdf_trunc", 0.016)),
        "depth_trunc": float(mesh_cfg.get("depth_trunc", 3.0)),
        "num_cluster": int(mesh_cfg.get("num_cluster", mesh_cfg.get("clusters", 1))),
        "depth_ratio": float(mesh_cfg.get("depth_ratio", 1.0)),
    }
    if str(method or "").replace("_", "-") == "mesh-splatting":
        kwargs["eval_split"] = bool(mesh_cfg.get("eval_split", False))
        kwargs["render_scaling"] = int(mesh_cfg.get("render_scaling", 1))
    return kwargs
