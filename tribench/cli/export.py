"""Offline geometry export commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from tribench.cli.config import (
    adapter_config,
    load_cli_config,
    merged_section,
    output_dir as config_output_dir,
)

export_app = typer.Typer(no_args_is_help=True)

DEFAULT_OUTPUT_NAME = "mesh.ply"
DEFAULT_NUM_POINTS = 500_000
DEFAULT_COLOR_MODE = "dc"
DEFAULT_NORMAL_MODE = "none"
DEFAULT_VOXEL_SIZE = 0.0
COLOR_MODES = {"dc", "opacity", "white"}
NORMAL_MODES = {"none", "primitive"}


@export_app.callback()
def export():
    """Export trained geometry without starting interactive tools."""


@export_app.command("point-cloud")
def point_cloud(
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Method name"),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench config YAML"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output PLY path. Indexed primitives preserve their mesh faces; "
            "other primitives are surface-sampled (default: <config output.dir>/mesh.ply)"
        ),
    ),
    num_points: int = typer.Option(
        DEFAULT_NUM_POINTS,
        "--num-points",
        help="Number of surface points for non-indexed primitives",
    ),
    color_mode: str = typer.Option(
        DEFAULT_COLOR_MODE,
        "--color-mode",
        help="Vertex/point color mode: dc, opacity, or white",
    ),
    normal_mode: str = typer.Option(
        DEFAULT_NORMAL_MODE,
        "--normal-mode",
        help="Vertex/point normal mode: none or primitive",
    ),
    voxel_size: float = typer.Option(
        DEFAULT_VOXEL_SIZE,
        "--voxel-size",
        help="Voxel size for optional point-cloud downsampling; 0 disables downsampling",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for non-indexed primitive sampling"
    ),
):
    """Export a colored PLY, preserving indexed mesh topology when available."""
    import torch

    from tribench.core.builder import build_adapter
    from tribench.core.viewer_geometry import export_indexed_mesh_ply, export_point_cloud_ply
    from tribench.primitives.mesh_triangle import IndexedMeshTriangle

    if config is not None:
        cfg = load_cli_config(config)
        assert cfg is not None
        export_cfg = merged_section(cfg, "export", nested="point_cloud")
        if output is None and export_cfg.get("output") is not None:
            output = Path(str(export_cfg["output"]))
        if num_points == DEFAULT_NUM_POINTS and export_cfg.get("num_points") is not None:
            num_points = int(export_cfg["num_points"])
        if color_mode == DEFAULT_COLOR_MODE and export_cfg.get("color_mode") is not None:
            color_mode = str(export_cfg["color_mode"])
        if normal_mode == DEFAULT_NORMAL_MODE and export_cfg.get("normal_mode") is not None:
            normal_mode = str(export_cfg["normal_mode"])
        if voxel_size == DEFAULT_VOXEL_SIZE and export_cfg.get("voxel_size") is not None:
            voxel_size = float(export_cfg["voxel_size"])
        adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
    else:
        if method is None or checkpoint is None:
            raise typer.BadParameter("Use --config, or provide --method and --checkpoint.")
        adapter_cfg = adapter_config(None, method=method, checkpoint=checkpoint)

    if adapter_cfg.get("type") is None or adapter_cfg.get("checkpoint") is None:
        raise typer.BadParameter(
            "Config must set adapter.type and adapter.checkpoint, or trainer.type and output.dir."
        )
    if output is None:
        if cfg is None:
            raise typer.BadParameter("Provide --output, or use --config to infer an output path.")
        run_dir = config_output_dir(cfg)
        if run_dir is None:
            raise typer.BadParameter(
                "Config has no output.dir; pass --output or set export.point_cloud.output."
            )
        output = Path(run_dir) / DEFAULT_OUTPUT_NAME
    if num_points <= 0:
        raise typer.BadParameter("--num-points must be positive")
    if voxel_size < 0:
        raise typer.BadParameter("--voxel-size must be non-negative")
    if color_mode not in COLOR_MODES:
        raise typer.BadParameter(f"--color-mode must be one of: {', '.join(sorted(COLOR_MODES))}")
    if normal_mode not in NORMAL_MODES:
        raise typer.BadParameter(f"--normal-mode must be one of: {', '.join(sorted(NORMAL_MODES))}")

    adapter_type = str(adapter_cfg["type"]).replace("_", "-").lower()
    if adapter_type == "mesh-splatting":
        from tribench.renderers.mesh_splatting_adapter import (
            load_mesh_splatting_primitive_checkpoint,
        )

        primitive = load_mesh_splatting_primitive_checkpoint(adapter_cfg["checkpoint"])
    else:
        adapter = build_adapter(adapter_cfg)
        primitive = adapter.to_primitive()
    if isinstance(primitive, IndexedMeshTriangle):
        result = export_indexed_mesh_ply(
            primitive,
            output,
            color_mode=color_mode,
            normal_mode=normal_mode,
        )
        details = [f"{result.num_vertices} vertices", f"{result.num_faces} faces"]
        details.append("RGB" if result.has_rgb else "geometry only")
        if result.has_normals:
            details.append("normals")
        typer.echo(f"Mesh saved to {result.ply} ({', '.join(details)})")
        return

    generator = None
    if seed is not None:
        vertices = getattr(primitive, "vertices", None)
        device = vertices.device if isinstance(vertices, torch.Tensor) else torch.device("cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    result = export_point_cloud_ply(
        primitive,
        output,
        num_points=num_points,
        color_mode=color_mode,
        normal_mode=normal_mode,
        voxel_size=voxel_size,
        generator=generator,
    )
    details = [f"{result.num_points} vertices"]
    details.append("RGB" if result.has_rgb else "XYZ only")
    if result.has_normals:
        details.append("normals")
    typer.echo(f"Point cloud saved to {result.ply} ({', '.join(details)})")
