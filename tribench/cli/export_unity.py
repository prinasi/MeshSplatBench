"""Export feature-preserving Unity-native TriAsset packages."""

from __future__ import annotations

from pathlib import Path

import typer


export_unity_app = typer.Typer(no_args_is_help=True)


@export_unity_app.callback(invoke_without_command=True)
def export_unity(
    method: str = typer.Option(..., "--method", "-m", help="2dts, triangle-splatting, mesh-splatting, or diffsoup."),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", exists=True, help="Checkpoint file or run directory."),
    output: Path = typer.Option(..., "--output", "-o", help="Output .triasset directory."),
    force: bool = typer.Option(False, "--force", help="Replace an existing output package."),
    mesh_opacity_floor: float | None = typer.Option(None, "--mesh-opacity-floor", help="Override MeshSplatting opacity floor for a legacy checkpoint."),
    export_topology: str = typer.Option(
        "indexed",
        "--export-topology",
        help="MeshSplatting export layout: indexed/mesh or soup/materialized-soup.",
    ),
    d2ts_gamma_rescale: bool | None = typer.Option(
        None,
        "--d2ts-gamma-rescale/--no-d2ts-gamma-rescale",
        help="Override the non-serialized 2DTS gamma-rescale flag for a legacy checkpoint.",
    ),
    background_color: str | None = typer.Option(
        None,
        "--background-color",
        help="Deterministic Unity evaluation background: black or white.",
    ),
):
    """Export full renderer inputs for a Unity-native method renderer.

    This does not produce a generic mesh.  Use it with the TriBench Unity
    package so view-dependent appearance, opacity, and neural features remain
    available to method-specific shaders.
    """

    from tribench.unity_assets import export_triasset

    try:
        package = export_triasset(
            method,
            checkpoint,
            output,
            overwrite=force,
            mesh_opacity_floor=mesh_opacity_floor,
            export_topology=export_topology,
            d2ts_gamma_rescale=d2ts_gamma_rescale,
            background_color=background_color,
        )
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Unity-native asset: {package.path}")
    typer.echo(f"Manifest: {package.manifest_path}")
    typer.echo(f"Method: {package.method}; buffers: {len(package.buffers)}")
