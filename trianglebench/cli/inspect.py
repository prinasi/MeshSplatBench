"""Inspect command: load a model and report statistics."""

from pathlib import Path

import typer

inspect_app = typer.Typer(no_args_is_help=True)


@inspect_app.callback(invoke_without_command=True)
def inspect(
    method: str = typer.Option(..., "--method", "-m", help="Method name (e.g., 2dts, triangle-splatting)"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    output: str = typer.Option("stats.json", "--output", "-o", help="Output JSON file path"),
):
    """Inspect a trained model and report statistics.
    
    Loads a model checkpoint and reports:
    - Primitive counts (triangles, vertices, faces)
    - Parameter counts
    - Geometry and opacity summaries
    - Backend-specific fields
    """
    from trianglebench.core.registry import get_adapter

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

    # Save to file
    import json
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    typer.echo(f"Statistics saved to {output_path}")

    # Print summary
    typer.echo("\n--- Model Statistics ---")
    for key, value in stats.items():
        typer.echo(f"  {key}: {value}")
