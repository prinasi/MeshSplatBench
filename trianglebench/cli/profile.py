"""Profile command: measure renderer performance."""

from pathlib import Path

import typer

profile_app = typer.Typer(no_args_is_help=True)


@profile_app.callback(invoke_without_command=True)
def profile(
    method: str = typer.Option(..., "--method", "-m", help="Method name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    repeats: int = typer.Option(100, "--repeats", "-r", help="Number of profiling repetitions"),
    output: str = typer.Option("profile.json", "--output", "-o", help="Output JSON file path"),
):
    """Profile renderer performance.
    
    Measures:
    - Forward pass latency (mean, std, p50, p95)
    - Backward pass latency (where available)
    - Peak CUDA memory usage
    - Rendering FPS
    """
    from trianglebench.core.registry import get_adapter

    typer.echo(f"Profiling method: {method}")
    typer.echo(f"Checkpoint: {checkpoint}")
    typer.echo(f"Dataset: {dataset} (split: {split})")
    typer.echo(f"Repeats: {repeats}")

    try:
        adapter_cls = get_adapter(method)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    adapter = adapter_cls()

    try:
        adapter.load_checkpoint(checkpoint)
        adapter.load_scene(dataset, split)
    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\nNote: Profiling requires a loaded model and CUDA device.")
    typer.echo("This command will be fully functional once adapters are implemented.")
