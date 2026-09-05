"""MeshSplatBench CLI entry point."""

import typer

from msbench.cli.inspect import inspect_app
from msbench.cli.profile import profile_app
from msbench.cli.eval import eval_app
from msbench.cli.compare import compare_app
from msbench.cli.train import TRAIN_CONTEXT_SETTINGS, train
from msbench.cli.render import render_app
from msbench.cli.export import export_app
from msbench.cli.export_unity import export_unity_app

app = typer.Typer(
    name="msbench",
    help="MeshSplatBench: a config-driven benchmark for triangle/splatting reconstruction.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.add_typer(inspect_app, name="inspect", help="Inspect a model and report statistics.")
app.add_typer(profile_app, name="profile", help="Profile renderer performance.")
app.add_typer(eval_app, name="eval", help="Evaluate rendering quality.")
app.add_typer(compare_app, name="compare", help="Compare results across methods.")
app.command(
    name="train",
    help="Train a triangle splatting model.",
    context_settings=TRAIN_CONTEXT_SETTINGS,
)(train)
app.add_typer(export_app, name="export", help="Export trained geometry assets.")
app.add_typer(render_app, name="render", help="Render images from a trained model.")
app.add_typer(export_unity_app, name="export-unity", help="Export feature-preserving Unity-native assets.")


@app.command()
def version():
    """Show MeshSplatBench version."""
    from msbench import __version__
    typer.echo(f"MeshSplatBench v{__version__}")


@app.command()
def backends():
    """Show external renderer backend availability."""
    import json

    from msbench.renderers.backends import BACKENDS

    typer.echo(json.dumps(
        {name: spec.status() for name, spec in BACKENDS.items()},
        indent=2,
    ))


if __name__ == "__main__":
    app()
