"""TriBench CLI entry point."""

import typer

from tribench.cli.inspect import inspect_app
from tribench.cli.profile import profile_app
from tribench.cli.eval import eval_app
from tribench.cli.compare import compare_app
from tribench.cli.train import TRAIN_CONTEXT_SETTINGS, train
from tribench.cli.render import render_app

app = typer.Typer(
    name="tribench",
    help="TriBench: a config-driven benchmark for triangle/splatting reconstruction.",
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
app.add_typer(render_app, name="render", help="Render images from a trained model.")


@app.command()
def version():
    """Show TriBench version."""
    from tribench import __version__
    typer.echo(f"TriBench v{__version__}")


@app.command()
def backends():
    """Show external renderer backend availability."""
    import json

    from tribench.renderers.backends import BACKENDS

    typer.echo(json.dumps(
        {name: spec.status() for name, spec in BACKENDS.items()},
        indent=2,
    ))


if __name__ == "__main__":
    app()
