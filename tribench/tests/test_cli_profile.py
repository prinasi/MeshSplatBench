"""Tests for the profile CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tribench.cli.profile import _resolve_scene_path
from tribench.cli.main import app


def test_profile_accepts_scene_and_config_options(tmp_path: Path):
    config = tmp_path / "2dts.yaml"
    config.write_text(
        "dataset:\n"
        "  root: data/mipnerf360\n"
        "adapter:\n"
        "  type: missing-method\n"
        "  checkpoint: outputs/2dts/{scene}\n"
    )

    result = CliRunner().invoke(
        app,
        ["profile", "--method", "missing-method", "--scene", "garden", "--config", str(config)],
    )

    assert result.exit_code != 2
    assert "No such option: --scene" not in result.output
    assert "missing-method" in result.output


def test_profile_infers_checkpoint_from_train_output_dir(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "dataset:\n"
        "  root: data/bicycle\n"
        "trainer:\n"
        "  type: missing-method\n"
        "output:\n"
        "  dir: outputs/triangle-splatting/bicycle\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "profile",
            "--method",
            "missing-method",
            "--scene",
            "garden",
            "--config",
            str(config),
        ],
    )

    assert result.exit_code != 2
    assert "Use --checkpoint" not in result.output
    assert "Checkpoint: outputs/triangle-splatting/garden" in result.output
    assert "Dataset: data/garden" in result.output


def test_resolve_scene_path_replaces_scene_component():
    assert (
        _resolve_scene_path("outputs/triangle-splatting/bicycle/profile.json", "garden")
        == "outputs/triangle-splatting/garden/profile.json"
    )
