"""Tests for config-driven CLI command surfaces."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tribench.cli.main import app


def _unified_config(path: Path) -> Path:
    config = path / "experiment.yaml"
    config.write_text(
        "dataset:\n"
        "  type: colmap\n"
        "  root: data/bicycle\n"
        "  image_dir: images\n"
        "  resolution: 4\n"
        "  eval_every: 8\n"
        "trainer:\n"
        "  type: missing-method\n"
        "adapter:\n"
        "  type: missing-method\n"
        "  checkpoint: outputs/missing/bicycle\n"
        "output:\n"
        "  dir: outputs/missing/bicycle\n"
        "  metrics_file: outputs/missing/bicycle/metrics.json\n"
        "render:\n"
        "  split: test\n"
        "eval:\n"
        "  split: test\n"
    )
    return config


def test_eval_images_accepts_config_without_required_triple(tmp_path: Path):
    result = CliRunner().invoke(app, ["eval", "images", "--config", str(_unified_config(tmp_path))])

    assert result.exit_code != 2
    assert "Missing option" not in result.output


def test_render_subcommands_accept_config_without_required_triple(tmp_path: Path):
    config = _unified_config(tmp_path)

    for subcommand in ("images", "split", "video", "viewer"):
        result = CliRunner().invoke(app, ["render", subcommand, "--config", str(config)])
        assert result.exit_code != 2, subcommand
        assert "Missing option" not in result.output


def test_render_mesh_accepts_config_without_method_checkpoint(tmp_path: Path):
    result = CliRunner().invoke(app, ["render", "mesh", "--config", str(_unified_config(tmp_path))])

    assert result.exit_code != 2
    assert "Missing option" not in result.output


def test_mesh_eval_subcommands_accept_config_options(tmp_path: Path):
    config = tmp_path / "mesh_eval.yaml"
    config.write_text(
        "eval:\n"
        "  chamfer:\n"
        "    pred: missing_pred.ply\n"
        "    gt: missing_gt.ply\n"
        "    samples: 1\n"
        "  dtu_mesh:\n"
        "    pred: missing_pred.ply\n"
        "    dtu_root: missing_dtu\n"
        "    scan_id: 24\n"
        "    samples: 1\n"
    )

    for subcommand in ("chamfer", "dtu-mesh"):
        result = CliRunner().invoke(app, ["eval", subcommand, "--config", str(config)])
        assert result.exit_code != 2, subcommand
        assert "Missing option" not in result.output


def test_inspect_accepts_config(tmp_path: Path):
    result = CliRunner().invoke(app, ["inspect", "--config", str(_unified_config(tmp_path))])

    assert result.exit_code != 2
    assert "missing-method" in result.output


def test_compare_accepts_config_inputs(tmp_path: Path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"num_views": 1, "aggregated": {"psnr_mean": 20.0}}')
    report = tmp_path / "report.md"
    config = tmp_path / "compare.yaml"
    config.write_text(
        "compare:\n"
        f"  inputs:\n    - {metrics}\n"
        f"  output: {report}\n"
    )

    result = CliRunner().invoke(app, ["compare", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert report.exists()
