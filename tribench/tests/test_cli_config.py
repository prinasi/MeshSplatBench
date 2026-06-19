"""Tests for config-driven CLI command surfaces."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tribench.cli.eval import _ensure_pred_mesh_exists, _infer_dtu_eval_target, _infer_pred_mesh
from tribench.cli.main import app
from tribench.core.config import Config


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
        "  mesh:\n"
        "    pred: missing_pred.ply\n"
        "    dtu_root: missing_dtu\n"
        "    scan_id: 24\n"
        "    samples: 1\n"
    )

    result = CliRunner().invoke(app, ["eval", "mesh", "--config", str(config)])
    assert result.exit_code != 2
    assert "Missing option" not in result.output


def test_dtu_mesh_eval_can_infer_required_options_from_config():
    config = "configs/triangle-splatting/scan24.yaml"

    cfg = Config.fromfile(config)

    assert _infer_pred_mesh(cfg) == "outputs/triangle-splatting/dtu/scan24/mesh.ply"
    assert _infer_dtu_eval_target(cfg) == ("data/dtu", "scan24")


def test_eval_mesh_exports_missing_configured_mesh(tmp_path: Path, monkeypatch):
    config = Config(
        {
            "adapter": {
                "type": "triangle-splatting",
                "checkpoint": str(tmp_path / "checkpoint"),
            }
        }
    )
    pred = tmp_path / "mesh.ply"
    calls = {}

    def fake_build_adapter(adapter_cfg):
        calls["adapter_cfg"] = adapter_cfg
        return object()

    def fake_export_adapter_mesh(adapter, path):
        calls["export_path"] = Path(path)
        Path(path).write_text("ply\n")
        return Path(path)

    monkeypatch.setattr("tribench.core.builder.build_adapter", fake_build_adapter)
    monkeypatch.setattr("tribench.core.mesh_eval.export_adapter_mesh", fake_export_adapter_mesh)

    assert _ensure_pred_mesh_exists(config, str(pred)) == str(pred)
    assert pred.exists()
    assert calls["adapter_cfg"]["type"] == "triangle-splatting"
    assert calls["export_path"] == pred


def test_eval_mesh_missing_mesh_without_adapter_is_left_for_metrics(tmp_path: Path):
    pred = tmp_path / "mesh.ply"

    assert _ensure_pred_mesh_exists(Config({}), str(pred)) == str(pred)
    assert not pred.exists()


def test_eval_chamfer_command_is_removed():
    result = CliRunner().invoke(app, ["eval", "chamfer", "--config", "configs/triangle-splatting/scan24.yaml"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_eval_dtu_mesh_command_is_removed():
    result = CliRunner().invoke(
        app,
        ["eval", "dtu-mesh", "--config", "configs/triangle-splatting/scan24.yaml"],
    )

    assert result.exit_code != 0
    assert "No such command" in result.output


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
