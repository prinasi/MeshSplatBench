"""Tests for config-driven CLI command surfaces."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tribench.cli.eval import (
    _ensure_pred_mesh_exists,
    _infer_dtu_eval_target,
    _infer_pred_mesh,
    _mesh_export_config,
)
from tribench.cli.train import _triangle_splatting_native_argv
from tribench.cli.main import app
from tribench.cli.render import _mesh_export_kwargs
from tribench.core.config import Config
from tribench.core.mesh_eval import export_adapter_mesh


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
    config = "configs/triangle-splatting/dtu/scan24.yaml"

    cfg = Config.fromfile(config)

    assert _infer_pred_mesh(cfg) == "outputs/triangle-splatting/dtu/scan24/fuse_post.ply"
    assert _infer_dtu_eval_target(cfg) == ("data/dtu", "scan24")


def test_triangle_splatting_native_argv_maps_dtu_config():
    cfg = Config.fromfile("configs/triangle-splatting/dtu/scan24.yaml")
    trainer_cfg = cfg.trainer.to_dict()
    dataset_cfg = cfg.dataset.to_dict()
    trainer_cfg["images"] = dataset_cfg["image_dir"]
    trainer_cfg["resolution"] = dataset_cfg["resolution"]
    trainer_cfg["eval_split"] = dataset_cfg["eval_split"]

    argv = _triangle_splatting_native_argv(
        trainer_cfg=trainer_cfg,
        dataset_root=dataset_cfg["root"],
        output_dir=Path(cfg.output.dir),
        max_steps=trainer_cfg["max_steps"],
        quiet=False,
    )

    assert argv[argv.index("-s") + 1] == "data/dtu/scan24"
    assert argv[argv.index("-m") + 1] == "outputs/triangle-splatting/dtu/scan24"
    assert argv[argv.index("-r") + 1] == "2"
    assert "--eval" in argv
    assert "--no_dome" in argv
    assert argv[argv.index("--max_shapes") + 1] == "500000"
    assert argv[argv.index("--lambda_opacity") + 1] == "0.0044"
    assert argv[argv.index("--importance_threshold") + 1] == "0.027"
    assert argv[argv.index("--test_iterations") + 1] == "-1"


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


def test_mesh_export_kwargs_resolve_dataset_and_mesh_options():
    cfg = Config.fromfile("configs/triangle-splatting/dtu/scan24.yaml")

    kwargs = _mesh_export_kwargs(cfg, cfg.mesh.to_dict())

    assert kwargs["dataset_path"] == "data/dtu/scan24"
    assert kwargs["split"] == "train"
    assert kwargs["resolution"] == 2
    assert kwargs["voxel_size"] == 0.004
    assert kwargs["sdf_trunc"] == 0.016
    assert kwargs["depth_trunc"] == 3.0
    assert kwargs["num_cluster"] == 1
    assert kwargs["depth_ratio"] == 1.0


def test_eval_split_does_not_override_mesh_export_split():
    cfg = Config(
        {
            "dataset": {
                "type": "dtu",
                "root": "data/dtu/scan24",
                "image_dir": "images",
                "resolution": 2,
                "eval_every": 8,
            },
            "mesh": {
                "split": "train",
                "voxel_size": 0.004,
            },
            "eval": {
                "split": "test",
                "mesh": {
                    "samples": 500000,
                },
            },
        }
    )

    mesh_cfg = _mesh_export_config(cfg)
    kwargs = _mesh_export_kwargs(cfg, mesh_cfg)

    assert mesh_cfg["split"] == "train"
    assert kwargs["split"] == "train"


def test_export_adapter_mesh_uses_native_export_when_context_is_available(tmp_path: Path):
    class Adapter:
        def export_mesh(self, path, **kwargs):
            self.path = Path(path)
            self.kwargs = kwargs
            self.path.write_text("ply\n")
            return self.path

    adapter = Adapter()
    output = tmp_path / "mesh.ply"

    assert export_adapter_mesh(adapter, output, dataset_path="data/dtu/scan24") == output
    assert output.exists()
    assert adapter.kwargs == {"dataset_path": "data/dtu/scan24"}


def test_eval_chamfer_command_is_removed():
    result = CliRunner().invoke(app, ["eval", "chamfer", "--config", "configs/triangle-splatting/dtu/scan24.yaml"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_eval_dtu_mesh_command_is_removed():
    result = CliRunner().invoke(
        app,
        ["eval", "dtu-mesh", "--config", "configs/triangle-splatting/dtu/scan24.yaml"],
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
