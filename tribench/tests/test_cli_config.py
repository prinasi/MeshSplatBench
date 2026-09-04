"""Tests for config-driven CLI command surfaces."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tribench.cli.eval import (
    _ensure_pred_mesh_exists,
    _infer_dtu_eval_target,
    _infer_pred_mesh,
    _mesh_export_config,
)
from tribench.cli.train import _mesh_splatting_native_argv, _triangle_splatting_native_argv
from tribench.cli.main import app
from tribench.cli.render import _default_viewer_pointcloud_paths, _mesh_export_kwargs
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


def test_render_viewer_help_uses_explicit_geometry_only():
    result = CliRunner().invoke(app, ["render", "viewer", "--help"])

    assert result.exit_code == 0
    assert "--geometry" in result.output
    assert "Auto-export viewer" in result.output
    assert "--cg-geometry" not in result.output
    assert "--mesh-geometry" not in result.output


def test_render_viewer_discovers_existing_viewer_pointcloud(tmp_path: Path):
    viewer_dir = tmp_path / "run" / "viewer"
    viewer_dir.mkdir(parents=True)
    pointcloud = viewer_dir / "geometry_viewer_points.ply"
    metadata = viewer_dir / "geometry_viewer_metadata.json"
    pointcloud.write_text("ply\n")
    metadata.write_text("{}\n")

    cfg = {
        "output": {
            "dir": str(tmp_path / "run"),
        },
    }

    geometry, geometry_metadata = _default_viewer_pointcloud_paths(cfg)

    assert geometry == str(pointcloud)
    assert geometry_metadata == str(metadata)


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
        "    geometry_mode: pcd\n"
    )

    result = CliRunner().invoke(app, ["eval", "mesh", "--config", str(config)])
    assert result.exit_code != 2
    assert "Missing option" not in result.output


def test_dtu_mesh_eval_can_infer_required_options_from_config():
    config = "configs/triangle-splatting/dtu/scan24.yaml"

    cfg = Config.fromfile(config)

    assert _infer_pred_mesh(cfg) == "outputs/triangle-splatting/dtu-fg/scan24/mesh/fuse_post.ply"
    assert _infer_dtu_eval_target(cfg) == ("data/dtu", "scan24")

    assert cfg.adapter.render_params.bg_color == "white"


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
    assert argv[argv.index("-m") + 1] == "outputs/triangle-splatting/dtu-fg/scan24"
    assert argv[argv.index("-r") + 1] == "2"
    assert "--eval" in argv
    assert "--no_dome" in argv
    assert argv[argv.index("--max_shapes") + 1] == "500000"
    assert argv[argv.index("--lambda_opacity") + 1] == "0.0044"
    assert argv[argv.index("--importance_threshold") + 1] == "0.027"
    assert argv[argv.index("--test_iterations") + 1] == "-1"
    assert "--white_background" in argv
    assert "--foreground_training" not in argv


def test_triangle_splatting_native_argv_maps_max_primitives_to_max_shapes():
    argv = _triangle_splatting_native_argv(
        trainer_cfg={"max_primitives": 15_000},
        dataset_root="data/scene",
        output_dir=Path("outputs/scene"),
        max_steps=30_000,
        quiet=True,
    )

    assert argv[argv.index("--max_shapes") + 1] == "15000"
    assert argv[argv.index("--max_primitives") + 1] == "15000"


def test_mesh_splatting_native_argv_excludes_max_primitives():
    argv = _mesh_splatting_native_argv(
        trainer_cfg={"max_primitives": 15_000},
        dataset_root="data/scene",
        output_dir=Path("outputs/scene"),
        max_steps=30_000,
        quiet=True,
    )

    assert "--max_primitives" not in argv


def test_2dts_config_resolves_native_training_paths():
    cfg = Config.fromfile("configs/2dts/dtu/scan24.yaml")

    assert cfg.trainer.type == "2dts"
    assert cfg.adapter.checkpoint == "outputs/2dts/dtu-fg/scan24/ckpt"
    assert cfg.dataset.root == "data/dtu/scan24"
    assert cfg.dataset.resolution == 2
    assert cfg.d2ts.native_config == "configs/2dts/native/dtu.yaml"
    assert cfg.d2ts.target_point_num == 1_000_000
    assert cfg.eval.mesh.geometry_mode == "pcd"
    assert cfg.output.mesh_file == "outputs/2dts/dtu-fg/scan24/mesh/30000_pcd.ply"


def test_2dts_dtu_foreground_keeps_native_full_image_mask_defaults():
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    cfg = Config.fromfile("configs/2dts/dtu/scan24.yaml")
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert native.dataset.dtu_eval_mode == "foreground"
    assert native.dataset.dtu_use_alpha is True
    assert native.trainer.train_alpha_mask is False
    assert native.trainer.eval_alpha_mask is False


def test_2dts_native_config_sets_hard_primitive_cap(tmp_path: Path):
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    base_config = Path("configs/2dts/mipnerf360/garden.yaml").resolve()
    config = tmp_path / "garden_cap.yaml"
    config.write_text(
        f"_base_: {base_config}\n"
        "trainer:\n"
        "  max_primitives: 15000\n"
        "d2ts:\n"
        "  target_point_num: 1000000\n"
    )

    cfg = Config.fromfile(config)
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert native.model.model_update.max_primitives == 15_000
    assert native.model.model_update.densification.target_point_num == 15_000



def test_2dts_native_config_auto_resumes_from_latest_checkpoint(tmp_path: Path):
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    base_config = Path("configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml").resolve()
    config = tmp_path / "bicycle_resume.yaml"
    config.write_text(f"_base_: {base_config}\n")

    ckpt_dir = tmp_path / "outputs" / "opaque_2dts_mipnerf360" / "2dts" / "mipnerf360" / "bicycle" / "ckpt"
    (ckpt_dir / "point_cloud").mkdir(parents=True)
    (ckpt_dir / "point_cloud" / "1500.ply").write_text("ply\n")
    (ckpt_dir / "1200.ckpt").write_bytes(b"checkpoint")

    cfg = Config.fromfile(config)
    run_output_dir = tmp_path / "outputs" / "opaque_2dts_mipnerf360" / "2dts" / "mipnerf360" / "bicycle"
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=run_output_dir,
        max_steps=2000,
    )

    assert native.trainer.start_checkpoint is None
    assert native.trainer.start_pointcloud == 1500



def test_2dts_native_config_skips_resume_when_checkpoint_reaches_max_steps(tmp_path: Path):
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    base_config = Path("configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml").resolve()
    config = tmp_path / "bicycle_done.yaml"
    config.write_text(f"_base_: {base_config}\n")

    ckpt_dir = tmp_path / "outputs" / "opaque_2dts_mipnerf360" / "2dts" / "mipnerf360" / "bicycle" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "3000.ckpt").write_bytes(b"checkpoint")

    cfg = Config.fromfile(config)
    run_output_dir = tmp_path / "outputs" / "opaque_2dts_mipnerf360" / "2dts" / "mipnerf360" / "bicycle"
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=run_output_dir,
        max_steps=3000,
    )

    assert native.trainer.start_checkpoint is None
    assert native.trainer.start_pointcloud is None


def test_2dts_dtu_full_disables_native_alpha_loading(tmp_path: Path):
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    config = tmp_path / "scan24_full.yaml"
    base_config = Path("configs/2dts/dtu/scan24.yaml").resolve()
    config.write_text(
        f"_base_: {base_config}\n"
        "dataset:\n"
        "  dtu_eval_mode: full\n"
    )

    cfg = Config.fromfile(config)
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert cfg.dataset.dtu_eval_mode == "full"
    assert "dtu_use_alpha" not in cfg.dataset
    assert cfg.adapter.render_params.bg_color == "black"
    assert cfg.output.dir == "outputs/2dts/dtu-full/scan24"
    assert native.dataset.dtu_eval_mode == "full"
    assert native.dataset.dtu_use_alpha is False
    assert native.trainer.train_alpha_mask is False
    assert native.trainer.eval_alpha_mask is False


def test_2dts_disables_native_training_eval_by_default():
    cfg = Config.fromfile("configs/2dts/mipnerf360/bicycle.yaml")

    assert cfg.d2ts.native_overrides.trainer.eval_interval_iter == 0



def test_structured_train_cli_max_steps_overrides_config(tmp_path: Path):
    from tribench.cli.train import _train_from_structured_config

    cfg = Config.fromfile("configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml")

    captured = {}

    def fake_run_d2ts_native_config(*, cfg, dataset_root, output_dir, max_steps, quiet):
        captured["cfg"] = cfg
        captured["dataset_root"] = dataset_root
        captured["output_dir"] = output_dir
        captured["max_steps"] = max_steps
        captured["quiet"] = quiet
        return {
            "total_steps": int(max_steps),
            "total_time_s": 0.0,
            "avg_step_time_ms": 0.0,
            "final_losses": {},
        }

    with patch("tribench.core.runtime_stats.run_with_training_stats", lambda fn, _: fn()):
        with patch("tribench.cli.train.run_with_training_stats", lambda fn, _: fn()):
            with patch("tribench.trainers.d2ts_native.run_d2ts_native_config", fake_run_d2ts_native_config):
                summary = _train_from_structured_config(
                    cfg,
                    output_dir=tmp_path,
                    quiet=True,
                    max_steps_override=20,
                )

    assert summary["total_steps"] == 20
    assert captured["max_steps"] == 20


def test_structured_train_cli_uses_configured_max_steps_unless_explicitly_overridden(
    tmp_path: Path,
):
    config = tmp_path / "diffsoup.yaml"
    run_dir = tmp_path / "run"
    config.write_text(
        "dataset:\n"
        f"  root: {tmp_path / 'dataset'}\n"
        "  type: colmap\n"
        "trainer:\n"
        "  type: diffsoup\n"
        "  max_steps: 10000\n"
        "output:\n"
        f"  dir: {run_dir}\n"
    )
    captured_steps = []

    def fake_run_diffsoup_native_config(**kwargs):
        captured_steps.append(kwargs["max_steps"])
        return {
            "total_steps": kwargs["max_steps"],
            "total_time_s": 0.0,
            "avg_step_time_ms": 0.0,
            "final_losses": {},
        }

    with patch("tribench.cli.train.run_with_training_stats", lambda fn, _: fn()):
        with patch(
            "tribench.trainers.diffsoup_native.run_diffsoup_native_config",
            fake_run_diffsoup_native_config,
        ):
            configured = CliRunner().invoke(app, ["train", "--config", str(config)])
            overridden = CliRunner().invoke(
                app,
                ["train", "--config", str(config), "--max-steps", "12000"],
            )

    assert configured.exit_code == 0, configured.output
    assert overridden.exit_code == 0, overridden.output
    assert captured_steps == [10_000, 12_000]



def test_2dts_mipnerf360_opaque_experiment_native_overrides_match_dtu_style():
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    cfg = Config.fromfile("configs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle.yaml")
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert cfg.adapter.checkpoint == "outputs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle/ckpt"
    assert cfg.output.dir == "outputs/opaque_2dts_mipnerf360/2dts/mipnerf360/bicycle"
    assert cfg.d2ts.native_config == "configs/opaque_2dts_mipnerf360/2dts/native/mipnerf360.yaml"
    assert cfg.adapter.render_params.ste_threshold == 0.3
    assert cfg.adapter.render_params.sort_level == 2
    assert native.model.ste_threshold == 0.3
    assert native.model.sort_level == 2
    assert native.model.sampling.init_opacity == 0.5
    assert native.model.optimizer.opacity.v_final == 0.001
    assert native.model.model_update.contribution_pruning.contrib_sum_threshold == 0.1
    assert native.model.model_update.contribution_pruning.end_iter == 30_000
    assert native.model.model_update.scale_pruning.radii_threshold == -1
    assert native.model.model_update.scale_pruning.scale_threshold is None
    assert native.model.model_update.opacity_reset.end_iter == 15_000
    assert native.model.model_update.opacity_reset.reset_value == 0.29
    assert native.model.model_update.gamma_schedule.gamma_final == 50.0
    assert native.trainer.iterations == 30_000
    assert native.trainer.save_iterations == [30_000]
    assert native.trainer.checkpoint_iterations == [30_000]
    assert native.trainer.save_pcd_iterations == [30_000]
    assert native.trainer.pcd_n_sample == 5_000_000
    assert native.trainer.geometry_loss.w_geometry == 0.05
    assert native.trainer.distortion_loss.w_distortion == 0.05


def test_nerf_synthetic_config_matrix_resolves_for_all_methods():
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    methods = ["triangle-splatting", "mesh-splatting", "diffsoup", "2dts"]

    for method in methods:
        for scene in scenes:
            cfg = Config.fromfile(f"configs/{method}/nerf_synthetic/{scene}.yaml")

            assert cfg.dataset.root == f"data/nerf_synthetic/{scene}"
            assert cfg.dataset.type == "synthetic"
            assert cfg.dataset.background == "white"
            assert cfg.dataset.use_alpha is True
            assert cfg.output.dir == f"outputs/{method}/nerf_synthetic/{scene}"
            assert cfg.adapter.checkpoint.startswith(f"outputs/{method}/nerf_synthetic/{scene}/")

            if method in {"triangle-splatting", "mesh-splatting"}:
                assert cfg.trainer.white_background is True
                assert cfg.adapter.render_params.bg_color == "white"
            elif method == "diffsoup":
                assert "random_init" not in cfg.trainer
                assert "point_cloud_init" not in cfg.trainer
                assert "point_cloud" not in cfg.trainer
                assert cfg.trainer.mobilenerf_root == "data/mobilenerf_results"
                assert cfg.trainer.downscale == 1
                assert cfg.adapter.render_params.bg_color == "white"
            elif method == "2dts":
                assert cfg.d2ts.native_config == "configs/2dts/native/nerf_synthetic.yaml"
                assert cfg.adapter.render_params.bg_color == "white"


def test_triangle_and_mesh_splatting_nerf_synthetic_readers_prefer_points3d_ply():
    import inspect

    from tribench.vendor.mesh_splatting.scene.dataset_readers import (
        readNerfSyntheticInfo as read_mesh_synthetic,
    )
    from tribench.vendor.triangle_splatting.scene.dataset_readers import (
        readNerfSyntheticInfo as read_triangle_synthetic,
    )

    for reader in (read_triangle_synthetic, read_mesh_synthetic):
        source = inspect.getsource(reader)
        assert 'os.path.join(path, "points3d.ply")' in source
        assert "if not os.path.exists(ply_path):" in source
        assert "pcd = fetchPly(ply_path)" in source


def test_triangle_and_mesh_splatting_nerf_synthetic_readers_compose_rgba_as_uint8(tmp_path: Path):
    import json

    import numpy as np
    from PIL import Image

    from tribench.vendor.mesh_splatting.scene.dataset_readers import (
        readCamerasFromTransforms as read_mesh_cameras,
    )
    from tribench.vendor.triangle_splatting.scene.dataset_readers import (
        readCamerasFromTransforms as read_triangle_cameras,
    )

    scene_dir = tmp_path / "scene"
    image_dir = scene_dir / "train"
    image_dir.mkdir(parents=True)
    Image.fromarray(
        np.array([[[255, 0, 0, 128]]], dtype=np.uint8),
        mode="RGBA",
    ).save(image_dir / "r_0.png")
    (scene_dir / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": "./train/r_0",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        )
    )

    for read_cameras in (read_triangle_cameras, read_mesh_cameras):
        cameras = read_cameras(str(scene_dir), "transforms_train.json", True)

        assert cameras[0].image.mode == "RGB"
        assert np.asarray(cameras[0].image).dtype == np.uint8


def test_2dts_nerf_synthetic_native_config_uses_white_alpha_background():
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    cfg = Config.fromfile("configs/2dts/nerf_synthetic/lego.yaml")
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert native.dataset.type == "NerfSynthetic"
    assert native.dataset.local_dir == "data/nerf_synthetic"
    assert native.dataset.scene_id == "lego"
    assert native.dataset.background == "random"
    assert native.dataset.test_background == "white"
    assert native.dataset.pcd_path == "points3d.ply"
    assert native.dataset.dtu_use_alpha is True
    assert native.model.random_init is None
    assert native.model.optimizer.f_rest.v_init == 0.0002
    assert native.model.optimizer.f_rest.v_final == 0.0002
    assert native.model.model_update.densification.max_grow_ratio == 0.01
    assert native.model.model_update.scale_pruning.radii_threshold == -1
    assert native.model.model_update.scale_pruning.scale_threshold == 0.5
    assert native.trainer.train_background is None
    assert native.trainer.eval_background is None


def test_2dts_nerf_synthetic_random_init_is_only_a_pointcloud_fallback(tmp_path: Path):
    from tribench.trainers.d2ts_native import _build_d2ts_native_config

    base_config = Path("configs/2dts/nerf_synthetic/lego.yaml").resolve()
    config = tmp_path / "lego_no_pcd.yaml"
    config.write_text(
        f"_base_: {base_config}\n"
        "d2ts:\n"
        "  native_overrides:\n"
        "    dataset:\n"
        "      pcd_path: null\n"
    )

    cfg = Config.fromfile(config)
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=cfg.dataset.root,
        output_dir=Path(cfg.output.dir),
        max_steps=cfg.trainer.max_steps,
    )

    assert native.dataset.pcd_path is None
    assert native.model.random_init.bbox_list == [[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]]
    assert native.model.random_init.point_num_list == [100000]
    assert native.model.random_init.normal_list == ["random"]



def test_2dts_render_params_follow_native_dataset_configs():
    bicycle = Config.fromfile("configs/2dts/mipnerf360/bicycle.yaml")
    scan24 = Config.fromfile("configs/2dts/dtu/scan24.yaml")

    assert bicycle.dataset.resolution_rounding == "floor"
    assert bicycle.adapter.render_params.gamma_rescale is True
    assert bicycle.adapter.render_params.ste_threshold is None
    assert bicycle.adapter.render_params.sort_level == 0

    assert scan24.dataset.resolution_rounding == "floor"
    assert scan24.adapter.render_params.gamma_rescale is True
    assert scan24.adapter.render_params.ste_threshold == 0.3
    assert scan24.adapter.render_params.sort_level == 2


def test_mesh_splatting_render_scaling_matches_native_render_script():
    bicycle = Config.fromfile("configs/mesh-splatting/mipnerf360/bicycle.yaml")
    scan24 = Config.fromfile("configs/mesh-splatting/dtu/scan24.yaml")

    assert bicycle.adapter.render_params.render_scaling == 4
    assert scan24.adapter.render_params.render_scaling == 4
    assert scan24.adapter.render_params.bg_color == "white"
    assert scan24.mesh.eval_split is False
    assert scan24.mesh.render_scaling == 1


def test_mesh_splatting_mesh_export_kwargs_follow_native_mesh_script():
    from tribench.cli.render import _mesh_export_kwargs

    cfg = Config.fromfile("configs/mesh-splatting/dtu/scan24.yaml")
    kwargs = _mesh_export_kwargs(cfg, cfg.mesh.to_dict(), method="mesh-splatting")

    assert kwargs["dataset_path"] == "data/dtu/scan24"
    assert kwargs["voxel_size"] == 0.004
    assert kwargs["sdf_trunc"] == 0.016
    assert kwargs["depth_trunc"] == 3.0
    assert kwargs["num_cluster"] == 1
    assert kwargs["depth_ratio"] == 1.0
    assert kwargs["eval_split"] is False
    assert kwargs["render_scaling"] == 1


def test_2dts_vanilla_ts_import_does_not_require_gaussian_rasterizer():
    from tribench.vendor.d2ts.diff_recon import VanillaTSTrainer
    from tribench.vendor.d2ts.diff_recon.renderer import GaussianRenderer, HybridRenderer, TriangleRenderer

    assert VanillaTSTrainer.__name__ == "VanillaTSTrainer"
    assert TriangleRenderer.__name__ == "TriangleRenderer"
    assert GaussianRenderer is None or GaussianRenderer.__name__ == "GaussianRenderer"
    assert HybridRenderer is None or HybridRenderer.__name__ == "HybridRenderer"


def test_2dts_checkpoint_iterations_are_forced_for_final_step():
    from tribench.trainers.d2ts_native import _retarget_iteration_list

    class NativeTrainerConfig:
        checkpoint_iterations = []

    _retarget_iteration_list(NativeTrainerConfig, "checkpoint_iterations", 30_000, force=True)

    assert NativeTrainerConfig.checkpoint_iterations == [30_000]


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


def test_eval_mesh_reexports_existing_mesh_splatting_mesh_without_native_metadata(
    tmp_path: Path,
    monkeypatch,
):
    config = Config(
        {
            "adapter": {
                "type": "mesh-splatting",
                "checkpoint": str(tmp_path / "point_cloud" / "iteration_30000"),
            }
        }
    )
    pred = tmp_path / "fuse_post.ply"
    pred.write_text("old primitive mesh\n")
    calls = {}

    def fake_build_adapter(adapter_cfg):
        calls["adapter_cfg"] = adapter_cfg
        return object()

    def fake_export_adapter_mesh(adapter, path, **kwargs):
        calls["export_path"] = Path(path)
        calls["kwargs"] = kwargs
        Path(path).write_text("native tsdf mesh\n")
        return Path(path)

    monkeypatch.setattr("tribench.core.builder.build_adapter", fake_build_adapter)
    monkeypatch.setattr("tribench.core.mesh_eval.export_adapter_mesh", fake_export_adapter_mesh)

    assert _ensure_pred_mesh_exists(config, str(pred)) == str(pred)
    assert pred.read_text() == "native tsdf mesh\n"
    assert calls["adapter_cfg"]["type"] == "mesh-splatting"
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
