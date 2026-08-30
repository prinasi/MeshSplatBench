"""Tests for TriBench config-driven construction."""

from __future__ import annotations

from pathlib import Path

from tribench.core.builder import build_from_cfg
from tribench.core.config import Config, apply_overrides, load_config, resolve_dataset_config
from tribench.core.registry import get_adapter


class ToyComponent:
    def __init__(self, width: int, name: str = "toy") -> None:
        self.width = width
        self.name = name


def test_load_config_with_base(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "dataset:\n"
        "  type: colmap\n"
        "  root: data/base\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "  max_steps: 10\n"
    )
    child.write_text(
        "_base_: base.yaml\n"
        "dataset:\n"
        "  root: data/child\n"
        "trainer:\n"
        "  max_steps: 20\n"
    )

    cfg = load_config(child)
    assert cfg["dataset"]["type"] == "colmap"
    assert cfg["dataset"]["root"] == "data/child"
    assert cfg["trainer"]["type"] == "triangle-splatting"
    assert cfg["trainer"]["max_steps"] == 20


def test_load_config_resolves_scene_and_templates(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "dataset:\n"
        "  name: mipnerf360\n"
        "  type: colmap\n"
        "  root: data/mipnerf360\n"
        "output:\n"
        "  dir: outputs/{method}/{dataset}/{scene}\n"
    )
    child.write_text(
        "_base_: base.yaml\n"
        "dataset:\n"
        "  scene: bicycle\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
    )

    cfg = load_config(child)

    assert cfg["dataset"]["root"] == "data/mipnerf360/bicycle"
    assert cfg["dataset"]["scene"] == "bicycle"
    assert cfg["output"]["dir"] == "outputs/triangle-splatting/mipnerf360/bicycle"


def test_dtu_config_templates_include_eval_mode(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "dataset:\n"
        "  name: dtu\n"
        "  type: dtu\n"
        "  root: data/dtu\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "  white_background: false\n"
        "adapter:\n"
        "  type: triangle-splatting\n"
        "  render_params:\n"
        "    bg_color: black\n"
        "output:\n"
        "  dir: outputs/{method}/{dataset}/{scene}\n"
        "  metrics_file: outputs/{method}/{dataset}/{scene}/metrics.json\n"
    )
    child.write_text(
        "_base_: base.yaml\n"
        "dataset:\n"
        "  scene: scan24\n"
        "  dtu_eval_mode: foreground\n"
    )

    cfg = load_config(child)

    assert cfg["dataset"]["name"] == "dtu"
    assert cfg["dataset"]["dtu_eval_mode"] == "foreground"
    assert cfg["dataset"]["root"] == "data/dtu/scan24"
    assert cfg["trainer"]["foreground_training"] is False
    assert cfg["trainer"]["white_background"] is True
    assert cfg["adapter"]["render_params"]["bg_color"] == "white"
    assert cfg["output"]["dir"] == "outputs/triangle-splatting/dtu-fg/scan24"
    assert cfg["output"]["metrics_file"] == "outputs/triangle-splatting/dtu-fg/scan24/metrics.json"


def test_dtu_foreground_defaults_enable_diff_soup_foreground_loss(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        "  name: dtu\n"
        "  type: dtu\n"
        "  root: data/dtu\n"
        "  scene: scan24\n"
        "  dtu_eval_mode: foreground\n"
        "trainer:\n"
        "  type: diffsoup\n"
        "adapter:\n"
        "  type: diffsoup\n"
        "  render_params:\n"
        "    bg_color: black\n"
    )

    cfg = load_config(config)

    assert cfg["trainer"]["foreground_training"] is True
    assert cfg["trainer"]["white_background"] is True
    assert cfg["adapter"]["render_params"]["bg_color"] == "white"


def test_dtu_config_templates_default_to_full(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        "  name: dtu\n"
        "  type: dtu\n"
        "  root: data/dtu\n"
        "  scene: scan24\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "adapter:\n"
        "  type: triangle-splatting\n"
        "  render_params:\n"
        "    bg_color: white\n"
        "output:\n"
        "  dir: outputs/{method}/{dataset}/{scene}\n"
    )

    cfg = load_config(config)

    assert cfg["dataset"]["dtu_eval_mode"] == "full"
    assert cfg["trainer"]["foreground_training"] is False
    assert cfg["trainer"]["white_background"] is False
    assert cfg["adapter"]["render_params"]["bg_color"] == "black"
    assert cfg["output"]["dir"] == "outputs/triangle-splatting/dtu-full/scan24"


def test_dtu_config_respects_explicit_foreground_training_override(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        "  name: dtu\n"
        "  type: dtu\n"
        "  root: data/dtu\n"
        "  scene: scan24\n"
        "  dtu_eval_mode: foreground\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "  foreground_training: false\n"
    )

    cfg = load_config(config)

    assert cfg["dataset"]["dtu_eval_mode"] == "foreground"
    assert cfg["trainer"]["foreground_training"] is False


def test_resolve_dataset_config_does_not_duplicate_scene():
    cfg = resolve_dataset_config({"root": "data/mipnerf360/bicycle", "scene": "bicycle"})

    assert cfg["root"] == "data/mipnerf360/bicycle"


def test_scene_triangle_caps_populate_trainer_max_shapes(tmp_path: Path):
    base = tmp_path / "caps.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "triangle_caps:\n"
        "  bicycle: 6400000\n"
    )
    child.write_text(
        "_base_: caps.yaml\n"
        "dataset:\n"
        "  scene: bicycle\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
    )

    cfg = load_config(child)

    assert cfg["trainer"]["max_shapes"] == 6_400_000


def test_scene_triangle_caps_do_not_override_explicit_max_shapes(tmp_path: Path):
    base = tmp_path / "caps.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "triangle_caps:\n"
        "  bicycle: 6400000\n"
    )
    child.write_text(
        "_base_: caps.yaml\n"
        "dataset:\n"
        "  scene: bicycle\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "  max_shapes: 10\n"
    )

    cfg = load_config(child)

    assert cfg["trainer"]["max_shapes"] == 10


def test_max_primitives_caps_triangle_max_shapes(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        "  scene: bicycle\n"
        "triangle_caps:\n"
        "  bicycle: 6400000\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
        "  max_primitives: 15000\n"
    )

    cfg = load_config(config)

    assert cfg["trainer"]["max_primitives"] == 15_000
    assert cfg["trainer"]["max_shapes"] == 15_000


def test_max_primitives_caps_2dts_target_point_num(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "trainer:\n"
        "  type: 2dts\n"
        "  max_primitives: 15000\n"
        "d2ts:\n"
        "  target_point_num: 1000000\n"
    )

    cfg = load_config(config)

    assert cfg["trainer"]["max_primitives"] == 15_000
    assert cfg["d2ts"]["target_point_num"] == 15_000


def test_diffsoup_config_resolves_final_params_checkpoint():
    cfg = load_config(Path("configs/diffsoup/mipnerf360/room.yaml"))

    assert cfg["trainer"]["type"] == "diffsoup"
    assert cfg["trainer"]["batch_size"] == 4
    assert cfg["trainer"]["downscale"] == 4
    assert "white_background" not in cfg["trainer"]
    assert cfg["dataset"]["root"].endswith("data/mipnerf360/room")
    assert cfg["dataset"]["image_dir"] in ("images_2", "images_4")
    assert cfg["dataset"]["resolution"] == 1
    assert cfg["adapter"]["checkpoint"] == "outputs/diffsoup/mipnerf360/room/ckpt/final_params.pt"



def test_scene_triangle_caps_support_scan_pattern(tmp_path: Path):
    base = tmp_path / "caps.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "triangle_caps:\n"
        "  \"scan*\": 500000\n"
    )
    child.write_text(
        "_base_: caps.yaml\n"
        "dataset:\n"
        "  scene: scan24\n"
        "trainer:\n"
        "  type: triangle-splatting\n"
    )

    cfg = load_config(child)

    assert cfg["trainer"]["max_shapes"] == 500_000


def test_triangle_splatting_scene_configs_apply_full_eval_caps():
    expected = {
        "mipnerf360/bicycle": 6_400_000,
        "mipnerf360/flowers": 5_500_000,
        "mipnerf360/garden": 5_200_000,
        "mipnerf360/stump": 4_750_000,
        "mipnerf360/treehill": 5_000_000,
        "mipnerf360/room": 2_100_000,
        "mipnerf360/counter": 2_500_000,
        "mipnerf360/kitchen": 2_400_000,
        "mipnerf360/bonsai": 3_000_000,
        "tandt/truck": 2_000_000,
        "tandt/train": 2_500_000,
        "dtu/scan24": 500_000,
    }
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "triangle-splatting"

    for scene_path, cap in expected.items():
        cfg = load_config(config_dir / f"{scene_path}.yaml")
        assert cfg["trainer"]["max_shapes"] == cap




def test_2dts_mipnerf360_opaque_experiment_configs_resolve_to_isolated_outputs():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "opaque_2dts_mipnerf360" / "2dts" / "mipnerf360"
    scene_configs = sorted(config_dir.glob("*.yaml"))

    assert len(scene_configs) == 9

    for path in scene_configs:
        cfg = Config.fromfile(path)
        assert cfg.output.dir.startswith("outputs/opaque_2dts_mipnerf360/2dts/mipnerf360/")
        assert cfg.adapter.checkpoint.startswith("outputs/opaque_2dts_mipnerf360/2dts/mipnerf360/")
        assert cfg.output.metrics_file.startswith(f"{cfg.output.dir}/")
        assert cfg.output.stats_file.startswith(f"{cfg.output.dir}/")
        assert cfg.render.video.output_dir.startswith(f"{cfg.output.dir}/")
        assert cfg.profile.output.startswith(f"{cfg.output.dir}/")
        assert cfg.inspect.output.startswith(f"{cfg.output.dir}/")
        assert cfg.d2ts.native_config == "configs/opaque_2dts_mipnerf360/2dts/native/mipnerf360.yaml"
        assert cfg.adapter.render_params.ste_threshold == 0.3
        assert cfg.adapter.render_params.sort_level == 2
        assert cfg.dataset.scene == path.stem





def test_build_from_cfg_with_params():
    obj = build_from_cfg(
        {"type": "toy", "params": {"width": 8}, "name": "demo"},
        {"toy": ToyComponent},
    )
    assert obj.width == 8
    assert obj.name == "demo"


def test_apply_dotlist_overrides():
    cfg = apply_overrides(
        {"trainer": {"max_steps": 10}, "render": {"save_gt": False}},
        ["trainer.max_steps=30", "render.save_gt=true"],
    )
    assert cfg["trainer"]["max_steps"] == 30
    assert cfg["render"]["save_gt"] is True


def test_config_attribute_access():
    cfg = Config({"trainer": {"type": "triangle-splatting"}})
    assert cfg.trainer.type == "triangle-splatting"
    assert cfg.to_dict() == {"trainer": {"type": "triangle-splatting"}}


def test_builtin_adapters_are_discovered():
    assert get_adapter("triangle-splatting").__name__ == "TriangleSplattingAdapter"


def test_tribench_package_imports():
    import tribench

    assert tribench.__version__ == "0.1.0"


def test_apply_max_primitive_limit_per_method_caps():
    from tribench.core.config import _apply_max_primitive_limit

    # 1. Triangle splatting with method-specific cap
    cfg_ts = {
        "trainer": {
            "type": "triangle-splatting",
            "max_primitives": 15000,
            "max_primitives_triangle_splatting": 25000,
        }
    }
    res_ts = _apply_max_primitive_limit(cfg_ts)
    assert res_ts["trainer"]["max_primitives"] == 25000
    assert res_ts["trainer"]["max_shapes"] == 25000

    # 2. Mesh splatting with method-specific cap and max_points bound
    cfg_ms = {
        "trainer": {
            "type": "mesh-splatting",
            "max_primitives": 15000,
            "max_primitives_mesh_splatting": 12000,
            "max_points": 4000000,
        }
    }
    res_ms = _apply_max_primitive_limit(cfg_ms)
    assert res_ms["trainer"]["max_primitives"] == 12000
    assert res_ms["trainer"]["max_points"] == 36000

    # 3. 2DTS with method-specific block cap
    cfg_2dts = {
        "trainer": {
            "type": "2dts",
            "max_primitives": 15000,
        },
        "d2ts": {
            "max_primitives": 8000,
        },
    }
    res_2dts = _apply_max_primitive_limit(cfg_2dts)
    assert res_2dts["trainer"]["max_primitives"] == 8000
    assert res_2dts["d2ts"]["target_point_num"] == 8000
