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
        "bicycle": 6_400_000,
        "flowers": 5_500_000,
        "garden": 5_200_000,
        "stump": 4_750_000,
        "treehill": 5_000_000,
        "room": 2_100_000,
        "counter": 2_500_000,
        "kitchen": 2_400_000,
        "bonsai": 3_000_000,
        "truck": 2_000_000,
        "train": 2_500_000,
        "scan24": 500_000,
    }
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "triangle-splatting"

    for scene, cap in expected.items():
        cfg = load_config(config_dir / f"{scene}.yaml")
        assert cfg["trainer"]["max_shapes"] == cap


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
