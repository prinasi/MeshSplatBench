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
