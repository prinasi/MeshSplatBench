"""Tests for the DiffSoup native training wrapper."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tribench.trainers import diffsoup_native


def test_diffsoup_dataset_kind_mapping():
    assert diffsoup_native._resolve_dataset_kind({}, {"type": "dtu"}, "data/dtu/scan24") == "dtu"
    assert diffsoup_native._resolve_dataset_kind({}, {"type": "tanks"}, "data/tandt/train") == "colmap"
    assert diffsoup_native._resolve_dataset_kind({}, {"type": "synthetic"}, "data/nerf/lego") == "synthetic"
    assert diffsoup_native._resolve_dataset_kind({"dataset_type": "dtu"}, {"type": "colmap"}, "x") == "dtu"


def test_diffsoup_examples_default_to_vendored_copy():
    examples_dir = diffsoup_native._resolve_examples_dir({})

    assert examples_dir == diffsoup_native.DEFAULT_DIFFSOUP_EXAMPLES_DIR
    assert diffsoup_native.DEFAULT_DIFFSOUP_ROOT.name == "diffsoup"
    assert (examples_dir / "01_mip360.py").is_file()
    assert (examples_dir / "02_synthetic.py").is_file()
    assert (examples_dir / "03_random_init.py").is_file()
    assert (examples_dir / "utils.py").is_file()


def test_diffsoup_synthetic_default_uses_mobilenerf_mesh(monkeypatch, tmp_path):
    calls = {}

    def fake_load_reference_script(examples_dir, script_name):
        calls["script_name"] = script_name

        def main():
            calls["argv"] = list(sys.argv)

        return SimpleNamespace(main=main)

    monkeypatch.setattr(diffsoup_native, "_load_reference_script", fake_load_reference_script)
    monkeypatch.setattr(diffsoup_native, "_reference_import_context", lambda examples_dir: nullcontext())

    dataset_root = tmp_path / "nerf_synthetic" / "lego"
    dataset_root.mkdir(parents=True)
    output_dir = tmp_path / "out"
    diffsoup_native._run_synthetic_script(
        examples_dir=tmp_path,
        trainer_cfg={
            "mobilenerf_root": "data/mobilenerf_results",
            "batch_size": 4,
            "target_prims": 15_000,
            "downscale": 1,
        },
        dataset_root=str(dataset_root),
        output_dir=output_dir,
        max_steps=10_000,
        dataset_kind="synthetic",
        dataset_cfg={"scene": "lego", "type": "synthetic"},
    )

    assert calls["script_name"] == "02_synthetic.py"
    assert calls["argv"][calls["argv"].index("--steps") + 1] == "10000"
    assert calls["argv"][calls["argv"].index("--batch_size") + 1] == "4"
    assert calls["argv"][calls["argv"].index("--target_prims") + 1] == "15000"
    assert calls["argv"][calls["argv"].index("--downscale") + 1] == "1"
    assert "--mobilenerf_root" in calls["argv"]
    assert calls["argv"][calls["argv"].index("--mobilenerf_root") + 1] == "data/mobilenerf_results"
    assert "--point_cloud" not in calls["argv"]
    assert "--n_points" not in calls["argv"]


def test_diffsoup_reference_script_loads_with_vendored_runtime():
    for dependency in ("open3d", "torchvision", "pytorch_msssim"):
        pytest.importorskip(dependency)

    old_diffsoup = sys.modules.get("diffsoup")
    try:
        module = diffsoup_native._load_reference_script(diffsoup_native.DEFAULT_DIFFSOUP_EXAMPLES_DIR, "01_mip360.py")
    except ImportError as exc:
        pytest.skip(f"DiffSoup native extension is not available: {exc}")

    assert hasattr(module, "main")
    assert Path(module.__file__).name == "01_mip360.py"
    assert sys.modules.get("diffsoup") is old_diffsoup
