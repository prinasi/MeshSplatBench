"""Tests for the DiffSoup native training wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tribench.trainers.diffsoup_native import (
    DEFAULT_DIFFSOUP_EXAMPLES_DIR,
    DEFAULT_DIFFSOUP_ROOT,
    _load_reference_script,
    _resolve_examples_dir,
    _resolve_dataset_kind,
)


def test_diffsoup_dataset_kind_mapping():
    assert _resolve_dataset_kind({}, {"type": "dtu"}, "data/dtu/scan24") == "dtu"
    assert _resolve_dataset_kind({}, {"type": "tanks"}, "data/tandt/train") == "colmap"
    assert _resolve_dataset_kind({}, {"type": "synthetic"}, "data/nerf/lego") == "synthetic"
    assert _resolve_dataset_kind({"dataset_type": "dtu"}, {"type": "colmap"}, "x") == "dtu"


def test_diffsoup_examples_default_to_vendored_copy():
    examples_dir = _resolve_examples_dir({})

    assert examples_dir == DEFAULT_DIFFSOUP_EXAMPLES_DIR
    assert DEFAULT_DIFFSOUP_ROOT.name == "diffsoup"
    assert (examples_dir / "01_mip360.py").is_file()
    assert (examples_dir / "02_synthetic.py").is_file()
    assert (examples_dir / "03_random_init.py").is_file()
    assert (examples_dir / "utils.py").is_file()


def test_diffsoup_reference_script_loads_with_vendored_runtime():
    for dependency in ("open3d", "torchvision", "pytorch_msssim"):
        pytest.importorskip(dependency)

    old_diffsoup = sys.modules.get("diffsoup")
    try:
        module = _load_reference_script(DEFAULT_DIFFSOUP_EXAMPLES_DIR, "01_mip360.py")
    except ImportError as exc:
        pytest.skip(f"DiffSoup native extension is not available: {exc}")

    assert hasattr(module, "main")
    assert Path(module.__file__).name == "01_mip360.py"
    assert sys.modules.get("diffsoup") is old_diffsoup
