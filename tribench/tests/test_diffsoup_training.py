"""Tests for the DiffSoup native training wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tribench.trainers.diffsoup_native import (
    DEFAULT_DIFFSOUP_ROOT,
    _load_reference_script,
    _resolve_dataset_kind,
)


def test_diffsoup_dataset_kind_mapping():
    assert _resolve_dataset_kind({}, {"type": "dtu"}, "data/dtu/scan24") == "dtu"
    assert _resolve_dataset_kind({}, {"type": "tanks"}, "data/tandt/train") == "colmap"
    assert _resolve_dataset_kind({}, {"type": "synthetic"}, "data/nerf/lego") == "synthetic"
    assert _resolve_dataset_kind({"dataset_type": "dtu"}, {"type": "colmap"}, "x") == "dtu"


@pytest.mark.skipif(
    not (DEFAULT_DIFFSOUP_ROOT / "examples" / "01_mip360.py").is_file(),
    reason="reference DiffSoup checkout is not available",
)
def test_diffsoup_reference_script_loads_with_vendored_runtime():
    old_diffsoup = sys.modules.get("diffsoup")
    module = _load_reference_script(DEFAULT_DIFFSOUP_ROOT / "examples", "01_mip360.py")

    assert hasattr(module, "main")
    assert Path(module.__file__).name == "01_mip360.py"
    assert sys.modules.get("diffsoup") is old_diffsoup
