"""Tests for the inspect CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tribench.cli.main import app


def test_inspect_scene_option_writes_scene_summary(tmp_path: Path):
    scene = tmp_path / "garden"
    image_dir = scene / "images"
    sparse_dir = scene / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    (image_dir / "frame.png").write_bytes(b"")

    output = tmp_path / "scene.json"
    result = CliRunner().invoke(app, ["inspect", "--scene", str(scene), "--output", str(output)])

    assert result.exit_code == 0, result.output
    summary = json.loads(output.read_text())
    assert summary["path"] == str(scene)
    assert summary["dataset_type"] == "colmap"
    assert summary["image_count"] == 1
    assert summary["has_sparse"] is True


def test_inspect_requires_scene_or_checkpoint():
    result = CliRunner().invoke(app, ["inspect", "--output", "stats.json"])

    assert result.exit_code != 0
    assert "Use --scene, or provide both --method and --checkpoint." in result.output
