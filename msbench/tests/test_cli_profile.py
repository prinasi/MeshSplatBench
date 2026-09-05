"""Tests for the profile CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from typer.testing import CliRunner

from msbench.cli.main import app
from msbench.cli.profile import _resolve_scene_path
from msbench.core.cameras import CameraBatch
from msbench.core.profiler import profile_forward_backward, profile_renderer
from msbench.renderers.base import RendererAdapter, RenderOutput


def test_profile_accepts_scene_and_config_options(tmp_path: Path):
    config = tmp_path / "2dts.yaml"
    config.write_text(
        "dataset:\n"
        "  root: data/mipnerf360\n"
        "adapter:\n"
        "  type: missing-method\n"
        "  checkpoint: outputs/2dts/{scene}\n"
    )

    result = CliRunner().invoke(
        app,
        ["profile", "--method", "missing-method", "--scene", "garden", "--config", str(config)],
    )

    assert result.exit_code != 2
    assert "No such option: --scene" not in result.output
    assert "missing-method" in result.output


def test_profile_infers_checkpoint_from_train_output_dir(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "dataset:\n"
        "  root: data/bicycle\n"
        "trainer:\n"
        "  type: missing-method\n"
        "output:\n"
        "  dir: outputs/triangle-splatting/bicycle\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "profile",
            "--method",
            "missing-method",
            "--scene",
            "garden",
            "--config",
            str(config),
        ],
    )

    assert result.exit_code != 2
    assert "Use --checkpoint" not in result.output
    assert "Checkpoint: outputs/triangle-splatting/garden" in result.output
    assert "Dataset: data/garden" in result.output


def test_resolve_scene_path_replaces_scene_component():
    assert (
        _resolve_scene_path("outputs/triangle-splatting/bicycle/profile.json", "garden")
        == "outputs/triangle-splatting/garden/profile.json"
    )


def test_profile_config_runs_measurements_and_preserves_adapter_config(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    config = tmp_path / "triangle-splatting.yaml"
    config.write_text(
        "dataset:\n"
        "  root: data/bicycle\n"
        "adapter:\n"
        "  type: triangle-splatting\n"
        "  checkpoint: outputs/bicycle\n"
        "  render_params:\n"
        "    bg_color: white\n"
        "profile:\n"
        "  repeats: 2\n"
        "  warmup: 1\n"
        "  backward: false\n"
        f"  output: {output}\n"
    )
    built = {}

    class FakeAdapter:
        device = torch.device("cpu")

        def __init__(self):
            self._model = torch.nn.Linear(1, 1, bias=False)

        def render(self, camera, *, mode="eval"):
            value = self._model(torch.ones(1, 1))
            return RenderOutput(rgb=value.expand(2, 2, 3))

        def profile_metadata(self):
            return {"primitive_count": 12, "checkpoint_size_mb": 3.5}

    camera = CameraBatch(
        viewmats=torch.eye(4).unsqueeze(0),
        camtoworlds=torch.eye(4).unsqueeze(0),
        Ks=torch.eye(3).unsqueeze(0),
        width=2,
        height=2,
    )

    class FakeDataset:
        def __len__(self):
            return 1

        def sample(self, index):
            return SimpleNamespace(camera=camera, name="camera-0")

    dataset = FakeDataset()

    def build_adapter(adapter_cfg):
        built.update(adapter_cfg)
        return FakeAdapter()

    monkeypatch.setattr("msbench.core.builder.build_adapter", build_adapter)
    monkeypatch.setattr("msbench.core.builder.build_dataset", lambda dataset_cfg: dataset)

    result = CliRunner().invoke(
        app,
        [
            "profile",
            "--config",
            str(config),
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--backward",
        ],
    )

    assert result.exit_code == 0, result.output
    assert built["render_params"] == {"bg_color": "white"}
    profile = json.loads(output.read_text())
    assert profile["status"] == "complete"
    assert profile["schema_version"] == 1
    assert profile["primitive_count"] == 12
    assert profile["checkpoint_size_mb"] == 3.5
    assert profile["camera_name"] == "camera-0"
    assert profile["repeats"] == 1
    assert profile["warmup"] == 0
    assert profile["backward_latency_ms"] >= 0.0


class _DifferentiableRenderer(RendererAdapter):
    def __init__(self) -> None:
        self._model = torch.nn.Linear(1, 1, bias=False)

    @property
    def name(self) -> str:
        return "test"

    def load_checkpoint(self, path: str) -> None:
        pass

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        pass

    def model_stats(self):
        return {"primitive_count": 7, "checkpoint_size_mb": 1.25}

    def render(self, cameras, *, mode: str = "eval") -> RenderOutput:
        value = self._model(torch.ones(1, 1))
        return RenderOutput(rgb=value.expand(2, 2, 1))


def test_profile_renderer_uses_shared_schema_on_cpu():
    renderer = _DifferentiableRenderer()

    result = profile_renderer(renderer, None, repeats=2, warmup=0)

    assert result["schema_version"] == 1
    assert result["primitive_count"] == 7
    assert result["checkpoint_size_mb"] == 1.25
    assert result["forward_latency_ms"] == result["forward"]["mean_ms"]
    assert result["fps"] == result["forward"]["fps"]
    assert result["peak_cuda_memory_gb"] == 0.0
    assert "CUDA" in result["memory"]["note"]
    assert "backward" not in result


def test_backward_latency_is_measured_directly():
    renderer = _DifferentiableRenderer()

    result = profile_forward_backward(renderer, None, repeats=2, warmup=1)

    assert result["forward"]["min_ms"] >= 0.0
    assert result["backward"]["min_ms"] >= 0.0
    assert result["total"]["mean_ms"] >= result["backward"]["mean_ms"]
    assert all(parameter.grad is None for parameter in renderer._model.parameters())


def test_triangle_splatting_profile_metadata_avoids_full_model_stats(tmp_path):
    from msbench.renderers.triangle_splatting_adapter import TriangleSplattingAdapter

    checkpoint = tmp_path / "point_cloud_state_dict.pt"
    checkpoint.write_bytes(b"checkpoint")
    adapter = TriangleSplattingAdapter()
    adapter._model = SimpleNamespace(_triangles_points=torch.empty(11, 3, 3))
    adapter._checkpoint_path = str(checkpoint)

    metadata = adapter.profile_metadata()

    assert metadata["primitive_count"] == 11
    assert metadata["checkpoint_size_mb"] >= 0.0
