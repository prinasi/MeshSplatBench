"""Tests for feature-preserving Unity-native asset export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from typer.testing import CliRunner

from tribench.cli.main import app
from tribench.unity_assets import export_triasset


def _save_triangle_splatting_checkpoint(path: Path) -> None:
    torch.save(
        {
            "triangles_points": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
            "sigma": torch.tensor([[0.1], [0.2]]),
            "active_sh_degree": 1,
            "features_dc": torch.zeros(2, 1, 3),
            "features_rest": torch.ones(2, 3, 3),
            "opacity": torch.tensor([[0.3], [0.4]]),
        },
        path,
    )


def _save_mesh_splatting_checkpoint(path: Path) -> None:
    torch.save(
        {
            "triangles_points": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "_triangle_indices": torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
            "vertex_weight": torch.zeros(4, 1),
            "opacity_floor": 0.9999,
            "sigma": -9.210340371976294,
            "active_sh_degree": 1,
            "features_dc": torch.zeros(4, 1, 3),
            "features_rest": torch.ones(4, 3, 3),
        },
        path,
    )


def _save_d2ts_checkpoint(path: Path) -> None:
    state = {
        "_vertex": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
        "_opacity": torch.tensor([[0.3]]),
        "_f_dc": torch.zeros(1, 3, 1, 3),
        "_f_rest": torch.ones(1, 3, 3, 3),
        "active_sh_degree": 1,
    }
    torch.save((state, {}, None, 1.7), path)


def _save_diffsoup_checkpoint(path: Path) -> None:
    torch.save(
        {
            "V": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "F": torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
            "feat_acc": torch.ones(2, 4, 7),
            "alpha_acc": torch.full((2, 4, 1), 0.5),
            "Rmin": 0,
            "Rmax": 1,
            "feat_dim": 7,
            "color_mlp": {
                "mlp.0.weight": torch.ones(16, 16),
                "mlp.0.bias": torch.zeros(16),
                "mlp.4.weight": torch.ones(3, 16),
                "mlp.4.bias": torch.zeros(3),
            },
        },
        path,
    )


def test_triangle_splatting_export_keeps_renderer_inputs(tmp_path: Path):
    checkpoint = tmp_path / "point_cloud_state_dict.pt"
    _save_triangle_splatting_checkpoint(checkpoint)

    package = export_triasset("triangle-splatting", checkpoint, tmp_path / "scene")
    manifest = json.loads(package.manifest_path.read_text())

    assert package.path.name == "scene.triasset"
    assert manifest["export_contract_revision"] == 2
    assert manifest["rendering"]["renderer"] == "triangle-splatting"
    assert manifest["rendering"]["generic_mesh_equivalent"] is False
    assert manifest["rendering"]["background_color"] == "black"
    assert manifest["rendering"]["unity_profile_name"] == "method-aware"
    assert manifest["rendering"]["cuda_equivalent"] is False
    assert manifest["rendering"]["general_purpose"]["supported"] is True
    assert manifest["buffers"]["positions"]["shape"] == [6, 3]
    assert manifest["buffers"]["opacity_logits"]["semantic"] == "per-triangle opacity logits"
    assert manifest["buffers"]["sigma_logits"]["shape"] == [2, 1]
    assert {entry["name"] for entry in manifest["buffer_list"]} == set(manifest["buffers"])
    assert (package.path / manifest["buffers"]["sh_rest"]["file"]).is_file()


def test_mesh_splatting_export_keeps_indexed_topology(tmp_path: Path):
    checkpoint = tmp_path / "point_cloud_state_dict.pt"
    _save_mesh_splatting_checkpoint(checkpoint)

    package = export_triasset("mesh_splatting", checkpoint, tmp_path / "mesh.triasset")
    manifest = json.loads(package.manifest_path.read_text())

    assert manifest["rendering"]["primitive_topology"] == "indexed-triangles"
    assert manifest["buffers"]["indices"]["dtype"] == "int32"
    assert manifest["buffers"]["vertex_weight_logits"]["shape"] == [4, 1]
    assert manifest["buffers"]["triangle_opacity"]["shape"] == [2]
    assert manifest["rendering"]["opacity_activation"] == "opacity_floor+(1-opacity_floor)*sigmoid"
    assert manifest["rendering"]["opacity_floor"] == 0.9999
    assert manifest["rendering"]["opacity_floor_source"] == "checkpoint"
    assert manifest["rendering"]["sigma_activation"] == "exp"
    assert manifest["rendering"]["terminal_solid_eligible"] is True
    assert manifest["rendering"]["unity_method_aware_requires_per_camera_depth_sort"] is False
    assert manifest["rendering"]["unity_method_aware_status"] == "opaque-terminal-solid"
    assert manifest["rendering"]["triangle_opacity_reduction"] == "min_after_activation"
    triangle_opacity = torch.from_file(
        str(package.path / manifest["buffers"]["triangle_opacity"]["file"]),
        dtype=torch.float32,
        size=2,
    )
    assert torch.all(triangle_opacity >= 0.9999)
    assert torch.all(triangle_opacity < 1.0)


def test_mesh_splatting_export_can_materialize_triangle_soup(tmp_path: Path):
    checkpoint = tmp_path / "point_cloud_state_dict.pt"
    _save_mesh_splatting_checkpoint(checkpoint)

    package = export_triasset(
        "mesh_splatting",
        checkpoint,
        tmp_path / "mesh_soup.triasset",
        export_topology="soup",
    )
    manifest = json.loads(package.manifest_path.read_text())

    assert manifest["rendering"]["primitive_topology"] == "triangle-soup"
    assert manifest["rendering"]["export_topology"] == "materialized-soup"
    assert manifest["rendering"]["source_vertex_count"] == 4
    assert manifest["rendering"]["exported_vertex_count"] == 6
    assert manifest["rendering"]["topology_vertex_expansion"] == 1.5
    assert manifest["buffers"]["positions"]["shape"] == [6, 3]
    assert manifest["buffers"]["indices"]["shape"] == [2, 3]
    assert manifest["buffers"]["vertex_weight_logits"]["shape"] == [6, 1]
    assert manifest["buffers"]["sh_dc"]["shape"] == [6, 1, 3]
    assert manifest["buffers"]["sh_rest"]["shape"] == [6, 3, 3]

    indices = torch.from_file(
        str(package.path / manifest["buffers"]["indices"]["file"]),
        dtype=torch.int32,
        size=6,
    ).reshape(2, 3)
    torch.testing.assert_close(indices, torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32))


def test_d2ts_export_keeps_vertex_sh_and_gamma(tmp_path: Path):
    checkpoint = tmp_path / "model.ckpt"
    _save_d2ts_checkpoint(checkpoint)

    package = export_triasset("d2ts", checkpoint, tmp_path / "d2ts")
    manifest = json.loads(package.manifest_path.read_text())

    assert manifest["method"] == "2dts"
    assert manifest["rendering"]["vertex_color"] is True
    assert manifest["rendering"]["gamma"] == 1.7
    assert manifest["rendering"]["gamma_rescale"] is True
    assert manifest["rendering"]["gamma_rescale_source"] == "legacy_inferred_from_gamma"
    assert 0.0 < manifest["rendering"]["gamma_vertex_rescale"] < 1.0
    assert manifest["rendering"]["background_color"] == "white"
    assert manifest["buffers"]["sh_dc"]["shape"] == [1, 3, 1, 3]


def test_diffsoup_export_keeps_neural_features_and_mlp(tmp_path: Path):
    checkpoint = tmp_path / "final_params.pt"
    _save_diffsoup_checkpoint(checkpoint)

    package = export_triasset("diffsoup", checkpoint, tmp_path / "diffsoup")
    manifest = json.loads(package.manifest_path.read_text())

    assert manifest["rendering"]["renderer"] == "diffsoup"
    assert manifest["buffers"]["features"]["shape"] == [2, 4, 7]
    assert manifest["buffers"]["alpha"]["shape"] == [2, 4, 1]
    assert set(manifest["rendering"]["color_mlp_buffers"]) == {
        "mlp.0.weight", "mlp.0.bias", "mlp.4.weight", "mlp.4.bias"
    }
    assert manifest["rendering"]["general_purpose"]["supported"] is False
    assert "texture" in manifest["rendering"]["general_purpose"]["reason"]


def test_export_unity_cli_writes_feature_preserving_package(tmp_path: Path):
    checkpoint = tmp_path / "point_cloud_state_dict.pt"
    _save_triangle_splatting_checkpoint(checkpoint)
    output = tmp_path / "cli_asset"

    result = CliRunner().invoke(
        app,
        [
            "export-unity",
            "--method", "triangle-splatting",
            "--checkpoint", str(checkpoint),
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli_asset.triasset" / "manifest.json").is_file()
    assert "buffers: 6" in result.output


def test_unity_bootstrap_installs_compute_and_method_renderers(tmp_path: Path):
    project = tmp_path / "UnityProject"
    (project / "Assets").mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "create_unity_triasset_package.py"),
            "--unity-project",
            str(project),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Metal procedural renderer" in result.stdout
    assert (project / "Assets" / "TriBench" / "Scripts" / "TriAssetSplatRenderer.cs").is_file()
    assert (project / "Assets" / "TriBench" / "Scripts" / "GeneralPurposeMeshTriAssetRenderer.cs").is_file()
    assert (project / "Assets" / "TriBench" / "Shaders" / "TriAssetSplat.shader").is_file()
    assert (project / "Assets" / "TriBench" / "Shaders" / "MeshSplatTerminalSolid.shader").is_file()
    assert (project / "Assets" / "TriBench" / "Shaders" / "GeneralPurposeVertexColor.shader").is_file()
    assert (project / "Assets" / "Resources" / "TriBench" / "TriAssetCull.compute").is_file()
    shader = (project / "Assets" / "TriBench" / "Shaders" / "TriAssetSplat.shader").read_text()
    renderer = (project / "Assets" / "TriBench" / "Scripts" / "TriAssetSplatRenderer.cs").read_text()
    assert "nointerpolation float flatAlpha" in shader
    assert "_Opacity[face]" in shader
    assert "_ScreenParams.xy" in shader
    assert 'OpacityBuffer { get { return "triangle_opacity"; } }' in renderer
    general = (project / "Assets" / "TriBench" / "Scripts" / "GeneralPurposeMeshTriAssetRenderer.cs").read_text()
    assert "Per-face colour cannot be stored on shared vertices" in general
    assert "fixed-budget appearance bake is required" in general
