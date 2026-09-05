"""Tests for ModelStats and triangle statistics."""

import json
import torch
import pytest

from msbench.core.stats import (
    ModelStats,
    compute_triangle_areas,
    compute_triangle_normals,
    compute_triangle_stats,
)


class TestModelStats:
    """Tests for ModelStats dataclass."""

    def test_default_creation(self):
        stats = ModelStats()
        assert stats.method == ""
        assert stats.primitive_count == 0
        assert stats.backend_extra == {}

    def test_custom_creation(self):
        stats = ModelStats(
            method="2dts",
            scene="garden",
            primitive_count=100000,
            vertex_count=300000,
        )
        assert stats.method == "2dts"
        assert stats.primitive_count == 100000

    def test_to_dict(self):
        stats = ModelStats(method="2dts", primitive_count=42)
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["method"] == "2dts"
        assert d["primitive_count"] == 42

    def test_to_json_string(self):
        stats = ModelStats(method="2dts", primitive_count=42)
        json_str = stats.to_json()
        data = json.loads(json_str)
        assert data["method"] == "2dts"
        assert data["primitive_count"] == 42

    def test_to_json_file(self, tmp_path):
        stats = ModelStats(method="2dts", primitive_count=42)
        path = tmp_path / "stats.json"
        stats.to_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["method"] == "2dts"

    def test_from_dict(self):
        data = {"method": "2dts", "primitive_count": 42, "unknown_field": "ignored"}
        stats = ModelStats.from_dict(data)
        assert stats.method == "2dts"
        assert stats.primitive_count == 42

    def test_from_json(self, tmp_path):
        stats = ModelStats(method="mesh-splatting", face_count=500)
        path = tmp_path / "stats.json"
        stats.to_json(path)

        loaded = ModelStats.from_json(path)
        assert loaded.method == "mesh-splatting"
        assert loaded.face_count == 500

    def test_roundtrip(self):
        original = ModelStats(
            method="triangle-splatting",
            primitive_count=1000,
            opacity_mean=0.5,
            backend_extra={"sigma": 2.0},
        )
        json_str = original.to_json()
        loaded = ModelStats.from_dict(json.loads(json_str))
        assert loaded.method == original.method
        assert loaded.primitive_count == original.primitive_count
        assert loaded.opacity_mean == original.opacity_mean
        assert loaded.backend_extra == original.backend_extra

    def test_backend_extra(self):
        stats = ModelStats(
            method="2dts",
            backend_extra={"gamma": 1.0, "sigma_mean": 2.0},
        )
        assert stats.backend_extra["gamma"] == 1.0


class TestTriangleAreas:
    """Tests for triangle area computation."""

    def test_unit_triangle(self):
        vertices = torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
        ])
        areas = compute_triangle_areas(vertices)
        assert areas.shape == (1,)
        assert torch.isclose(areas[0], torch.tensor(0.5))

    def test_degenerate_triangle(self):
        vertices = torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0]],  # Collinear points
        ])
        areas = compute_triangle_areas(vertices)
        assert torch.isclose(areas[0], torch.tensor(0.0))

    def test_batch_triangles(self):
        torch.manual_seed(42)
        vertices = torch.randn(10, 3, 3)
        areas = compute_triangle_areas(vertices)
        assert areas.shape == (10,)
        assert (areas >= 0).all()


class TestTriangleNormals:
    """Tests for triangle normal computation."""

    def test_xy_plane_normal(self):
        vertices = torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
        ])
        normals = compute_triangle_normals(vertices)
        assert normals.shape == (1, 3)
        expected = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.allclose(normals, expected, atol=1e-6)

    def test_unit_normals(self, dummy_triangle_vertices):
        normals = compute_triangle_normals(dummy_triangle_vertices)
        norms = torch.norm(normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestComputeTriangleStats:
    """Tests for compute_triangle_stats helper."""

    def test_without_opacity(self, dummy_triangle_vertices):
        stats = compute_triangle_stats(dummy_triangle_vertices)
        assert "triangle_area_mean" in stats
        assert "triangle_area_median" in stats
        assert "triangle_area_p95" in stats
        assert "opacity_mean" not in stats

    def test_with_opacity(self, dummy_triangle_vertices, dummy_opacity):
        stats = compute_triangle_stats(dummy_triangle_vertices, dummy_opacity)
        assert "opacity_mean" in stats
        assert "opacity_median" in stats
        assert "opacity_histogram" in stats
        assert len(stats["opacity_histogram"]) == 10  # 10 bins
