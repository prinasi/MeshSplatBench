"""Tests for primitive types."""

import torch
import pytest

from trianglebench.primitives.triangle import IndependentTriangle
from trianglebench.primitives.mesh_triangle import IndexedMeshTriangle
from trianglebench.primitives.convex_triangle import ConvexTriangle


class TestIndependentTriangle:
    """Tests for IndependentTriangle (2DTS-style)."""

    def test_creation(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        assert prim.num_primitives == 10
        assert prim.num_vertices == 30

    def test_default_opacity(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        assert prim.opacity.shape == (10,)
        assert (prim.opacity == 1.0).all()

    def test_custom_opacity(self, dummy_triangle_vertices, dummy_opacity):
        prim = IndependentTriangle(dummy_triangle_vertices, opacity=dummy_opacity)
        assert torch.equal(prim.get_opacity(), dummy_opacity)

    def test_compute_areas(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        areas = prim.compute_areas()
        assert areas.shape == (10,)
        assert (areas >= 0).all()

    def test_compute_normals(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        normals = prim.compute_normals()
        assert normals.shape == (10, 3)
        # Check unit length
        norms = torch.norm(normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_to_tensor(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        tensor = prim.to_tensor()
        assert tensor.shape == (10, 3, 3)
        assert torch.equal(tensor, dummy_triangle_vertices)

    def test_to_device(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices)
        prim_cpu = prim.to("cpu")
        assert prim_cpu.vertices.device == torch.device("cpu")

    def test_with_sh(self, dummy_triangle_vertices):
        sh = torch.randn(10, 9)  # SH degree 1 (3 coeffs per channel)
        prim = IndependentTriangle(dummy_triangle_vertices, sh_coeffs=sh)
        colors = prim.get_colors()
        assert colors is not None
        assert colors.shape == (10, 3)

    def test_extra_stats(self, dummy_triangle_vertices):
        prim = IndependentTriangle(dummy_triangle_vertices, gamma=2.0, back_culling=True)
        stats = prim.extra_stats()
        assert stats["gamma"] == 2.0
        assert stats["back_culling"] is True

    def test_invalid_shape_raises(self):
        with pytest.raises(AssertionError):
            IndependentTriangle(torch.randn(10, 3))  # Wrong shape


class TestIndexedMeshTriangle:
    """Tests for IndexedMeshTriangle (MeshSplatting-style)."""

    def test_creation(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        assert prim.num_primitives == 2  # 2 faces
        assert prim.num_vertices == 4  # 4 unique vertices

    def test_get_face_vertices(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        fv = prim.get_face_vertices()
        assert fv.shape == (2, 3, 3)

    def test_compute_areas(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        areas = prim.compute_areas()
        assert areas.shape == (2,)
        # Both triangles in a unit square have area 0.5
        assert torch.allclose(areas, torch.tensor([0.5, 0.5]))

    def test_compute_normals(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        normals = prim.compute_normals()
        assert normals.shape == (2, 3)
        # Both faces are in XY plane, normals should be [0, 0, 1]
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(normals[0], expected, atol=1e-6)

    def test_with_vertex_weights(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        prim = IndexedMeshTriangle(vertices, faces, vertex_weights=weights)
        stats = prim.extra_stats()
        assert "vertex_weight_mean" in stats

    def test_with_sigma(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces, sigma=2.5)
        stats = prim.extra_stats()
        assert stats["sigma"] == 2.5

    def test_to_tensor(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        tensor = prim.to_tensor()
        assert tensor.shape == (2, 3, 3)

    def test_to_device(self, dummy_mesh_data):
        vertices, faces = dummy_mesh_data
        prim = IndexedMeshTriangle(vertices, faces)
        prim_cpu = prim.to("cpu")
        assert prim_cpu.vertices.device == torch.device("cpu")


class TestConvexTriangle:
    """Tests for ConvexTriangle (Triangle Splatting-style)."""

    def test_creation(self, dummy_triangle_vertices):
        T = dummy_triangle_vertices.shape[0]
        points = dummy_triangle_vertices.mean(dim=1)  # centroids
        sigma = torch.ones(T)
        prim = ConvexTriangle(points, dummy_triangle_vertices, sigma)
        assert prim.num_primitives == T
        assert prim.num_vertices == T * 3

    def test_sigma_stats(self, dummy_triangle_vertices):
        T = dummy_triangle_vertices.shape[0]
        points = dummy_triangle_vertices.mean(dim=1)
        sigma = torch.rand(T) * 3.0
        prim = ConvexTriangle(points, dummy_triangle_vertices, sigma)
        stats = prim.extra_stats()
        assert "sigma_mean" in stats
        assert "sigma_std" in stats
        assert "sigma_min" in stats
        assert "sigma_max" in stats

    def test_compute_areas(self, dummy_triangle_vertices):
        T = dummy_triangle_vertices.shape[0]
        points = dummy_triangle_vertices.mean(dim=1)
        sigma = torch.ones(T)
        prim = ConvexTriangle(points, dummy_triangle_vertices, sigma)
        areas = prim.compute_areas()
        assert areas.shape == (T,)

    def test_to_device(self, dummy_triangle_vertices):
        T = dummy_triangle_vertices.shape[0]
        points = dummy_triangle_vertices.mean(dim=1)
        sigma = torch.ones(T)
        prim = ConvexTriangle(points, dummy_triangle_vertices, sigma)
        prim_cpu = prim.to("cpu")
        assert prim_cpu.points.device == torch.device("cpu")
