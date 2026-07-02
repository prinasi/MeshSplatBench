"""Tests for the dual point cloud geometry export module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tribench.core.viewer_geometry import (
    GeometryExportConfig,
    _colors_from_dc,
    _extract_indexed_mesh_from_primitive,
    _extract_mesh_from_primitive,
    _opacity_colormap,
    _primitive_triangle_vertices,
    _sample_triangle_surface,
    _voxel_downsample,
    export_original_geometry,
    export_viewer_point_cloud,
    export_viewer_geometry,
    primitive_to_point_cloud,
    write_mesh_obj,
    write_mesh_ply,
    write_point_cloud_ply,
)
from tribench.primitives.mesh_triangle import IndexedMeshTriangle
from tribench.primitives.triangle import IndependentTriangle


# ─── Test fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simple_triangles():
    """10 random triangles with opacity and SH colors."""
    torch.manual_seed(42)
    verts = torch.randn(10, 3, 3)
    opacity = torch.rand(10)
    sh = torch.rand(10, 3)  # DC SH coefficients
    return IndependentTriangle(
        vertices=verts, opacity=opacity, sh_coeffs=sh,
    )


@pytest.fixture
def known_triangle():
    """A single unit triangle in the XY plane."""
    verts = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ], dtype=torch.float32)
    return IndependentTriangle(vertices=verts)


@pytest.fixture
def indexed_mesh():
    """A square mesh represented with shared vertices and two faces."""
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    opacity = torch.tensor([0.25, 0.75], dtype=torch.float32)
    return IndexedMeshTriangle(vertices=vertices, faces=faces, opacity=opacity)


# ─── _sample_triangle_surface tests ───────────────────────────────────────


class TestSampleTriangleSurface:
    def test_point_count(self, known_triangle):
        verts = known_triangle.vertices
        points, tri_ids = _sample_triangle_surface(verts, 1000)
        assert points.shape == (1000, 3)
        assert tri_ids.shape == (1000,)

    def test_points_on_surface(self, known_triangle):
        verts = known_triangle.vertices
        points, _ = _sample_triangle_surface(verts, 1000)
        # All points should have z ≈ 0 (triangle in XY plane)
        assert torch.all(points[:, 2].abs() < 1e-5)

    def test_barycentric_validity(self, known_triangle):
        """All sampled points must be inside the triangle (barycentric >= 0)."""
        verts = known_triangle.vertices
        points, _ = _sample_triangle_surface(verts, 10000)
        v0, v1, v2 = verts[0, 0], verts[0, 1], verts[0, 2]
        # Compute barycentric coordinates
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = points - v0.unsqueeze(0)
        d00 = torch.dot(v0v1, v0v1)
        d01 = torch.dot(v0v1, v0v2)
        d11 = torch.dot(v0v2, v0v2)
        d20 = (v0p * v0v1.unsqueeze(0)).sum(-1)
        d21 = (v0p * v0v2.unsqueeze(0)).sum(-1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        # All barycentric coords should be >= -epsilon
        assert (u >= -1e-4).all()
        assert (v >= -1e-4).all()
        assert (w >= -1e-4).all()

    def test_area_weighting(self):
        """Larger triangles should get proportionally more points."""
        verts = torch.tensor([
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],  # Small triangle
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],  # Large triangle
        ], dtype=torch.float32)
        gen = torch.Generator().manual_seed(42)
        _, tri_ids = _sample_triangle_surface(verts, 10000, generator=gen)
        # Large triangle (area=50) vs small (area=0.005) → ~99.99% should be from large
        large_fraction = (tri_ids == 1).float().mean()
        assert large_fraction > 0.99

    def test_deterministic_with_generator(self, known_triangle):
        verts = known_triangle.vertices
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        p1, _ = _sample_triangle_surface(verts, 100, generator=gen1)
        p2, _ = _sample_triangle_surface(verts, 100, generator=gen2)
        torch.testing.assert_close(p1, p2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_vertices_with_cpu_generator(self, known_triangle):
        verts = known_triangle.vertices.cuda()
        gen = torch.Generator().manual_seed(123)
        points, tri_ids = _sample_triangle_surface(verts, 100, generator=gen)
        assert points.device.type == "cuda"
        assert tri_ids.device.type == "cuda"


# ─── write_point_cloud_ply tests ──────────────────────────────────────────


class TestWritePointCloudPLY:
    def test_xyz_only(self, tmp_path):
        xyz = np.random.randn(100, 3).astype(np.float32)
        path = tmp_path / "cloud.ply"
        write_point_cloud_ply(path, xyz, None, None)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert data["vertex"].count == 100
        assert "x" in data["vertex"].data.dtype.names
        assert "red" not in data["vertex"].data.dtype.names

    def test_with_color(self, tmp_path):
        xyz = np.random.randn(50, 3).astype(np.float32)
        rgb = np.random.randint(0, 256, (50, 3), dtype=np.uint8)
        path = tmp_path / "cloud_color.ply"
        write_point_cloud_ply(path, xyz, rgb, None)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert data["vertex"].count == 50
        assert "red" in data["vertex"].data.dtype.names
        assert "green" in data["vertex"].data.dtype.names

    def test_with_normals(self, tmp_path):
        xyz = np.random.randn(30, 3).astype(np.float32)
        nrm = np.random.randn(30, 3).astype(np.float32)
        path = tmp_path / "cloud_nrm.ply"
        write_point_cloud_ply(path, xyz, None, nrm)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert "nx" in data["vertex"].data.dtype.names
        assert "nz" in data["vertex"].data.dtype.names

    def test_full_schema(self, tmp_path):
        xyz = np.random.randn(20, 3).astype(np.float32)
        rgb = np.random.randint(0, 256, (20, 3), dtype=np.uint8)
        nrm = np.random.randn(20, 3).astype(np.float32)
        path = tmp_path / "cloud_full.ply"
        write_point_cloud_ply(path, xyz, rgb, nrm)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        names = data["vertex"].data.dtype.names
        for n in ["x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"]:
            assert n in names


# ─── write_mesh_ply tests ─────────────────────────────────────────────────


class TestWriteMeshPLY:
    def test_basic_mesh(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        path = tmp_path / "mesh.ply"
        write_mesh_ply(path, verts, faces)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert data["vertex"].count == 3
        assert data["face"].count == 1

    def test_with_color(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        path = tmp_path / "mesh_color.ply"
        write_mesh_ply(path, verts, faces, rgb)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert "red" in data["vertex"].data.dtype.names

    def test_with_normals(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        nrm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        path = tmp_path / "mesh_nrm.ply"
        write_mesh_ply(path, verts, faces, None, nrm)
        from plyfile import PlyData
        data = PlyData.read(str(path))
        assert "nx" in data["vertex"].data.dtype.names


class TestWriteMeshOBJ:
    def test_obj_and_mtl_with_vertex_colors(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        path = tmp_path / "mesh.obj"

        mtl_path = write_mesh_obj(path, verts, faces, rgb)

        obj_text = path.read_text()
        mtl_text = Path(mtl_path).read_text()
        assert "mtllib mesh.mtl" in obj_text
        assert "usemtl mat_" in obj_text
        assert "v 0" in obj_text
        assert "f 1 2 3" in obj_text
        assert "newmtl mat_" in mtl_text
        assert "Kd " in mtl_text


# ─── primitive_to_point_cloud tests ───────────────────────────────────────


class TestPrimitiveToPointCloud:
    def test_basic_sampling(self, simple_triangles):
        xyz, rgb, nrm, tri_ids = primitive_to_point_cloud(
            simple_triangles, 500, color_mode="dc", normal_mode="none",
        )
        assert xyz.shape == (500, 3)
        assert tri_ids.shape == (500,)
        # DC colors should be present since sh_coeffs were provided
        assert rgb is not None
        assert rgb.shape == (500, 3)

    def test_opacity_colormap(self, simple_triangles):
        xyz, rgb, _, _ = primitive_to_point_cloud(
            simple_triangles, 500, color_mode="opacity",
        )
        assert rgb is not None
        assert rgb.shape == (500, 3)
        assert (rgb >= 0).all() and (rgb <= 1).all()

    def test_white_color(self, known_triangle):
        xyz, rgb, _, _ = primitive_to_point_cloud(
            known_triangle, 100, color_mode="white",
        )
        assert rgb is not None
        assert (rgb == 1.0).all()

    def test_normal_mode_primitive(self, simple_triangles):
        _, _, nrm, _ = primitive_to_point_cloud(
            simple_triangles, 500, color_mode="white", normal_mode="primitive",
        )
        assert nrm is not None
        assert nrm.shape == (500, 3)

    def test_normal_mode_none(self, simple_triangles):
        _, _, nrm, _ = primitive_to_point_cloud(
            simple_triangles, 500, normal_mode="none",
        )
        assert nrm is None

    def test_indexed_mesh_sampling(self, indexed_mesh):
        xyz, rgb, nrm, tri_ids = primitive_to_point_cloud(
            indexed_mesh, 200, color_mode="opacity", normal_mode="primitive",
        )
        assert xyz.shape == (200, 3)
        assert rgb is not None and rgb.shape == (200, 3)
        assert nrm is not None and nrm.shape == (200, 3)
        assert tri_ids.max().item() < indexed_mesh.num_primitives

    def test_indexed_mesh_triangle_vertices(self, indexed_mesh):
        tri_verts = _primitive_triangle_vertices(indexed_mesh)
        assert tri_verts.shape == (2, 3, 3)
        torch.testing.assert_close(tri_verts[0], indexed_mesh.vertices[indexed_mesh.faces[0]])

    def test_deterministic(self, simple_triangles):
        gen1 = torch.Generator().manual_seed(99)
        gen2 = torch.Generator().manual_seed(99)
        xyz1, _, _, _ = primitive_to_point_cloud(
            simple_triangles, 100, generator=gen1,
        )
        xyz2, _, _, _ = primitive_to_point_cloud(
            simple_triangles, 100, generator=gen2,
        )
        torch.testing.assert_close(xyz1, xyz2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_mixed_device_primitive_exports_on_cuda(self):
        torch.manual_seed(7)
        primitive = IndependentTriangle(
            vertices=torch.randn(8, 3, 3, device="cuda"),
            opacity=torch.rand(8),
            sh_coeffs=torch.rand(8, 3),
        )
        xyz, rgb, nrm, tri_ids = primitive_to_point_cloud(
            primitive,
            64,
            color_mode="dc",
            normal_mode="primitive",
            generator=torch.Generator().manual_seed(7),
        )
        assert xyz.device.type == "cuda"
        assert rgb is not None and rgb.device.type == "cuda"
        assert nrm is not None and nrm.device.type == "cuda"
        assert tri_ids.device.type == "cuda"


# ─── _opacity_colormap tests ──────────────────────────────────────────────


class TestOpacityColormap:
    def test_shape(self):
        op = torch.rand(100)
        rgb = _opacity_colormap(op)
        assert rgb.shape == (100, 3)

    def test_range(self):
        op = torch.rand(100)
        rgb = _opacity_colormap(op)
        assert (rgb >= 0).all() and (rgb <= 1).all()

    def test_zero_opacity(self):
        op = torch.tensor([0.0])
        rgb = _opacity_colormap(op)
        # Blue-ish for zero
        assert rgb[0, 2] > rgb[0, 0]

    def test_one_opacity(self):
        op = torch.tensor([1.0])
        rgb = _opacity_colormap(op)
        # Red-ish for one
        assert rgb[0, 0] > rgb[0, 2]


# ─── _colors_from_dc tests ────────────────────────────────────────────────


class TestColorsFromDC:
    def test_valid_sh(self):
        sh = torch.rand(10, 3)
        dc = _colors_from_dc(sh, 10)
        assert dc is not None
        assert dc.shape == (10, 3)

    def test_none_sh(self):
        assert _colors_from_dc(None, 10) is None

    def test_short_sh(self):
        sh = torch.rand(10, 2)
        assert _colors_from_dc(sh, 10) is None


# ─── _voxel_downsample tests ──────────────────────────────────────────────


class TestVoxelDownsample:
    def test_reduces_points(self):
        # Two clusters of points
        pts = torch.cat([torch.randn(500, 3) * 0.01, torch.randn(500, 3) * 0.01 + 5.0])
        result, _, _ = _voxel_downsample(pts, None, None, voxel_size=0.5)
        assert result.shape[0] < pts.shape[0]

    def test_preserves_none_colors(self):
        pts = torch.randn(100, 3)
        result, colors, normals = _voxel_downsample(pts, None, None, voxel_size=1.0)
        assert colors is None
        assert normals is None


# ─── _extract_mesh_from_primitive tests ───────────────────────────────────


class TestExtractMesh:
    def test_triangle_soup(self, simple_triangles):
        result = _extract_mesh_from_primitive(simple_triangles)
        assert result is not None
        verts, faces, rgb, nrm = result
        N = simple_triangles.num_primitives
        assert verts.shape == (N * 3, 3)
        assert faces.shape == (N, 3)
        # Faces should be sequential indices
        expected = np.arange(N * 3, dtype=np.int32).reshape(N, 3)
        np.testing.assert_array_equal(faces, expected)

    def test_no_vertices_returns_none(self):
        class NoVerts:
            pass
        assert _extract_mesh_from_primitive(NoVerts()) is None


# ─── _extract_indexed_mesh_from_primitive tests ──────────────────────────


class TestExtractIndexedMesh:
    def test_fallback_to_triangle_soup(self, simple_triangles):
        """Without V/F attributes, should fall back to triangle-soup extraction."""
        result = _extract_indexed_mesh_from_primitive(simple_triangles)
        assert result is not None
        verts, faces, _, _ = result
        N = simple_triangles.num_primitives
        assert verts.shape == (N * 3, 3)

    def test_indexed_mesh_vertices_faces(self, indexed_mesh):
        result = _extract_indexed_mesh_from_primitive(indexed_mesh)
        assert result is not None
        verts, faces, _, _ = result
        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)


# ─── export_viewer_geometry tests ─────────────────────────────────────────


class TestExportViewerGeometry:
    def test_full_export(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=1000,
            cg_num_points=2000,
            color_mode="dc",
            normal_mode="none",
            export_mesh=True,
            export_glb=False,
            voxel_downsample=False,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            method_name="triangle-splatting",
            checkpoint_path="/fake/path",
            config=cfg,
            generator=gen,
        )
        assert result.num_viewer_points > 0
        assert result.num_cg_points > 0
        assert result.has_mesh
        # Files should exist
        from pathlib import Path
        assert Path(result.viewer_ply).exists()
        assert Path(result.cg_ply).exists()
        assert Path(result.mesh_ply).exists()
        assert Path(result.mesh_obj_path).exists()
        assert Path(result.metadata_path).exists()
        assert Path(result.attributes_path).exists()

    def test_metadata_content(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=500,
            cg_num_points=1000,
            voxel_downsample=False,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            method_name="test-method",
            config=cfg,
            generator=gen,
        )
        with open(result.metadata_path) as f:
            meta = json.load(f)
        assert meta["method"] == "test-method"
        assert meta["schema_version"] == "1.0"
        assert meta["coordinate_system"] == "tribench_world"
        assert meta["up_axis"] == "+y"
        assert meta["num_viewer_points"] == result.num_viewer_points
        assert meta["num_cg_points"] == result.num_cg_points
        assert "area_stats" in meta
        assert "total" in meta["area_stats"]

    def test_attributes_npz(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=500,
            cg_num_points=1000,
            voxel_downsample=False,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            config=cfg,
            generator=gen,
        )
        attrs = np.load(result.attributes_path)
        assert "opacity" in attrs
        assert "area" in attrs
        assert "primitive_id" in attrs
        assert "cg_tri_ids" in attrs

    def test_viewer_cloud_smaller_than_cg(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=500,
            cg_num_points=2000,
            voxel_downsample=False,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            config=cfg,
            generator=gen,
        )
        # CG cloud should have more points than viewer (before voxel downsampling)
        assert result.num_cg_points >= result.num_viewer_points

    def test_no_mesh_when_disabled(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=500,
            cg_num_points=1000,
            export_mesh=False,
            voxel_downsample=False,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            config=cfg,
            generator=gen,
        )
        assert not result.has_mesh
        assert result.mesh_ply is None
        assert result.mesh_obj_path is None

    def test_voxel_downsample_reduces_points(self, simple_triangles, tmp_path):
        cfg = GeometryExportConfig(
            viewer_num_points=5000,
            cg_num_points=10000,
            voxel_downsample=True,
            voxel_size=0.5,
        )
        gen = torch.Generator().manual_seed(42)
        result = export_viewer_geometry(
            simple_triangles, tmp_path,
            config=cfg,
            generator=gen,
        )
        # Voxel downsampling should reduce viewer cloud below target
        assert result.num_viewer_points <= 5000


class TestExportViewerPointCloud:
    def test_exports_only_viewer_pointcloud(self, simple_triangles, tmp_path):
        result = export_viewer_point_cloud(simple_triangles, tmp_path)

        assert Path(result.viewer_ply).exists()
        assert Path(result.metadata_path).exists()
        assert not (tmp_path / "geometry_cg_points.ply").exists()


class TestExportOriginalGeometry:
    def test_triangle_soup_export_keeps_original_counts(self, simple_triangles, tmp_path):
        result = export_original_geometry(
            simple_triangles,
            tmp_path,
            method_name="triangle-splatting",
            checkpoint_path="/fake/path",
        )

        assert result.num_vertices == simple_triangles.num_vertices
        assert result.num_faces == simple_triangles.num_primitives
        from plyfile import PlyData

        data = PlyData.read(result.geometry_ply)
        assert data["vertex"].count == simple_triangles.num_vertices
        assert data["face"].count == simple_triangles.num_primitives
        with open(result.metadata_path) as f:
            meta = json.load(f)
        assert meta["export_mode"] == "original"
        assert meta["sampling"] is None
        assert meta["downsampling"] is None

    def test_indexed_mesh_export_keeps_shared_topology(self, indexed_mesh, tmp_path):
        result = export_original_geometry(indexed_mesh, tmp_path)

        assert result.num_vertices == indexed_mesh.num_vertices
        assert result.num_faces == indexed_mesh.num_primitives

    def test_obj_export_is_primary_geometry_when_requested(self, simple_triangles, tmp_path):
        result = export_original_geometry(
            simple_triangles,
            tmp_path,
            filename="geometry_original.obj",
        )

        assert result.geometry_path.endswith(".obj")
        assert result.geometry_ply is None
        assert result.has_obj
        assert Path(result.mesh_obj_path).exists()
        assert Path(result.mesh_obj_path).with_suffix(".mtl").exists()
        with open(result.metadata_path) as f:
            meta = json.load(f)
        assert meta["assets"]["geometry"] == "geometry_original.obj"
        assert meta["assets"]["geometry_format"] == "obj"
