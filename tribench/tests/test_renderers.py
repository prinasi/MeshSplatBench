"""Tests for renderer adapters."""

import pytest
import torch

from tribench.renderers.base import RendererAdapter, RenderOutput
from tribench.renderers.d2ts_adapter import D2TSAdapter
from tribench.renderers.triangle_splatting_adapter import TriangleSplattingAdapter
from tribench.renderers.mesh_splatting_adapter import MeshSplattingAdapter
from tribench.renderers.diffsoup_adapter import DiffSoupAdapter


# CUDA availability check
_has_cuda = torch.cuda.is_available()

# Check if triangle-splatting CUDA extension is importable
_has_ts_cuda = False
try:
    import importlib
    spec = importlib.util.find_spec("diff_triangle_rasterization")
    _has_ts_cuda = spec is not None
except Exception:
    pass


class TestRenderOutput:
    """Tests for RenderOutput dataclass."""

    def test_minimal_creation(self):
        rgb = torch.rand(48, 64, 3)
        output = RenderOutput(rgb=rgb)
        assert output.rgb.shape == (48, 64, 3)
        assert output.alpha is None
        assert output.depth is None
        assert output.normal is None
        assert output.radii is None
        assert output.visibility is None
        assert output.extras is None

    def test_full_creation(self):
        output = RenderOutput(
            rgb=torch.rand(48, 64, 3),
            alpha=torch.rand(48, 64),
            depth=torch.rand(48, 64),
            normal=torch.rand(48, 64, 3),
            radii=torch.rand(100),
            visibility=torch.ones(100),
            extras={"contributions": 42},
        )
        assert output.extras["contributions"] == 42


class TestStubAdapterNames:
    """Tests that stub adapters report correct names."""

    @pytest.mark.parametrize("adapter_cls,name", [
        (D2TSAdapter, "2dts"),
        (TriangleSplattingAdapter, "triangle-splatting"),
        (MeshSplattingAdapter, "mesh-splatting"),
        (DiffSoupAdapter, "diffsoup"),
    ])
    def test_adapter_name(self, adapter_cls, name):
        adapter = adapter_cls()
        assert adapter.name == name


class TestPartiallyImplementedAdapters:
    """Tests for adapter surfaces that still need dataset loader mapping."""

    @pytest.mark.parametrize("adapter_cls", [
        D2TSAdapter,
        MeshSplattingAdapter,
        DiffSoupAdapter,
    ])
    def test_load_scene_raises(self, adapter_cls):
        adapter = adapter_cls()
        with pytest.raises(NotImplementedError):
            adapter.load_scene("/fake/path")


class TestBackendAwareAdapters:
    """Tests for adapters that now expose backend diagnostics."""

    @pytest.mark.parametrize("adapter_cls,name", [
        (D2TSAdapter, "2dts"),
        (TriangleSplattingAdapter, "triangle-splatting"),
        (MeshSplattingAdapter, "mesh-splatting"),
        (DiffSoupAdapter, "diffsoup"),
    ])
    def test_backend_status_shape(self, adapter_cls, name):
        status = adapter_cls().backend_status()
        assert status["name"] == name
        assert "repo_root" in status
        assert "modules" in status

    @pytest.mark.parametrize("adapter_cls", [
        D2TSAdapter,
        MeshSplattingAdapter,
        DiffSoupAdapter,
    ])
    def test_model_stats_no_model_raises(self, adapter_cls):
        adapter = adapter_cls()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.model_stats()

    @pytest.mark.parametrize("adapter_cls", [
        D2TSAdapter,
        MeshSplattingAdapter,
        DiffSoupAdapter,
    ])
    def test_render_no_model_raises(self, adapter_cls, dummy_camera_batch):
        adapter = adapter_cls()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.render(dummy_camera_batch)

    @pytest.mark.parametrize("adapter_cls", [
        D2TSAdapter,
        MeshSplattingAdapter,
        DiffSoupAdapter,
    ])
    def test_to_primitive_no_model_raises(self, adapter_cls):
        adapter = adapter_cls()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.to_primitive()


class TestD2TSAdapter:
    """Tests for 2DTS checkpoint discovery fallbacks."""

    def test_load_checkpoint_falls_back_to_point_cloud_ply(self, tmp_path):
        run_dir = tmp_path / "scan24"
        point_cloud = run_dir / "point_cloud"
        point_cloud.mkdir(parents=True)
        ply = point_cloud / "30000.ply"
        ply.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float x1\n"
            "property float y1\n"
            "property float z1\n"
            "property float x2\n"
            "property float y2\n"
            "property float z2\n"
            "property float x3\n"
            "property float y3\n"
            "property float z3\n"
            "property float opacity\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "end_header\n"
            "0 0 0 1 0 0 0 1 0 0.5 0.1 0.2 0.3\n"
        )

        adapter = D2TSAdapter()
        adapter.load_checkpoint(str(run_dir))

        assert adapter._checkpoint_path == str(ply)
        assert adapter.model_stats()["primitive_count"] == 1


class TestTriangleSplattingAdapter:
    """Tests for the TriangleSplattingAdapter (real implementation)."""

    def test_creation(self):
        adapter = TriangleSplattingAdapter()
        assert adapter.name == "triangle-splatting"
        assert adapter._model is None
        assert adapter.device == torch.device("cpu")

    def test_custom_repo_root(self, tmp_path):
        adapter = TriangleSplattingAdapter(repo_root=str(tmp_path))
        assert adapter.repo_root == tmp_path

    def test_configure_bg_color_overrides_checkpoint_background(self):
        adapter = TriangleSplattingAdapter()
        adapter.configure(bg_color="white")

        assert adapter._background_color == [1.0, 1.0, 1.0]
        assert adapter._background_color_override == [1.0, 1.0, 1.0]

    def test_model_stats_no_model_raises(self):
        adapter = TriangleSplattingAdapter()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.model_stats()

    def test_render_no_model_raises(self, dummy_camera_batch):
        adapter = TriangleSplattingAdapter()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.render(dummy_camera_batch)

    def test_to_primitive_no_model_raises(self):
        adapter = TriangleSplattingAdapter()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.to_primitive()

    def test_load_checkpoint_bad_path(self):
        """load_checkpoint with nonexistent path raises ImportError or FileNotFoundError."""
        adapter = TriangleSplattingAdapter(repo_root="/nonexistent/repo")
        with pytest.raises(ImportError):
            adapter.load_checkpoint("/nonexistent/checkpoint")

    @pytest.mark.skipif(not (_has_cuda and _has_ts_cuda),
                        reason="Requires CUDA + built diff-triangle-rasterization")
    def test_load_checkpoint_from_state_dict(self, tmp_path):
        """Test loading a synthetic checkpoint through the full pipeline."""
        import math

        # Create a minimal synthetic checkpoint
        N = 100  # number of triangles
        sh_degree = 3
        num_sh_coeffs = (sh_degree + 1) ** 2  # 16

        state_dict = {
            "triangles_points": torch.randn(N, 3, 3),
            "sigma": torch.randn(N, 1),
            "features_dc": torch.randn(N, 1, 3),
            "features_rest": torch.randn(N, num_sh_coeffs - 1, 3),
            "opacity": torch.randn(N, 1),
            "active_sh_degree": sh_degree,
        }

        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        torch.save(state_dict, ckpt_dir / "point_cloud_state_dict.pt")

        adapter = TriangleSplattingAdapter()
        adapter.load_checkpoint(str(ckpt_dir))

        assert adapter._model is not None
        assert adapter._model._triangles_points.shape == (N, 3, 3)
        assert adapter._model.active_sh_degree == sh_degree
        assert adapter._model.max_sh_degree == sh_degree

    @pytest.mark.skipif(not (_has_cuda and _has_ts_cuda),
                        reason="Requires CUDA + built diff-triangle-rasterization")
    def test_model_stats_after_load(self, tmp_path):
        """Test model_stats returns correct fields after loading."""
        N = 50
        sh_degree = 1
        num_sh_coeffs = (sh_degree + 1) ** 2

        state_dict = {
            "triangles_points": torch.randn(N, 3, 3),
            "sigma": torch.randn(N, 1),
            "features_dc": torch.randn(N, 1, 3),
            "features_rest": torch.randn(N, num_sh_coeffs - 1, 3),
            "opacity": torch.randn(N, 1),
            "active_sh_degree": sh_degree,
        }

        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        torch.save(state_dict, ckpt_dir / "point_cloud_state_dict.pt")

        adapter = TriangleSplattingAdapter()
        adapter.load_checkpoint(str(ckpt_dir))
        stats = adapter.model_stats()

        assert stats["method"] == "triangle-splatting"
        assert stats["primitive_count"] == N
        assert stats["vertex_count"] == N * 3
        assert stats["face_count"] == N
        assert stats["trainable_param_count"] > 0
        assert "sigma_mean" in stats["backend_extra"]
        assert stats["backend_extra"]["max_sh_degree"] == sh_degree

    @pytest.mark.skipif(not (_has_cuda and _has_ts_cuda),
                        reason="Requires CUDA + built diff-triangle-rasterization")
    def test_to_primitive_after_load(self, tmp_path):
        """Test to_primitive returns valid ConvexTriangle after loading."""
        from tribench.primitives.convex_triangle import ConvexTriangle

        N = 20
        sh_degree = 0
        num_sh_coeffs = 1

        state_dict = {
            "triangles_points": torch.randn(N, 3, 3),
            "sigma": torch.randn(N, 1),
            "features_dc": torch.randn(N, 1, 3),
            "features_rest": torch.randn(N, num_sh_coeffs - 1, 3),
            "opacity": torch.randn(N, 1),
            "active_sh_degree": sh_degree,
        }

        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        torch.save(state_dict, ckpt_dir / "point_cloud_state_dict.pt")

        adapter = TriangleSplattingAdapter()
        adapter.load_checkpoint(str(ckpt_dir))
        prim = adapter.to_primitive()

        assert isinstance(prim, ConvexTriangle)
        assert prim.num_primitives == N
        assert prim.sigma.shape == (N,)
        assert prim.opacity.shape == (N,)


class TestRendererAdapterBase:
    """Tests for RendererAdapter abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RendererAdapter()

    def test_device_default(self):
        adapter = D2TSAdapter()
        assert adapter.device == torch.device("cpu")
