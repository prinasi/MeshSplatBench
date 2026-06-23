"""Tests for evaluation metrics."""

from types import SimpleNamespace

import numpy as np
import torch
import pytest
from PIL import Image

from tribench.core.metrics import compute_psnr
from tribench.core.datasets import DatasetBase, DatasetSample
from tribench.core.rendering import _metric_rgb_for_sample, render_dataset_split
from tribench.core.mesh_eval import _official_dtu_sample_geometry, _resolve_dtu_geometry_mode
from tribench.renderers.base import RendererAdapter
from tribench.renderers.base import RenderOutput


class TestPSNR:
    """Tests for PSNR computation."""

    def test_identical_images(self):
        img = torch.rand(48, 64, 3)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self):
        torch.manual_seed(42)
        img1 = torch.rand(48, 64, 3)
        img2 = torch.rand(48, 64, 3)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float)
        assert psnr > 0
        assert psnr < 100  # PSNR should be finite for random images

    def test_batch_input(self):
        torch.manual_seed(42)
        img1 = torch.rand(2, 48, 64, 3)
        img2 = torch.rand(2, 48, 64, 3)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float)
        assert psnr > 0

    def test_nearly_identical(self):
        img = torch.rand(48, 64, 3)
        noise = img + 1e-6 * torch.rand_like(img)
        psnr = compute_psnr(img, noise)
        assert psnr > 50  # Should be very high for nearly identical images

    def test_zero_images(self):
        img = torch.zeros(48, 64, 3)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")


class TestSSIMLPIPS:
    """Tests for SSIM and LPIPS (require external packages)."""

    def test_ssim_import(self):
        from tribench.core.metrics import compute_ssim
        assert callable(compute_ssim)

    def test_lpips_import(self):
        from tribench.core.metrics import compute_lpips
        assert callable(compute_lpips)

    def test_compute_all_metrics_import(self):
        from tribench.core.metrics import compute_all_metrics
        assert callable(compute_all_metrics)

    def test_lpips_default_matches_triangle_splatting(self):
        import inspect

        from tribench.core.metrics import compute_lpips

        assert inspect.signature(compute_lpips).parameters["net"].default == "vgg"


class TestMetricPreprocessing:
    """Tests for image normalization before metric computation."""

    def test_alpha_render_is_recomposited_to_dtu_white_background(self):
        output = RenderOutput(
            rgb=torch.zeros(1, 1, 3),
            alpha=torch.zeros(1, 1),
            extras={"background_color": [0.0, 0.0, 0.0]},
        )
        sample = DatasetSample(
            camera=None,
            image=torch.ones(1, 1, 3),
            name="000000",
            metadata={"eval_background_color": (1.0, 1.0, 1.0)},
        )

        pred, target = _metric_rgb_for_sample(output, sample)

        assert torch.allclose(pred, torch.ones(1, 1, 3))
        assert torch.allclose(target, torch.ones(1, 1, 3))

    def test_dtu_mask_forces_render_background_to_white(self):
        output = RenderOutput(
            rgb=torch.zeros(1, 2, 3),
            alpha=torch.ones(1, 2),
            extras={"background_color": [1.0, 1.0, 1.0]},
        )
        sample = DatasetSample(
            camera=None,
            image=torch.zeros(1, 2, 3),
            mask=torch.tensor([[1.0, 0.0]]),
            name="000000",
            metadata={"eval_background_color": (1.0, 1.0, 1.0)},
        )

        pred, target = _metric_rgb_for_sample(output, sample)

        assert torch.allclose(pred[0, 0], torch.zeros(3))
        assert torch.allclose(pred[0, 1], torch.ones(3))
        assert torch.allclose(target[0, 1], torch.ones(3))

    def test_camera_metadata_triggers_dtu_mask_background(self):
        class DummyCamera:
            metadata = {"eval_background_color": (1.0, 1.0, 1.0)}

        output = RenderOutput(
            rgb=torch.zeros(1, 1, 3),
            alpha=torch.ones(1, 1),
            extras={"background_color": [1.0, 1.0, 1.0]},
        )
        sample = DatasetSample(
            camera=DummyCamera(),
            image=torch.zeros(1, 1, 3),
            mask=torch.zeros(1, 1),
            name="000000",
        )

        pred, target = _metric_rgb_for_sample(output, sample)

        assert torch.allclose(pred[0, 0], torch.ones(3))
        assert torch.allclose(target[0, 0], torch.ones(3))

    def test_saved_render_uses_dtu_white_background(self, tmp_path):
        class DummyCamera:
            def to(self, device):
                return self

        class DummyDataset(DatasetBase):
            def __init__(self):
                super().__init__(tmp_path, "test")

            def __len__(self):
                return 1

            def sample(self, idx):
                return DatasetSample(
                    camera=DummyCamera(),
                    image=torch.ones(1, 1, 3),
                    mask=torch.zeros(1, 1),
                    name="000000",
                    metadata={"eval_background_color": (1.0, 1.0, 1.0)},
                )

        class DummyAdapter(RendererAdapter):
            @property
            def name(self):
                return "dummy"

            def load_checkpoint(self, path):
                raise NotImplementedError

            def load_scene(self, dataset_path, split="test"):
                raise NotImplementedError

            def model_stats(self):
                return {}

            def render(self, cameras, *, mode="eval"):
                return RenderOutput(
                    rgb=torch.zeros(1, 1, 3),
                    alpha=torch.zeros(1, 1),
                    extras={"background_color": [0.0, 0.0, 0.0]},
                )

        manifest = render_dataset_split(
            DummyAdapter(),
            DummyDataset(),
            tmp_path / "out",
            device="cpu",
            metrics=False,
        )
        render_path = manifest["frames"][0]["render_path"]

        assert Image.open(render_path).getpixel((0, 0)) == (255, 255, 255)


class TestDTUGeometryMode:
    """Tests for DTU mesh/point-cloud interpretation."""

    def test_auto_mode_treats_2dts_pcd_exports_as_point_clouds(self):
        assert _resolve_dtu_geometry_mode("mesh_ply/30000_pcd.ply") == "pcd"
        assert _resolve_dtu_geometry_mode("fuse_post.ply") == "mesh"

    def test_pcd_mode_ignores_faces_during_official_sampling(self):
        obj = SimpleNamespace(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )

        pcd_points = _official_dtu_sample_geometry(obj, 1.0, geometry_mode="pcd")
        mesh_points = _official_dtu_sample_geometry(obj, 1.0, geometry_mode="mesh")

        assert np.array_equal(pcd_points, obj.vertices)
        assert mesh_points.shape[0] > obj.vertices.shape[0]
