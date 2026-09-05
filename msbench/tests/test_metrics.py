"""Tests for evaluation metrics."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import pytest
from PIL import Image

from msbench.core.metrics import compute_psnr
from msbench.core.cameras import CameraBatch
from msbench.core.datasets import DatasetBase, DatasetSample
from msbench.core.rendering import (
    compute_metrics_from_render_dir,
    _metric_rgb_for_sample,
    generate_ellipse_cameras,
    render_dataset_split,
    render_video,
)
from msbench.core.mesh_eval import _official_dtu_sample_geometry, _resolve_dtu_geometry_mode
from msbench.renderers.base import RendererAdapter
from msbench.renderers.base import RenderOutput


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


class TestRenderMetricCache:
    def test_checkpoint_mismatch_invalidates_saved_renders(self, tmp_path):
        render_root = tmp_path / "renders_test"
        (render_root / "renders").mkdir(parents=True)
        (render_root / "gt").mkdir()
        Image.new("RGB", (4, 4), (0, 0, 0)).save(
            render_root / "renders" / "00000.png"
        )
        Image.new("RGB", (4, 4), (0, 0, 0)).save(render_root / "gt" / "00000.png")
        (render_root / "manifest.json").write_text(
            """
            {
              "renderer": {
                "method": "triangle-splatting",
                "checkpoint": "/tmp/old/point_cloud_state_dict.pt"
              }
            }
            """,
            encoding="utf-8",
        )

        manifest = compute_metrics_from_render_dir(
            render_root,
            expected_method="triangle-splatting",
            expected_checkpoint=tmp_path / "new" / "iteration_30000",
        )

        assert manifest is None

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
        from msbench.core.metrics import compute_ssim
        assert callable(compute_ssim)

    def test_lpips_import(self):
        from msbench.core.metrics import compute_lpips
        assert callable(compute_lpips)

    def test_compute_all_metrics_import(self):
        from msbench.core.metrics import compute_all_metrics
        assert callable(compute_all_metrics)

    def test_lpips_default_matches_triangle_splatting(self):
        import inspect

        from msbench.core.metrics import compute_lpips

        assert inspect.signature(compute_lpips).parameters["net"].default == "vgg"

    def test_lpips_model_is_cached_per_device(self, monkeypatch):
        from msbench.core import metrics

        class FakeLPIPS:
            instances = 0

            def __init__(self, net, verbose):
                FakeLPIPS.instances += 1
                self.net = net
                self.verbose = verbose

            def to(self, device):
                self.device = torch.device(device)
                return self

            def eval(self):
                return self

            def __call__(self, pred, target):
                return torch.zeros(pred.shape[0], 1, 1, 1, device=pred.device)

        fake_lpips = ModuleType("lpips")
        fake_lpips.LPIPS = FakeLPIPS
        monkeypatch.setitem(sys.modules, "lpips", fake_lpips)
        metrics._LPIPS_MODEL_CACHE.clear()

        pred = torch.zeros(4, 4, 3)
        target = torch.ones(4, 4, 3)

        assert metrics.compute_lpips(pred, target) == 0.0
        assert metrics.compute_lpips(pred, target) == 0.0
        assert FakeLPIPS.instances == 1
        metrics._LPIPS_MODEL_CACHE.clear()


class TestEvalDeviceSelection:
    def test_default_metrics_device_prefers_cuda(self, monkeypatch):
        from msbench.cli import eval as eval_cli

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert eval_cli._default_metrics_device() == "cuda"

    def test_default_metrics_device_falls_back_to_cpu(self, monkeypatch):
        from msbench.cli import eval as eval_cli

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert eval_cli._default_metrics_device() == "cpu"


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

    def test_dataset_mask_does_not_force_render_background(self):
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
        assert torch.allclose(pred[0, 1], torch.zeros(3))
        assert torch.allclose(target[0, 1], torch.zeros(3))

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

    def test_video_render_uses_dtu_white_background(self, tmp_path, monkeypatch):
        imageio = ModuleType("imageio")
        imageio.__path__ = []
        imageio_v3 = ModuleType("imageio.v3")
        written = {}

        def imwrite(path, frames, fps):
            written["path"] = path
            written["frames"] = frames
            written["fps"] = fps

        imageio_v3.imwrite = imwrite
        imageio.v3 = imageio_v3
        monkeypatch.setitem(sys.modules, "imageio", imageio)
        monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)

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

        camera = CameraBatch(
            viewmats=torch.eye(4).unsqueeze(0),
            camtoworlds=torch.eye(4).unsqueeze(0),
            Ks=torch.eye(3).unsqueeze(0),
            width=1,
            height=1,
            metadata={"eval_background_color": (1.0, 1.0, 1.0)},
        )

        render_video(DummyAdapter(), [camera], tmp_path / "video", fps=12, device="cpu")

        assert Image.open(tmp_path / "video" / "frames" / "00000.png").getpixel((0, 0)) == (255, 255, 255)
        assert written["frames"][0, 0, 0].tolist() == [255, 255, 255]
        assert written["fps"] == 12

    def test_ellipse_cameras_preserve_background_metadata(self):
        def look_at_colmap(position):
            position = np.asarray(position, dtype=np.float32)
            forward = -position / (np.linalg.norm(position) + 1e-8)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            cam_up = np.cross(right, forward)
            cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)
            c2w_gl = np.eye(4, dtype=np.float32)
            c2w_gl[:3, 0] = right
            c2w_gl[:3, 1] = cam_up
            c2w_gl[:3, 2] = -forward
            c2w_gl[:3, 3] = position
            flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            return c2w_gl @ flip

        c2ws_np = np.stack(
            [
                look_at_colmap((1.0, 0.0, 0.0)),
                look_at_colmap((0.0, 1.0, 0.0)),
                look_at_colmap((-1.0, 0.0, 0.0)),
                look_at_colmap((0.0, -1.0, 0.0)),
            ]
        )
        c2ws = torch.from_numpy(c2ws_np)
        viewmats = torch.from_numpy(np.linalg.inv(c2ws_np).astype(np.float32))
        ks = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        base = CameraBatch(
            viewmats=viewmats,
            camtoworlds=c2ws,
            Ks=ks,
            width=4,
            height=4,
            metadata={"split": "train", "eval_background_color": (1.0, 1.0, 1.0)},
        )

        cameras = generate_ellipse_cameras(base, n_frames=2)

        assert cameras[0].metadata["eval_background_color"] == (1.0, 1.0, 1.0)
        assert cameras[0].metadata["trajectory"] == "ellipse_pca"


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
