"""Tests for the interactive TriBench viewer backend (nerfbaselines-compatible)."""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from tribench.core.cameras import CameraBatch
from tribench.core.datasets import DatasetBase, DatasetSample
from tribench.core.viewer import (
    ViewerBadRequest,
    ViewerNotFound,
    _build_dataset_json,
    _build_viewer_params,
    _camera_from_request,
    _combine_outputs,
    _display_host,
    _format_output,
    _image_size_from_request,
    _intrinsics_matrix,
    _matrix_to_list,
    _normalize_output_type,
    _parse_image_path,
    _pose_matrix,
    _render_viewer_frame,
    _resize_camera,
    _robust_range,
    _scene_from_dataset,
    _select_output,
    _to_hwc,
    get_viewer_dataset,
    get_viewer_html,
    get_viewer_info,
    render_viewer_frame,
)
from tribench.renderers.base import RenderOutput, RendererAdapter


# ─── Test fixtures ────────────────────────────────────────────────────────


class DummyViewerDataset(DatasetBase):
    def __init__(self):
        super().__init__("/tmp/dummy-scene", split="test")
        self._samples = [
            self._sample(0, [0.0, 0.0, 2.0]),
            self._sample(1, [1.0, 0.0, 2.0]),
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, idx: int) -> DatasetSample:
        return self._samples[idx]

    def _sample(self, idx: int, center: list[float]) -> DatasetSample:
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor(center)
        viewmat = torch.inverse(c2w)
        K = torch.tensor(
            [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        camera = CameraBatch(
            viewmats=viewmat.unsqueeze(0),
            camtoworlds=c2w.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=64,
            height=48,
            metadata={"split": "test", "image_name": f"cam_{idx}.png"},
        )
        image = torch.full((48, 64, 3), 0.2 + idx * 0.2)
        return DatasetSample(
            camera=camera,
            image=image,
            name=f"cam_{idx}",
            metadata={"split": "test", "image_name": f"cam_{idx}.png"},
        )


class DummyViewerAdapter(RendererAdapter):
    @property
    def name(self) -> str:
        return "triangle-splatting"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        raise NotImplementedError

    def model_stats(self) -> dict:
        return {"method": self.name, "primitive_count": 3}

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        height = cameras.height
        width = cameras.width
        x = torch.linspace(0.0, 1.0, width).view(1, width, 1).expand(height, width, 1)
        y = torch.linspace(0.0, 1.0, height).view(height, 1, 1).expand(height, width, 1)
        rgb = torch.cat([x, y, torch.full_like(x, 0.5)], dim=-1)
        alpha = torch.ones(height, width) * 0.75
        depth = x[..., 0] + y[..., 0]
        normal = torch.zeros(height, width, 3)
        normal[..., 2] = 1.0
        return RenderOutput(rgb=rgb, alpha=alpha, depth=depth, normal=normal)


@pytest.fixture
def adapter():
    return DummyViewerAdapter()


@pytest.fixture
def dataset():
    return DummyViewerDataset()


# ─── Public API tests ─────────────────────────────────────────────────────


class TestPublicAPI:
    def test_get_viewer_info_reports_remote_renderer(self, adapter, dataset):
        info = get_viewer_info(adapter, dataset)
        assert info["renderer"]["type"] == "remote"
        assert "websocket_url" in info["renderer"]
        assert "http_url" in info["renderer"]
        assert "depth" in info["renderer"]["output_types"]
        assert info["state"]["output_types"] == info["renderer"]["output_types"]
        assert info["state"]["output_type"] == "color"
        assert info["state"]["split_output_type"] == "color"
        assert info["state"]["camera_control_mode"] == "orbit"

    def test_get_viewer_info_has_method_info(self, adapter, dataset):
        info = get_viewer_info(adapter, dataset)
        assert info["state"]["method_info"]["primitive_count"] == 3

    def test_get_viewer_info_has_scene_metadata(self, adapter, dataset):
        info = get_viewer_info(adapter, dataset)
        di = info["state"]["dataset_info"]
        assert di["type"] == "object-centric"
        assert "scene_center" in di
        assert "scene_radius" in di
        assert "up" in di

    def test_get_viewer_info_has_initial_pose(self, adapter, dataset):
        info = get_viewer_info(adapter, dataset)
        assert "viewer_initial_pose" in info
        assert len(info["viewer_initial_pose"]) == 12
        assert "viewer_initial_target" in info
        assert len(info["viewer_initial_target"]) == 3
        assert "viewer_up" in info
        assert len(info["viewer_up"]) == 3

    def test_get_viewer_dataset_manifest(self, dataset):
        manifest = get_viewer_dataset(dataset)
        cameras = manifest["test"]["cameras"]
        assert len(cameras) == 2
        assert cameras[0]["image_size"] == [64, 48]
        assert cameras[0]["image_name"] == "cam_0.png"
        assert cameras[0]["thumbnail_url"].endswith("?thumb=1")
        assert cameras[0]["image_url"] == "./dataset/images/test/0.jpg"

    def test_get_viewer_dataset_intrinsics_flat(self, dataset):
        manifest = get_viewer_dataset(dataset)
        cam = manifest["test"]["cameras"][0]
        assert len(cam["intrinsics"]) == 4
        fx, fy, cx, cy = cam["intrinsics"]
        assert fx == 80.0
        assert fy == 80.0
        assert cx == 32.0
        assert cy == 24.0

    def test_render_viewer_frame_color(self, adapter, dataset):
        camera = dataset.sample(0).camera
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80.0, 80.0, 32.0, 24.0],
                "image_size": [80, 40],
                "output_type": "color",
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.size == (80, 40)

    def test_render_viewer_frame_depth_colorized(self, adapter, dataset):
        camera = dataset.sample(0).camera
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80.0, 80.0, 32.0, 24.0],
                "image_size": [32, 24],
                "output_type": "depth",
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.mode == "RGB"
            assert img.size == (32, 24)

    def test_render_viewer_frame_jpeg(self, adapter, dataset):
        camera = dataset.sample(0).camera
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80.0, 80.0, 32.0, 24.0],
                "image_size": [32, 24],
                "output_type": "color",
                "lossless": False,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/jpeg"

    def test_get_viewer_html_has_template_replaced(self, adapter, dataset):
        html = get_viewer_html(adapter, dataset)
        assert "{{ data|safe }}" not in html
        assert "renderer" in html
        assert "remote" in html

    def test_viewer_static_uses_standard_array_helpers(self):
        from pathlib import Path

        viewer_js = Path("tribench/core/viewer_static/viewer.js").read_text("utf-8")
        assert ".any(" not in viewer_js
        assert ".matchAll(" in viewer_js
        assert ".map(x => parseFloat(x)).toArray()" not in viewer_js


# ─── Mesh preview mode tests ──────────────────────────────────────────────


class TestMeshPreviewMode:
    def test_mesh_preview_renders_mesh_type(self, adapter, dataset, tmp_path):
        mesh_path = tmp_path / "mesh.ply"
        mesh_path.write_bytes(b"dummy ply")
        info = _build_viewer_params(
            adapter,
            train_dataset=None,
            test_dataset=dataset,
            primary_split="test",
            max_render_size=1280,
            has_pointcloud=True,
            mesh_preview=True,
            mesh_url="./dataset/geometry/mesh.ply",
        )
        assert info["renderer"]["type"] == "mesh"
        assert info["renderer"]["mesh_url"] == "./dataset/geometry/mesh.ply"

    def test_remote_mode_when_no_mesh(self, adapter, dataset):
        info = _build_viewer_params(
            adapter,
            train_dataset=None,
            test_dataset=dataset,
            primary_split="test",
            max_render_size=1280,
            has_pointcloud=True,
            mesh_preview=False,
            mesh_url=None,
        )
        assert info["renderer"]["type"] == "remote"

    def test_mesh_preview_ignored_without_mesh_url(self, adapter, dataset):
        info = _build_viewer_params(
            adapter,
            train_dataset=None,
            test_dataset=dataset,
            primary_split="test",
            max_render_size=1280,
            has_pointcloud=True,
            mesh_preview=True,
            mesh_url=None,
        )
        # Falls back to remote when mesh_url is None
        assert info["renderer"]["type"] == "remote"


# ─── _build_dataset_json tests ────────────────────────────────────────────


class TestDatasetJSON:
    def test_train_test_splits(self, dataset):
        result = _build_dataset_json({"train": dataset, "test": dataset})
        assert "train" in result
        assert "test" in result
        assert len(result["train"]["cameras"]) == 2
        assert len(result["test"]["cameras"]) == 2

    def test_none_split_omitted(self, dataset):
        result = _build_dataset_json({"train": None, "test": dataset})
        assert result["train"] is None
        assert result["test"] is not None

    def test_metadata_block(self, dataset):
        result = _build_dataset_json({"test": dataset})
        assert "metadata" in result
        assert "scene_center" in result["metadata"]
        assert "scene_radius" in result["metadata"]

    def test_pose_is_12_floats(self, dataset):
        result = _build_dataset_json({"test": dataset})
        pose = result["test"]["cameras"][0]["pose"]
        assert len(pose) == 12
        assert all(isinstance(v, float) for v in pose)

    def test_near_far_present(self, dataset):
        result = _build_dataset_json({"test": dataset})
        cam = result["test"]["cameras"][0]
        assert "near" in cam
        assert "far" in cam


# ─── _pose_matrix tests ───────────────────────────────────────────────────


class TestPoseMatrix:
    def test_3x4_array(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = _pose_matrix(arr)
        np.testing.assert_array_equal(result, arr)

    def test_flat_12(self):
        flat = list(range(12))
        result = _pose_matrix(flat)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.array(flat).reshape(3, 4))

    def test_flat_16(self):
        flat = list(range(16))
        result = _pose_matrix(flat)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.array(flat).reshape(4, 4)[:3, :4])

    def test_invalid_size_raises(self):
        with pytest.raises(ViewerBadRequest):
            _pose_matrix([1.0, 2.0, 3.0])


# ─── _intrinsics_matrix tests ─────────────────────────────────────────────


class TestIntrinsicsMatrix:
    def test_flat_4(self):
        K = _intrinsics_matrix([100, 200, 320, 240], width=640, height=480, dataset=None)
        assert K.shape == (3, 3)
        assert K[0, 0] == 100
        assert K[1, 1] == 200
        assert K[0, 2] == 320
        assert K[1, 2] == 240

    def test_3x3_matrix(self):
        K_in = np.eye(3, dtype=np.float32) * 500
        K_in[2, 2] = 1.0
        K = _intrinsics_matrix(K_in, width=640, height=480, dataset=None)
        np.testing.assert_array_equal(K, K_in)

    def test_flat_9(self):
        K = _intrinsics_matrix([100, 0, 320, 0, 200, 240, 0, 0, 1], width=640, height=480, dataset=None)
        assert K.shape == (3, 3)
        assert K[0, 0] == 100

    def test_none_with_dataset(self, dataset):
        K = _intrinsics_matrix(None, width=128, height=96, dataset=dataset)
        assert K.shape == (3, 3)
        # Should scale from 64x48 to 128x96
        assert K[0, 0] == pytest.approx(160.0)  # 80 * 2

    def test_none_without_dataset(self):
        K = _intrinsics_matrix(None, width=640, height=480, dataset=None)
        assert K.shape == (3, 3)
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0

    def test_invalid_raises(self):
        with pytest.raises(ViewerBadRequest):
            _intrinsics_matrix([1, 2, 3], width=640, height=480, dataset=None)


# ─── _image_size_from_request tests ───────────────────────────────────────


class TestImageSize:
    def test_default(self):
        w, h = _image_size_from_request(None, max_render_size=0)
        assert w == 768 and h == 512

    def test_explicit(self):
        w, h = _image_size_from_request([320, 240], max_render_size=0)
        assert w == 320 and h == 240

    def test_clamped(self):
        w, h = _image_size_from_request([2000, 1000], max_render_size=1280)
        assert max(w, h) <= 1280

    def test_no_clamp_when_zero(self):
        w, h = _image_size_from_request([2000, 1000], max_render_size=0)
        assert w == 2000 and h == 1000

    def test_invalid_raises(self):
        with pytest.raises(ViewerBadRequest):
            _image_size_from_request([1, 2, 3], max_render_size=0)


# ─── _camera_from_request tests ───────────────────────────────────────────


class TestCameraFromRequest:
    def test_with_pose(self, dataset):
        req = {
            "pose": torch.eye(4)[:3, :4].reshape(-1).tolist(),
            "intrinsics": [80, 80, 32, 24],
            "image_size": [64, 48],
        }
        cam = _camera_from_request(req, dataset=dataset, max_render_size=0)
        assert cam.width == 64
        assert cam.height == 48
        assert cam.batch_size == 1

    def test_singular_pose_raises(self, dataset):
        req = {
            "pose": [0.0] * 12,
            "intrinsics": [80, 80, 32, 24],
            "image_size": [64, 48],
        }
        with pytest.raises(ViewerBadRequest):
            _camera_from_request(req, dataset=dataset, max_render_size=0)

    def test_without_pose_uses_idx(self, dataset):
        req = {"idx": 1, "image_size": [32, 24]}
        cam = _camera_from_request(req, dataset=dataset, max_render_size=0)
        assert cam.width == 32
        assert cam.height == 24

    def test_without_pose_and_dataset_raises(self):
        with pytest.raises(ViewerBadRequest):
            _camera_from_request({}, dataset=None, max_render_size=0)

    def test_max_render_size_clamping(self, dataset):
        req = {
            "pose": torch.eye(4)[:3, :4].reshape(-1).tolist(),
            "intrinsics": [80, 80, 32, 24],
            "image_size": [2000, 1000],
        }
        cam = _camera_from_request(req, dataset=dataset, max_render_size=1280)
        assert max(cam.width, cam.height) <= 1280


# ─── _combine_outputs tests ───────────────────────────────────────────────


class TestCombineOutputs:
    def test_vertical_split(self):
        o1 = np.zeros((10, 10, 3), dtype=np.uint8)
        o2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = _combine_outputs(o1, o2, split_percentage=0.5, split_tilt=0)
        # Left half should be o1, right half should be o2
        assert result[5, 2, 0] == 0
        assert result[5, 7, 0] == 255

    def test_all_first(self):
        o1 = np.zeros((10, 10, 3), dtype=np.uint8)
        o2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = _combine_outputs(o1, o2, split_percentage=1.0, split_tilt=0)
        assert (result == 0).all()

    def test_all_second(self):
        o1 = np.zeros((10, 10, 3), dtype=np.uint8)
        o2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = _combine_outputs(o1, o2, split_percentage=0.0, split_tilt=0)
        assert (result == 255).all()

    def test_tilted(self):
        o1 = np.zeros((20, 20, 3), dtype=np.uint8)
        o2 = np.ones((20, 20, 3), dtype=np.uint8) * 255
        result = _combine_outputs(o1, o2, split_percentage=0.5, split_tilt=45)
        assert result.shape == (20, 20, 3)

    def test_shape_mismatch_raises(self):
        o1 = np.zeros((10, 10, 3), dtype=np.uint8)
        o2 = np.zeros((20, 20, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            _combine_outputs(o1, o2)


# ─── _to_hwc tests ────────────────────────────────────────────────────────


class TestToHWC:
    def test_2d(self):
        t = torch.rand(10, 10)
        result = _to_hwc(t)
        assert result.shape == (10, 10, 1)

    def test_3d_hwc(self):
        t = torch.rand(10, 10, 3)
        result = _to_hwc(t)
        assert result.shape == (10, 10, 3)

    def test_3d_chw(self):
        t = torch.rand(3, 10, 10)
        result = _to_hwc(t)
        assert result.shape == (10, 10, 3)

    def test_4d_nchw(self):
        t = torch.rand(1, 3, 10, 10)
        result = _to_hwc(t)
        assert result.shape == (10, 10, 3)


# ─── _format_output / _select_output tests ────────────────────────────────


class TestFormatOutput:
    def _make_output(self, h=10, w=10):
        return RenderOutput(
            rgb=torch.rand(h, w, 3),
            alpha=torch.rand(h, w),
            depth=torch.rand(h, w),
            normal=torch.rand(h, w, 3),
        )

    def test_color(self):
        out = self._make_output()
        arr = _format_output(out, "color")
        assert arr.shape == (10, 10, 3)
        assert arr.dtype == np.uint8

    def test_depth_colorized(self):
        out = self._make_output()
        arr = _format_output(out, "depth")
        assert arr.shape == (10, 10, 3)
        assert arr.dtype == np.uint8

    def test_alpha_colorized(self):
        out = self._make_output()
        arr = _format_output(out, "alpha")
        assert arr.shape == (10, 10, 3)

    def test_normal_remapped(self):
        out = self._make_output()
        arr = _format_output(out, "normal")
        assert arr.shape == (10, 10, 3)

    def test_missing_output_raises(self):
        out = RenderOutput(rgb=torch.rand(10, 10, 3))
        with pytest.raises(ViewerBadRequest):
            _select_output(out, "depth")

    def test_output_range_override(self):
        out = self._make_output()
        arr = _format_output(out, "depth", output_range=[0.0, 1.0])
        assert arr.shape == (10, 10, 3)


# ─── _normalize_output_type tests ─────────────────────────────────────────


class TestNormalizeOutputType:
    def test_rgb_to_color(self):
        assert _normalize_output_type("rgb") == "color"

    def test_opacity_to_alpha(self):
        assert _normalize_output_type("opacity") == "alpha"

    def test_none_defaults_color(self):
        assert _normalize_output_type(None) == "color"

    def test_passthrough(self):
        assert _normalize_output_type("depth") == "depth"


# ─── _scene_from_dataset tests ────────────────────────────────────────────


class TestSceneFromDataset:
    def test_normal(self, dataset):
        scene = _scene_from_dataset(dataset)
        assert len(scene["center"]) == 3
        assert scene["radius"] > 0
        assert len(scene["up"]) == 3

    def test_single_camera(self):
        ds = DummyViewerDataset()
        ds._samples = [ds._sample(0, [0.0, 0.0, 1.0])]
        scene = _scene_from_dataset(ds)
        assert scene["radius"] == 1.0  # Fallback for single camera


# ─── _robust_range tests ──────────────────────────────────────────────────


class TestRobustRange:
    def test_normal(self):
        values = np.random.randn(1000).astype(np.float32)
        finite = np.ones_like(values, dtype=bool)
        lo, hi = _robust_range(values, finite)
        assert lo < hi

    def test_all_same(self):
        values = np.full(100, 0.5, dtype=np.float32)
        finite = np.ones_like(values, dtype=bool)
        lo, hi = _robust_range(values, finite)
        assert hi > lo


# ─── _parse_image_path tests ──────────────────────────────────────────────


class TestParseImagePath:
    def test_valid(self):
        split, idx = _parse_image_path("/dataset/images/test/0.jpg")
        assert split == "test"
        assert idx == 0

    def test_nested_split(self):
        split, idx = _parse_image_path("/dataset/images/train/5.jpg")
        assert split == "train"
        assert idx == 5

    def test_invalid_idx(self):
        with pytest.raises(ViewerBadRequest):
            _parse_image_path("/dataset/images/test/abc.jpg")

    def test_missing_slash(self):
        with pytest.raises(ViewerBadRequest):
            _parse_image_path("/dataset/images/test.jpg")


# ─── _display_host tests ──────────────────────────────────────────────────


class TestDisplayHost:
    def test_ipv4(self):
        assert _display_host("127.0.0.1", 7007) == "127.0.0.1:7007"

    def test_ipv6(self):
        assert _display_host("::1", 7007) == "[::1]:7007"

    def test_wildcard(self):
        assert _display_host("0.0.0.0", 7007) == "localhost:7007"

    def test_empty(self):
        assert _display_host("", 7007) == "localhost:7007"


# ─── _resize_camera tests ─────────────────────────────────────────────────


class TestResizeCamera:
    def test_scale_intrinsics(self, dataset):
        cam = dataset.sample(0).camera
        resized = _resize_camera(cam, 128, 96)
        assert resized.width == 128
        assert resized.height == 96
        # 64→128 = 2x scale, so fx should be 80*2=160
        assert resized.Ks[0, 0, 0] == pytest.approx(160.0)

    def test_preserves_near_far(self, dataset):
        cam = dataset.sample(0).camera
        resized = _resize_camera(cam, 128, 96)
        assert resized.near == cam.near
        assert resized.far == cam.far


# ─── _matrix_to_list tests ────────────────────────────────────────────────


class TestMatrixToList:
    def test_3x4(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = _matrix_to_list(t)
        assert len(result) == 12
        assert all(isinstance(v, float) for v in result)


# ─── WebSocket message tests ──────────────────────────────────────────────


class TestWebSocketMessage:
    def _make_backend(self, adapter, dataset):
        from tribench.core.viewer import _ViewerBackend
        return _ViewerBackend(
            adapter,
            test_dataset=dataset,
            geometry_path=None,
        )

    def test_valid_render(self, adapter, dataset):
        backend = self._make_backend(adapter, dataset)
        camera = dataset.sample(0).camera
        req = json.dumps({
            "thread": 1,
            "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
            "intrinsics": [80, 80, 32, 24],
            "image_size": [32, 24],
            "output_type": "color",
            "lossless": True,
        })
        response = backend.handle_websocket_message(req)
        # Response should be binary: uint32 header_length + header + payload
        header_len = struct.unpack("!I", response[:4])[0]
        header = json.loads(response[4:4 + header_len])
        assert header["status"] == "ok"
        assert header["thread"] == 1
        assert header["mimetype"] == "image/png"
        # Payload follows header
        payload = response[4 + header_len:]
        assert len(payload) > 0

    def test_error_response(self, adapter, dataset):
        backend = self._make_backend(adapter, dataset)
        req = json.dumps({"thread": 2, "pose": [0.0] * 12})
        response = backend.handle_websocket_message(req)
        # Error responses are plain JSON (no binary framing)
        result = json.loads(response)
        assert result["status"] == "error"
        assert result["thread"] == 2

    def test_thread_passthrough(self, adapter, dataset):
        backend = self._make_backend(adapter, dataset)
        camera = dataset.sample(0).camera
        req = json.dumps({
            "thread": 42,
            "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
            "intrinsics": [80, 80, 32, 24],
            "image_size": [16, 12],
            "output_type": "color",
        })
        response = backend.handle_websocket_message(req)
        header_len = struct.unpack("!I", response[:4])[0]
        header = json.loads(response[4:4 + header_len])
        assert header["thread"] == 42

    def test_no_thread_field(self, adapter, dataset):
        backend = self._make_backend(adapter, dataset)
        camera = dataset.sample(0).camera
        req = json.dumps({
            "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
            "intrinsics": [80, 80, 32, 24],
            "image_size": [16, 12],
            "output_type": "color",
        })
        response = backend.handle_websocket_message(req)
        header_len = struct.unpack("!I", response[:4])[0]
        header = json.loads(response[4:4 + header_len])
        assert header["thread"] is None


# ─── Split compare integration tests ──────────────────────────────────────


class TestSplitCompare:
    def test_split_render(self, adapter, dataset):
        camera = dataset.sample(0).camera
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80, 80, 32, 24],
                "image_size": [64, 48],
                "output_type": "color",
                "split_output_type": "depth",
                "split_percentage": 0.5,
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.size == (64, 48)
            assert img.mode == "RGB"

    def test_split_with_tilt(self, adapter, dataset):
        camera = dataset.sample(0).camera
        payload, _ = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80, 80, 32, 24],
                "image_size": [64, 48],
                "output_type": "color",
                "split_output_type": "alpha",
                "split_percentage": 0.3,
                "split_tilt": 15.0,
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert len(payload) > 0


# ─── Different output types ───────────────────────────────────────────────


class TestOutputTypes:
    @pytest.mark.parametrize("output_type", ["color", "depth", "alpha", "normal"])
    def test_each_output_type(self, adapter, dataset, output_type):
        camera = dataset.sample(0).camera
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "pose": camera.camtoworlds[0][:3, :4].reshape(-1).tolist(),
                "intrinsics": [80, 80, 32, 24],
                "image_size": [32, 24],
                "output_type": output_type,
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.size == (32, 24)
            assert img.mode == "RGB"


# ─── Render with idx (no pose) ────────────────────────────────────────────


class TestRenderWithIdx:
    def test_idx_based_render(self, adapter, dataset):
        payload, mimetype = render_viewer_frame(
            adapter,
            {
                "idx": 1,
                "image_size": [32, 24],
                "output_type": "color",
                "lossless": True,
            },
            dataset=dataset,
            device="cpu",
        )
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(payload)) as img:
            assert img.size == (32, 24)
