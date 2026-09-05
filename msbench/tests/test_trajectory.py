"""Tests for trajectory loading and rendering (nerfbaselines-v1 format)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from msbench.core.cameras import CameraBatch
from msbench.core.trajectory import (
    load_trajectory,
    save_trajectory,
    trajectory_cameras,
    trajectory_get_fps,
    trajectory_get_image_size,
)
from msbench.renderers.base import RenderOutput, RendererAdapter


# ─── Test fixtures ────────────────────────────────────────────────────────


class DummyTrajAdapter(RendererAdapter):
    @property
    def name(self) -> str:
        return "triangle-splatting"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def load_checkpoint(self, path: str) -> None:
        pass

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        pass

    def model_stats(self) -> dict:
        return {"method": self.name}

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        h, w = cameras.height, cameras.width
        rgb = torch.zeros(h, w, 3)
        rgb[..., 0] = 0.5  # Flat red-ish image
        return RenderOutput(rgb=rgb, alpha=torch.ones(h, w))


def _make_trajectory_json(num_frames=5, w=64, h=48, fps=30):
    """Create a minimal nerfbaselines-v1 trajectory JSON."""
    frames = []
    for i in range(num_frames):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(i) * 0.1  # Move camera along x
        frames.append({
            "pose": c2w.flatten().tolist(),
            "intrinsics": [80.0, 80.0, w / 2, h / 2],
        })
    return {
        "format": "nerfbaselines-v1",
        "camera_model": "pinhole",
        "image_size": [w, h],
        "fps": fps,
        "frames": frames,
    }


# ─── load_trajectory tests ────────────────────────────────────────────────


class TestLoadTrajectory:
    def test_valid(self, tmp_path):
        traj_data = _make_trajectory_json()
        path = tmp_path / "traj.json"
        path.write_text(json.dumps(traj_data))
        traj = load_trajectory(path)
        assert traj["format"] == "nerfbaselines-v1"
        assert len(traj["frames"]) == 5

    def test_invalid_format(self, tmp_path):
        traj_data = {"format": "unknown", "frames": [{"pose": [0] * 16}]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(traj_data))
        with pytest.raises(ValueError, match="Unsupported"):
            load_trajectory(path)

    def test_no_frames(self, tmp_path):
        traj_data = {"format": "nerfbaselines-v1", "frames": []}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(traj_data))
        with pytest.raises(ValueError, match="at least one"):
            load_trajectory(path)

    def test_frame_missing_pose(self, tmp_path):
        traj_data = {
            "format": "nerfbaselines-v1",
            "frames": [{"intrinsics": [80, 80, 32, 24]}],
        }
        path = tmp_path / "bad2.json"
        path.write_text(json.dumps(traj_data))
        with pytest.raises(ValueError, match="pose"):
            load_trajectory(path)


# ─── trajectory_cameras tests ─────────────────────────────────────────────


class TestTrajectoryCameras:
    def test_camera_count(self, tmp_path):
        traj_data = _make_trajectory_json(num_frames=10)
        traj = json.loads(json.dumps(traj_data))  # deep copy
        cameras = trajectory_cameras(traj)
        assert len(cameras) == 10

    def test_camera_dimensions(self, tmp_path):
        traj_data = _make_trajectory_json(w=128, h=96)
        cameras = trajectory_cameras(traj_data)
        assert cameras[0].width == 128
        assert cameras[0].height == 96

    def test_pose_4x4(self):
        traj = _make_trajectory_json(num_frames=3)
        cameras = trajectory_cameras(traj)
        # Camera 0 should be at identity
        c2w = cameras[0].camtoworlds[0].numpy()
        np.testing.assert_array_almost_equal(c2w, np.eye(4))

    def test_pose_3x4(self):
        traj = _make_trajectory_json(num_frames=3)
        # Convert to 3x4 poses (12 floats)
        for frame in traj["frames"]:
            frame["pose"] = frame["pose"][:12]
        cameras = trajectory_cameras(traj)
        assert len(cameras) == 3

    def test_intrinsics_flat4(self):
        traj = _make_trajectory_json(num_frames=2)
        cameras = trajectory_cameras(traj)
        K = cameras[0].Ks[0].numpy()
        assert K[0, 0] == 80.0
        assert K[1, 1] == 80.0
        assert K[0, 2] == 32.0
        assert K[1, 2] == 24.0

    def test_intrinsics_3x3(self):
        traj = _make_trajectory_json(num_frames=2)
        K_flat = [100, 0, 32, 0, 100, 24, 0, 0, 1]
        for frame in traj["frames"]:
            frame["intrinsics"] = K_flat
        cameras = trajectory_cameras(traj)
        K = cameras[0].Ks[0].numpy()
        assert K[0, 0] == 100.0

    def test_device_transfer(self):
        traj = _make_trajectory_json(num_frames=2)
        cameras = trajectory_cameras(traj, device="cpu")
        assert cameras[0].Ks.device.type == "cpu"


# ─── trajectory_get_fps / image_size tests ────────────────────────────────


class TestTrajectoryMetadata:
    def test_fps(self):
        traj = _make_trajectory_json(fps=60)
        assert trajectory_get_fps(traj) == 60

    def test_fps_default(self):
        traj = {"format": "nerfbaselines-v1", "frames": [{"pose": [0] * 16}]}
        assert trajectory_get_fps(traj) == 30

    def test_image_size(self):
        traj = _make_trajectory_json(w=1920, h=1080)
        w, h = trajectory_get_image_size(traj)
        assert w == 1920
        assert h == 1080

    def test_image_size_default(self):
        traj = {"format": "nerfbaselines-v1", "frames": [{"pose": [0] * 16}]}
        w, h = trajectory_get_image_size(traj)
        assert w == 768
        assert h == 512


# ─── save_trajectory tests ────────────────────────────────────────────────


class TestSaveTrajectory:
    def _make_cameras(self, n=3, w=64, h=48):
        cameras = []
        for i in range(n):
            c2w = np.eye(4, dtype=np.float32)
            c2w[0, 3] = float(i) * 0.5
            w2c = np.linalg.inv(c2w).astype(np.float32)
            K = np.array([[80, 0, 32], [0, 80, 24], [0, 0, 1]], dtype=np.float32)
            cameras.append(CameraBatch(
                viewmats=torch.from_numpy(w2c).unsqueeze(0),
                camtoworlds=torch.from_numpy(c2w).unsqueeze(0),
                Ks=torch.from_numpy(K).unsqueeze(0),
                width=w,
                height=h,
            ))
        return cameras

    def test_roundtrip(self, tmp_path):
        cameras = self._make_cameras(5)
        path = tmp_path / "traj.json"
        save_trajectory(path, cameras, fps=30, image_size=(64, 48))
        traj = load_trajectory(path)
        assert traj["fps"] == 30
        assert len(traj["frames"]) == 5
        assert traj["image_size"] == [64, 48]

    def test_frame_poses_4x4(self, tmp_path):
        cameras = self._make_cameras(3)
        path = tmp_path / "traj.json"
        save_trajectory(path, cameras)
        traj = load_trajectory(path)
        for frame in traj["frames"]:
            assert len(frame["pose"]) == 16

    def test_intrinsics_flat4(self, tmp_path):
        cameras = self._make_cameras(2)
        path = tmp_path / "traj.json"
        save_trajectory(path, cameras)
        traj = load_trajectory(path)
        for frame in traj["frames"]:
            assert len(frame["intrinsics"]) == 4

    def test_with_keyframes(self, tmp_path):
        cameras = self._make_cameras(3)
        keyframes = [{"pose": list(np.eye(4).flatten()), "fov": 60}]
        path = tmp_path / "traj.json"
        save_trajectory(path, cameras, keyframes=keyframes, interpolation="linear")
        traj = load_trajectory(path)
        assert "source" in traj
        assert traj["source"]["interpolation"] == "linear"


# ─── render_trajectory_frames tests ───────────────────────────────────────


class TestRenderTrajectoryFrames:
    def test_basic_render(self, tmp_path):
        from msbench.core.trajectory import render_trajectory_frames
        adapter = DummyTrajAdapter()
        traj = _make_trajectory_json(num_frames=3, w=32, h=24)
        cameras = trajectory_cameras(traj)
        manifest = render_trajectory_frames(
            adapter, cameras, tmp_path / "out", fps=30, output_types=("color",),
        )
        assert manifest["num_frames"] == 3
        assert manifest["fps"] == 30
        # Color frames should be saved
        from pathlib import Path
        color_dir = Path(manifest["frame_dirs"]["color"])
        assert color_dir.exists()
        assert len(list(color_dir.glob("*.jpg"))) == 3

    def test_video_output(self, tmp_path):
        from msbench.core.trajectory import render_trajectory_frames
        adapter = DummyTrajAdapter()
        traj = _make_trajectory_json(num_frames=5, w=32, h=24)
        cameras = trajectory_cameras(traj)
        manifest = render_trajectory_frames(
            adapter, cameras, tmp_path / "out", fps=30, output_types=("color",),
        )
        if manifest.get("video_path"):
            from pathlib import Path
            assert Path(manifest["video_path"]).exists()
