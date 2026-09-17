"""CPU-only tests for Unity video trajectory rendering functionality."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from msbench.core.cameras import CameraBatch
from msbench.core.rendering import generate_ellipse_cameras
from msbench.core.trajectory import load_trajectory, save_trajectory
from tools.run_unity_triasset_video import (
    encode_video_from_frames,
    find_unity_executable,
    find_unity_project,
    generate_trajectory_from_dataset,
    validate_captured_frames,
)


def _make_dummy_camera_batch(n: int = 5, w: int = 64, h: int = 48) -> CameraBatch:
    eye = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
    eye[:, 0, 3] = torch.linspace(-1.0, 1.0, n)
    eye[:, 1, 3] = torch.linspace(-0.5, 0.5, n)
    eye[:, 2, 3] = torch.linspace(2.0, 3.0, n)

    w2c = torch.inverse(eye)
    Ks = torch.zeros((n, 3, 3), dtype=torch.float32)
    Ks[:, 0, 0] = 50.0  # fx
    Ks[:, 1, 1] = 50.0  # fy
    Ks[:, 0, 2] = w / 2.0  # cx
    Ks[:, 1, 2] = h / 2.0  # cy
    Ks[:, 2, 2] = 1.0

    return CameraBatch(
        viewmats=w2c,
        camtoworlds=eye,
        Ks=Ks,
        width=w,
        height=h,
        near=0.01,
        far=100.0,
    )


class UnityVideoTrajectoryTests(unittest.TestCase):
    def test_trajectory_matches_ellipse_cameras(self) -> None:
        base_cams = _make_dummy_camera_batch(5, 64, 48)
        ellipse_cams = generate_ellipse_cameras(base_cams, n_frames=10, zoom=1.2)
        self.assertEqual(len(ellipse_cams), 10)

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "trajectory.json"
            save_trajectory(traj_path, ellipse_cams, fps=24)
            self.assertTrue(traj_path.is_file())

            traj_data = load_trajectory(traj_path)
            self.assertEqual(traj_data["format"], "nerfbaselines-v1")
            self.assertEqual(traj_data["fps"], 24)
            self.assertEqual(len(traj_data["frames"]), 10)
            self.assertEqual(traj_data["image_size"], [64, 48])

            first_frame = traj_data["frames"][0]
            self.assertIn("pose", first_frame)
            self.assertEqual(len(first_frame["pose"]), 16)
            self.assertIn("intrinsics", first_frame)
            self.assertEqual(len(first_frame["intrinsics"]), 4)

            # Check focal length was scaled by zoom
            expected_fx = 50.0 * 1.2
            self.assertAlmostEqual(first_frame["intrinsics"][0], expected_fx, places=3)
            self.assertAlmostEqual(first_frame["intrinsics"][1], expected_fx, places=3)

    def test_pose_extraction_math_matches_colmap_convention(self) -> None:
        """Verify row-major c2w extraction yields the exact same position, down, forward vectors."""
        # Construct a synthetic c2w matrix with known translation and rotation
        # COLMAP right = +X, down = +Y, forward = +Z
        c2w = np.array([
            [1.0, 0.0, 0.0, 2.5],
            [0.0, 0.0, -1.0, 3.5],
            [0.0, 1.0, 0.0, 4.5],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        pose_flat = c2w.flatten().tolist()

        # Extract as Unity TriAssetVideoBatchCapture does:
        # Row 0: right.x, down.x, forward.x, pos.x
        # Row 1: right.y, down.y, forward.y, pos.y
        # Row 2: right.z, down.z, forward.z, pos.z
        pos = (pose_flat[3], pose_flat[7], pose_flat[11])
        right = (pose_flat[0], pose_flat[4], pose_flat[8])
        down = (pose_flat[1], pose_flat[5], pose_flat[9])
        forward = (pose_flat[2], pose_flat[6], pose_flat[10])

        self.assertEqual(pos, (2.5, 3.5, 4.5))
        self.assertEqual(right, (1.0, 0.0, 0.0))
        self.assertEqual(down, (0.0, 0.0, 1.0))
        self.assertEqual(forward, (0.0, -1.0, 0.0))

    def test_encode_video_from_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            for i in range(5):
                arr = np.full((32, 32, 3), (i * 40) % 255, dtype=np.uint8)
                Image.fromarray(arr).save(frames_dir / f"{i:05d}.png")

            video_path = Path(tmp) / "output.mp4"
            out = encode_video_from_frames(frames_dir, video_path, fps=10)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_validate_captured_frames_rejects_solid_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            for i in range(3):
                arr = np.zeros((16, 16, 3), dtype=np.uint8)  # Solid black
                Image.fromarray(arr).save(frames_dir / f"{i:05d}.png")

            with self.assertRaisesRegex(RuntimeError, "solid-color"):
                validate_captured_frames(frames_dir, 3)

    def test_validate_captured_frames_accepts_valid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            for i in range(3):
                arr = np.zeros((16, 16, 3), dtype=np.uint8)
                arr[i * 4, i * 4] = [255, 128, 64]  # Non-solid
                Image.fromarray(arr).save(frames_dir / f"{i:05d}.png")

            frames = validate_captured_frames(frames_dir, 3)
            self.assertEqual(len(frames), 3)

    def test_find_unity_executable_and_project(self) -> None:
        proj = find_unity_project()
        self.assertIsNotNone(proj)
        self.assertTrue((proj / "Assets").is_dir())
        self.assertTrue((proj / "ProjectSettings").is_dir())


if __name__ == "__main__":
    unittest.main()
