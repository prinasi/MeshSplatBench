"""Trajectory loading and rendering for nerfbaselines-v1 camera paths.

Supports the ``nerfbaselines-v1`` trajectory JSON format exported by the
nerfbaselines viewer's ``save_trajectory()`` function. Each trajectory
contains a list of frames with 4x4 pose matrices and [fx, fy, cx, cy]
intrinsics, plus metadata like fps and image_size.

Usage::

    from msbench.core.trajectory import load_trajectory, trajectory_cameras

    traj = load_trajectory("path.json")
    cameras = trajectory_cameras(traj)          # list[CameraBatch]
    render_trajectory_frames(adapter, cameras, "output/", fps=traj["fps"])
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from msbench.core.cameras import CameraBatch
from msbench.renderers.base import RenderOutput, RendererAdapter


# ─── Trajectory JSON I/O ──────────────────────────────────────────────────


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a nerfbaselines-v1 trajectory JSON file.

    Expected structure::

        {
          "format": "nerfbaselines-v1",
          "camera_model": "pinhole",
          "image_size": [w, h],
          "fps": 30,
          "frames": [
            {"pose": [16 floats, 4x4 row-major], "intrinsics": [fx, fy, cx, cy], ...},
            ...
          ]
        }
    """
    with open(path) as f:
        traj = json.load(f)
    _validate_trajectory(traj)
    return traj


def _validate_trajectory(traj: dict[str, Any]) -> None:
    fmt = traj.get("format", "")
    if "nerfbaselines" not in fmt:
        raise ValueError(
            f"Unsupported trajectory format: {fmt!r}. Expected 'nerfbaselines-v1'."
        )
    frames = traj.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Trajectory must contain at least one frame")
    for i, frame in enumerate(frames):
        if "pose" not in frame:
            raise ValueError(f"Frame {i} missing 'pose' field")
        pose = frame["pose"]
        if not isinstance(pose, list) or len(pose) not in (12, 16):
            raise ValueError(
                f"Frame {i} pose must be 12 or 16 floats, got {len(pose)}"
            )


def save_trajectory(
    path: str | Path,
    cameras: list[CameraBatch],
    *,
    fps: int = 30,
    image_size: tuple[int, int] | None = None,
    keyframes: list[dict[str, Any]] | None = None,
    interpolation: str = "none",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trajectory JSON in nerfbaselines-v1 format.

    Args:
        cameras: List of CameraBatch, one per frame.
        fps: Target framerate.
        image_size: (width, height) override; defaults to camera dimensions.
        keyframes: Optional list of keyframe dicts for the ``source`` field.
        interpolation: Interpolation mode used to generate the frames.
        metadata: Extra metadata to merge into the trajectory.
    """
    w, h = image_size or (cameras[0].width, cameras[0].height)
    frames = []
    for cam in cameras:
        c2w = cam.camtoworlds[0].detach().cpu().numpy()
        K = cam.Ks[0].detach().cpu().numpy()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        frames.append({
            "pose": c2w.flatten().tolist(),
            "intrinsics": [fx, fy, cx, cy],
        })

    traj: dict[str, Any] = {
        "format": "nerfbaselines-v1",
        "camera_model": "pinhole",
        "image_size": [w, h],
        "fps": fps,
        "frames": frames,
    }

    if keyframes is not None:
        traj["source"] = {
            "type": "interpolation",
            "interpolation": interpolation,
            "keyframes": keyframes,
        }

    if metadata:
        traj.update(metadata)

    with open(path, "w") as f:
        json.dump(traj, f, indent=2)


# ─── Trajectory → Cameras ─────────────────────────────────────────────────


def trajectory_cameras(
    traj: dict[str, Any],
    *,
    device: str | torch.device | None = None,
) -> list[CameraBatch]:
    """Build a list of CameraBatch from a trajectory JSON.

    Each frame becomes one CameraBatch with a single camera.
    """
    image_size = traj.get("image_size")
    default_w, default_h = (
        (int(image_size[0]), int(image_size[1])) if image_size else (768, 512)
    )

    cameras: list[CameraBatch] = []
    for frame in traj["frames"]:
        w, h = default_w, default_h
        if "image_size" in frame:
            w, h = int(frame["image_size"][0]), int(frame["image_size"][1])

        # Pose: 4x4 or 3x4 row-major → 4x4 cam-to-world
        pose_flat = frame["pose"]
        if len(pose_flat) == 16:
            c2w = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        elif len(pose_flat) == 12:
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = np.array(pose_flat, dtype=np.float32).reshape(3, 4)
        else:
            raise ValueError(f"Invalid pose length: {len(pose_flat)}")

        w2c = np.linalg.inv(c2w).astype(np.float32)

        # Intrinsics: [fx, fy, cx, cy] or 3x3 matrix
        intrinsics = frame.get("intrinsics", [0.5 * w / math.tan(math.radians(30)), 0.5 * w / math.tan(math.radians(30)), w / 2, h / 2])
        if len(intrinsics) == 4:
            fx, fy, cx, cy = [float(x) for x in intrinsics]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        elif len(intrinsics) == 9:
            K = np.array(intrinsics, dtype=np.float32).reshape(3, 3)
        else:
            focal = 0.5 * w / math.tan(math.radians(30))
            K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)

        cam = CameraBatch(
            viewmats=torch.from_numpy(w2c).unsqueeze(0),
            camtoworlds=torch.from_numpy(c2w).unsqueeze(0),
            Ks=torch.from_numpy(K).unsqueeze(0),
            width=w,
            height=h,
            near=float(frame.get("near", 0.01)),
            far=float(frame.get("far", 100.0)),
        )
        if device is not None:
            cam = cam.to(device)
        cameras.append(cam)

    return cameras


def trajectory_get_fps(traj: dict[str, Any]) -> int:
    """Return the trajectory's framerate."""
    return int(traj.get("fps", 30))


def trajectory_get_image_size(traj: dict[str, Any]) -> tuple[int, int]:
    """Return (width, height) from trajectory."""
    sz = traj.get("image_size", [768, 512])
    return int(sz[0]), int(sz[1])


# ─── Trajectory rendering ─────────────────────────────────────────────────


def render_trajectory_frames(
    adapter: RendererAdapter,
    cameras: list[CameraBatch],
    output_dir: str | Path,
    *,
    fps: int = 30,
    output_types: tuple[str, ...] = ("color",),
    device: str | torch.device | None = None,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Render all frames in a trajectory and save to output_dir.

    For each output_type, saves:
      - color: <output_dir>/color/frame_NNNNN.jpg
      - depth/alpha/normal: <output_dir>/<type>/frame_NNNNN.png
      - Video: <output_dir>/video.mp4 (if color is in output_types)

    Returns a manifest dict with frame count, paths, and metadata.
    """
    import io
    import imageio

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    render_device = torch.device(device) if device is not None else adapter.device
    frame_lists: dict[str, list[np.ndarray]] = {ot: [] for ot in output_types}

    for idx, cam in enumerate(cameras):
        cam = cam.to(render_device)
        mode = "eval_aux" if (
            str(getattr(adapter, "name", "")).replace("_", "-").lower() == "diffsoup"
            and any(ot in {"depth", "alpha"} for ot in output_types)
        ) else "eval"

        with torch.no_grad():
            output = adapter.render(cam, mode=mode)

        for ot in output_types:
            arr = _format_trajectory_output(output, ot)
            frame_lists[ot].append(arr)

            # Save individual frames
            subdir = out / ot
            subdir.mkdir(parents=True, exist_ok=True)
            ext = "jpg" if ot == "color" else "png"
            img = Image.fromarray(arr)
            with io.BytesIO() as buf:
                if ext == "jpg":
                    img.save(buf, format="JPEG", quality=jpeg_quality)
                else:
                    img.save(buf, format="PNG")
                (subdir / f"frame_{idx:05d}.{ext}").write_bytes(buf.getvalue())

    # Write video if color frames exist
    video_path = None
    if "color" in frame_lists and frame_lists["color"]:
        video_path = str(out / "video.mp4")
        imageio.mimsave(video_path, frame_lists["color"], fps=fps, codec="libx264")

    return {
        "num_frames": len(cameras),
        "fps": fps,
        "output_types": list(output_types),
        "video_path": video_path,
        "frame_dirs": {ot: str(out / ot) for ot in output_types},
    }


def _format_trajectory_output(output: RenderOutput, output_type: str) -> np.ndarray:
    """Format a RenderOutput tensor to uint8 [H, W, 3] numpy array."""
    from msbench.core.viewer import _to_hwc, _format_output

    return _format_output(output, output_type)
