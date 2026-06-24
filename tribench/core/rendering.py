"""Rendering utilities for split renders, trajectory videos, and viewers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from tribench.core.cameras import CameraBatch
from tribench.core.datasets import DatasetBase, DatasetSample
from tribench.renderers.base import RenderOutput, RendererAdapter


@dataclass
class RenderRecord:
    """One rendered frame record."""

    index: int
    name: str
    render_path: Path
    gt_path: Path | None = None
    depth_path: Path | None = None
    alpha_path: Path | None = None
    metrics: dict[str, float] | None = None


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert an RGB/HW tensor in [0, 1] to uint8 numpy."""
    image = image.detach().float().cpu().clamp(0, 1)
    if image.dim() == 2:
        image = image.unsqueeze(-1).repeat(1, 1, 3)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    return (image.numpy() * 255.0 + 0.5).astype(np.uint8)


def save_image(path: str | Path, image: torch.Tensor) -> Path:
    """Save an image tensor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(image)).save(path)
    return path


def save_depth(path: str | Path, depth: torch.Tensor) -> Path:
    """Save a depth tensor as a normalized grayscale PNG."""
    depth = depth.detach().float().cpu()
    finite = torch.isfinite(depth)
    if finite.any():
        lo = depth[finite].min()
        hi = depth[finite].max()
        depth = (depth - lo) / (hi - lo).clamp(min=1e-8)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    return save_image(path, depth)


def load_adapter(method: str, checkpoint: str) -> RendererAdapter:
    """Construct and load a renderer adapter."""
    from tribench.core.builder import build_adapter

    return build_adapter({"type": method, "checkpoint": checkpoint})


def load_adapter_from_cfg(config: dict[str, Any]) -> RendererAdapter:
    """Construct and optionally load a renderer adapter from config."""
    from tribench.core.builder import build_adapter

    return build_adapter(config)


def render_sample(
    adapter: RendererAdapter,
    sample: DatasetSample,
    *,
    device: str | torch.device = "cuda",
) -> RenderOutput:
    camera = sample.camera.to(device)
    with torch.no_grad():
        output = adapter.render(camera, mode="eval")
    return output


def _background_color_from_output(output: RenderOutput) -> Any:
    source_bg = getattr(output, "background_color", None)
    if source_bg is None:
        source_bg = output.extras.get("background_color") if output.extras else None
    if source_bg is None:
        source_bg = [0.0, 0.0, 0.0]
    return source_bg


def _composite_output_rgb(output: RenderOutput, bg_color: Any) -> torch.Tensor:
    pred = output.rgb.clamp(0, 1)
    if bg_color is not None and output.alpha is not None:
        bg = torch.tensor(bg_color, dtype=pred.dtype, device=pred.device).view(1, 1, 3)
        alpha = output.alpha.to(pred.device, dtype=pred.dtype).clamp(0, 1).unsqueeze(-1)
        src = torch.tensor(
            _background_color_from_output(output),
            dtype=pred.dtype,
            device=pred.device,
        ).view(1, 1, 3)
        color = pred - src * (1.0 - alpha)
        pred = (color + bg * (1.0 - alpha)).clamp(0, 1)
    return pred


def _metric_rgb_for_sample(
    output: RenderOutput,
    sample: DatasetSample,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    target = sample.image.to(output.rgb.device).clamp(0, 1) if sample.image is not None else None
    metadata = sample.metadata or getattr(sample.camera, "metadata", None) or {}
    bg_color = metadata.get("eval_background_color")
    pred = _composite_output_rgb(output, bg_color)
    return pred, target


def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render_dataset_split(
    adapter: RendererAdapter,
    dataset: DatasetBase,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    save_gt: bool = False,
    save_aux: bool = False,
    metrics: bool = False,
) -> dict[str, Any]:
    """Render every camera in a dataset split and optionally compute metrics."""
    from tribench.core.metrics import compute_all_metrics

    output_dir = Path(output_dir)
    render_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    depth_dir = output_dir / "depth"
    alpha_dir = output_dir / "alpha"

    warmup_time_s = 0.0
    warmup_frames = 0
    if len(dataset) > 0 and torch.cuda.is_available():
        sample = dataset.sample(0)
        camera = sample.camera.to(device)
        _cuda_sync_if_available()
        warmup_start = time.perf_counter()
        with torch.no_grad():
            adapter.render(camera, mode="eval")
        _cuda_sync_if_available()
        warmup_time_s = time.perf_counter() - warmup_start
        warmup_frames = 1

    records: list[RenderRecord] = []
    per_view: list[dict[str, float]] = []
    inference_time_s = 0.0
    frame_times_s: list[float] = []
    wall_start = time.perf_counter()
    for idx in range(len(dataset)):
        sample = dataset.sample(idx)
        camera = sample.camera.to(device)
        _cuda_sync_if_available()
        render_start = time.perf_counter()
        with torch.no_grad():
            output = adapter.render(camera, mode="eval")
        _cuda_sync_if_available()
        frame_time_s = time.perf_counter() - render_start
        inference_time_s += frame_time_s
        frame_times_s.append(frame_time_s)

        stem = f"{idx:05d}_{sample.name}"
        render_rgb, _ = _metric_rgb_for_sample(output, sample)
        render_path = save_image(render_dir / f"{stem}.png", render_rgb)

        gt_path = None
        if save_gt and sample.image is not None:
            gt_path = save_image(gt_dir / f"{stem}.png", sample.image)

        depth_path = None
        if save_aux and output.depth is not None:
            depth_path = save_depth(depth_dir / f"{stem}.png", output.depth)

        alpha_path = None
        if save_aux and output.alpha is not None:
            alpha_path = save_image(alpha_dir / f"{stem}.png", output.alpha)

        view_metrics = None
        if metrics and sample.image is not None:
            pred, target = render_rgb, sample.image.to(render_rgb.device).clamp(0, 1)
            assert target is not None
            view_metrics = compute_all_metrics(pred, target)
            per_view.append({"name": sample.name, **view_metrics})

        records.append(
            RenderRecord(
                index=idx,
                name=sample.name,
                render_path=render_path,
                gt_path=gt_path,
                depth_path=depth_path,
                alpha_path=alpha_path,
                metrics=view_metrics,
            )
        )

    aggregate: dict[str, float] = {}
    if per_view:
        metric_keys = [k for k in per_view[0] if k != "name"]
        for key in metric_keys:
            vals = [float(row[key]) for row in per_view]
            aggregate[f"{key}_mean"] = float(np.mean(vals))
            aggregate[f"{key}_std"] = float(np.std(vals))

    wall_time_s = time.perf_counter() - wall_start
    num_frames = len(records)
    timing = {
        "num_frames": num_frames,
        "time_s": inference_time_s,
        "fps": num_frames / inference_time_s if inference_time_s > 0 else 0.0,
        "mean_frame_time_ms": (inference_time_s / num_frames * 1000.0) if num_frames > 0 else 0.0,
        "median_frame_time_ms": float(np.median(frame_times_s) * 1000.0) if frame_times_s else 0.0,
        "warmup_frames": warmup_frames,
        "warmup_time_s": warmup_time_s,
        "wall_time_s": wall_time_s,
        "wall_fps": num_frames / wall_time_s if wall_time_s > 0 else 0.0,
    }
    manifest = {
        "num_frames": num_frames,
        "timing": timing,
        "aggregate": aggregate,
        "frames": [
            {
                "index": r.index,
                "name": r.name,
                "render_path": str(r.render_path),
                "gt_path": str(r.gt_path) if r.gt_path else None,
                "depth_path": str(r.depth_path) if r.depth_path else None,
                "alpha_path": str(r.alpha_path) if r.alpha_path else None,
                "metrics": r.metrics,
            }
            for r in records
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if per_view:
        (output_dir / "metrics.json").write_text(
            json.dumps({"per_view": per_view, "aggregate": aggregate, "inference": timing}, indent=2)
        )
    return manifest


# ---------------------------------------------------------------------------
# Trajectory helpers (ported from triangle-splatting/utils/render_utils.py)
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)


def _pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] poses with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def _unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def _viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct a look-at view matrix (c2w convention)."""
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Nearest point to all focal axes (z-axes) in *poses*."""
    directions = poses[:, :3, 2:3]
    origins = poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.pinv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def _transform_poses_pca(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recenter and rotate *poses* so principal components lie on XYZ axes.

    Args:
        poses: ``(N, 3, 4)`` camera-to-world matrices.

    Returns:
        ``(poses_recentered, transform)`` where *transform* is the ``4x4``
        matrix that was applied (so it can be inverted later).
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = _unpad_poses(transform @ _pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of average y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def _generate_ellipse_path(
    poses: np.ndarray,
    *,
    n_frames: int = 120,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical camera path in the (PCA-aligned) pose space.

    Args:
        poses: ``(N, 3, 4)`` camera-to-world matrices in PCA space.
        n_frames: Number of output frames.
        z_variation: Amplitude of vertical oscillation (0 = flat).
        z_phase: Phase offset for vertical oscillation, in ``[0, 1]``.

    Returns:
        ``(n_frames, 3, 4)`` array of c2w matrices along the ellipse.
    """
    center = _focus_point_fn(poses)
    # Ellipse sits at z=0 in PCA space (middle of the capture pattern).
    offset = np.array([center[0], center[1], 0.0])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    sc = np.maximum(sc, 1e-3)
    low = -sc + offset
    high = sc + offset

    z_low = np.percentile(poses[:, :3, 3], 10, axis=0)
    z_high = np.percentile(poses[:, :3, 3], 90, axis=0)

    def _positions(theta):
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
            low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
            z_variation * (
                z_low[2]
                + (z_high - z_low)[2]
                * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
            ),
        ], axis=-1)

    theta = np.linspace(0, 2 * np.pi, n_frames + 1, endpoint=True)
    positions = _positions(theta)[:-1]  # drop duplicate last frame

    # Snap up vector to the closest cardinal axis of the average camera up.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / (np.linalg.norm(avg_up) + 1e-8)
    ind_up = int(np.argmax(np.abs(avg_up)))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([
        _viewmatrix(p - center, up, p) for p in positions
    ])


def generate_ellipse_cameras(
    base_cameras: CameraBatch,
    *,
    n_frames: int = 240,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    zoom: float = 1.0,
) -> list[CameraBatch]:
    """Generate an ellipse camera path with PCA-aligned trajectory.

    The algorithm mirrors the upstream triangle-splatting ``generate_path()``:

    1. Extract c2w matrices from *base_cameras*.
    2. Flip Y/Z axes (COLMAP → OpenGL convention).
    3. PCA-align camera positions so principal components lie on XYZ.
    4. Fit an ellipse in PCA space, always looking at the scene's focus
       point, with the up vector snapped to a clean cardinal axis.
    5. Inverse-transform the generated poses back to world space.
    6. Build ``CameraBatch`` frames using the **original** intrinsics
       (from the first camera), optionally scaled by *zoom*.

    Args:
        base_cameras: Training (or test) cameras whose positions define
            the ellipse geometry.
        n_frames: Total number of frames in the trajectory.
        z_variation: Vertical oscillation amplitude (0 = flat orbit).
        z_phase: Phase of vertical oscillation in ``[0, 1]``.
        zoom: Focal-length multiplier applied to the first camera's
            intrinsics (>1 zooms in, <1 zooms out).

    Returns:
        A list of single-camera ``CameraBatch`` instances.
    """
    base_metadata = dict(base_cameras.metadata or {})

    # -- 1. Extract c2w -----------------------------------------------
    c2ws = base_cameras.camtoworlds.detach().cpu().numpy()  # [N, 4, 4]

    # -- 2. COLMAP → OpenGL: flip Y and Z axes -----------------------
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    poses_gl = c2ws[:, :3, :] @ flip  # [N, 3, 4]

    # -- 3. PCA alignment ---------------------------------------------
    poses_pca, colmap_to_world = _transform_poses_pca(poses_gl)

    # -- 4. Ellipse in PCA space --------------------------------------
    new_poses = _generate_ellipse_path(
        poses_pca, n_frames=n_frames, z_variation=z_variation, z_phase=z_phase,
    )  # [n_frames, 3, 4]

    # -- 5. Inverse transform back to world (OpenGL convention) ------
    new_poses = np.linalg.inv(colmap_to_world) @ _pad_poses(new_poses)

    # -- 6. Flip back to COLMAP convention and build CameraBatch -----
    K0 = base_cameras.Ks[0].detach().cpu().numpy().copy()
    if zoom != 1.0:
        K0[0, 0] *= zoom
        K0[1, 1] *= zoom

    frames: list[CameraBatch] = []
    for i, c2w_gl in enumerate(new_poses):
        c2w = c2w_gl @ flip  # back to COLMAP
        w2c = np.linalg.inv(c2w).astype(np.float32)
        c2w = c2w.astype(np.float32)
        metadata = {**base_metadata, "trajectory": "ellipse_pca", "frame": i}
        frames.append(CameraBatch(
            viewmats=torch.from_numpy(w2c).unsqueeze(0),
            camtoworlds=torch.from_numpy(c2w).unsqueeze(0),
            Ks=torch.from_numpy(K0.astype(np.float32)).unsqueeze(0),
            width=base_cameras.width,
            height=base_cameras.height,
            near=base_cameras.near,
            far=base_cameras.far,
            metadata=metadata,
        ))
    return frames


def render_video(
    adapter: RendererAdapter,
    cameras: Iterable[CameraBatch],
    output_dir: str | Path,
    *,
    fps: int = 30,
    device: str | torch.device = "cuda",
    write_frames: bool = True,
) -> Path:
    """Render a camera path to an mp4 video."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("render_video requires imageio. Install imageio[ffmpeg].") from exc

    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, camera in enumerate(cameras):
        output = adapter.render(camera.to(device), mode="eval")
        metadata = camera.metadata or {}
        arr = tensor_to_uint8(_composite_output_rgb(output, metadata.get("eval_background_color")))
        frames.append(arr)
        if write_frames:
            Image.fromarray(arr).save(frame_dir / f"{idx:05d}.png")

    video_path = output_dir / "render_traj.mp4"
    iio.imwrite(video_path, np.asarray(frames), fps=fps)
    return video_path
