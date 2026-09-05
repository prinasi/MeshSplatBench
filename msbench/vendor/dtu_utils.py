"""Shared helpers for DTU ``cameras.npz`` scenes used by vendored trainers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


def dtu_image_paths(scene_root: str | Path, image_dir: str = "images") -> list[Path]:
    """Return sorted DTU image paths from ``images/`` or ``image/``."""
    root = Path(scene_root)
    folders = [root / image_dir]
    if image_dir != "images":
        folders.append(root / "images")
    folders.append(root / "image")

    paths: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
            paths.extend(p for p in folder.glob(suffix) if not p.name.startswith("._"))
        if paths:
            break
    if not paths:
        raise FileNotFoundError(f"No DTU images found under {root}")
    return sorted(paths)


def load_dtu_pil_image(scene_root: str | Path, image_path: str | Path) -> Image.Image:
    """Load a DTU image without applying the separate ``mask/`` directory."""
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        return image.convert("RGBA")


def split_dtu_indices(
    num_images: int,
    eval: bool,
    llffhold: int = 8,
) -> tuple[list[int], list[int]]:
    """Return train/test indices matching the MeshSplatBench DTU evaluation split."""
    indices = list(range(num_images))
    if not eval:
        return indices, []
    holdout = max(1, int(llffhold))
    train = [idx for idx in indices if idx % holdout != 0]
    test = [idx for idx in indices if idx % holdout == 0]
    return train, test


def decompose_dtu_projection(
    camera_data,
    image_idx: int,
    *,
    width: int,
    height: int,
    original_width: int | None = None,
    original_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return ``K, native_R, native_T, fov_x, fov_y`` for one DTU camera.

    ``native_R``/``native_T`` follow the 3DGS-style convention used by the
    vendored triangle and mesh splatting renderers: ``getWorld2View2`` will
    transpose ``native_R`` to recover the world-to-camera rotation.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("DTU cameras.npz support requires opencv-python.") from exc

    world_mat = camera_data[f"world_mat_{image_idx}"].astype(np.float32)
    scale_key = f"scale_mat_{image_idx}"
    scale_mat = (
        camera_data[scale_key].astype(np.float32)
        if scale_key in camera_data.files
        else np.eye(4, dtype=np.float32)
    )
    projection = (world_mat @ scale_mat)[:3, :4]
    K, R, t, *_ = cv2.decomposeProjectionMatrix(projection)
    K = (K / K[2, 2]).astype(np.float32)

    if original_width is not None and original_height is not None:
        K[:2, :] *= np.array(
            [[width / float(original_width)], [height / float(original_height)]],
            dtype=np.float32,
        )

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T.astype(np.float32)
    c2w[:3, 3] = (t[:3] / t[3])[:, 0].astype(np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    native_R = w2c[:3, :3].T.astype(np.float32)
    native_T = w2c[:3, 3].astype(np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    fov_x = 2.0 * math.atan(float(width) / (2.0 * fx))
    fov_y = 2.0 * math.atan(float(height) / (2.0 * fy))
    return K[:3, :3], native_R, native_T, fov_x, fov_y


def random_dtu_points(
    num_points: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic normalized-space DTU point initialization."""
    rng = np.random.default_rng(seed)
    xyz = (rng.random((num_points, 3), dtype=np.float32) * 2.0) - 1.0
    rgb = (rng.random((num_points, 3), dtype=np.float32) * 255.0).astype(np.uint8)
    return xyz.astype(np.float32), rgb


def dtu_point_cloud(
    scene_root: str | Path,
    *,
    fetch_ply: Callable[[str], object],
    store_ply: Callable[[str, np.ndarray, np.ndarray], None],
    read_points3d_binary: Callable[[str], tuple[np.ndarray, np.ndarray, object]],
    read_points3d_text: Callable[[str], tuple[np.ndarray, np.ndarray, object]],
    point_cloud_cls: type,
    num_random_points: int = 100_000,
) -> tuple[object, str]:
    """Load or create the point cloud required by native triangle initializers."""
    root = Path(scene_root)
    ply_candidates = [
        root / "sparse" / "0" / "points3D.ply",
        root / "sparse" / "points3D.ply",
        root / "points3D.ply",
        root / "points3d.ply",
        root / "points3d_dtu.ply",
    ]
    for ply_path in ply_candidates:
        if ply_path.is_file():
            return fetch_ply(str(ply_path)), str(ply_path)

    bin_txt_pairs = [
        (
            root / "sparse" / "0" / "points3D.bin",
            root / "sparse" / "0" / "points3D.txt",
        ),
        (root / "sparse" / "points3D.bin", root / "sparse" / "points3D.txt"),
    ]
    output_path = root / "points3d_dtu.ply"
    for bin_path, txt_path in bin_txt_pairs:
        if bin_path.is_file():
            xyz, rgb, _ = read_points3d_binary(str(bin_path))
            store_ply(str(output_path), xyz, rgb)
            return fetch_ply(str(output_path)), str(output_path)
        if txt_path.is_file():
            xyz, rgb, _ = read_points3d_text(str(txt_path))
            store_ply(str(output_path), xyz, rgb)
            return fetch_ply(str(output_path)), str(output_path)

    xyz, rgb = random_dtu_points(num_random_points)
    store_ply(str(output_path), xyz, rgb)
    normals = np.zeros_like(xyz, dtype=np.float32)
    return (
        point_cloud_cls(points=xyz, colors=rgb.astype(np.float32) / 255.0, normals=normals),
        str(output_path),
    )
