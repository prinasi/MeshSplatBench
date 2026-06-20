"""Dataset loading abstractions for TriBench."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribench.core.cameras import CameraBatch


@dataclass
class DatasetSample:
    """One camera/image sample."""

    camera: CameraBatch
    image: torch.Tensor | None
    name: str
    image_path: Path | None = None
    mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


def _resolve_resolution(
    width: int,
    height: int,
    resolution: int = 1,
) -> tuple[int, int]:
    """Match the native 3DGS/triangle-splatting resolution semantics."""
    if resolution in {1, 2, 4, 8}:
        new_width = round(width / resolution)
        new_height = round(height / resolution)
    else:
        if resolution == -1:
            global_down = width / 1600 if width > 1600 else 1
        else:
            global_down = width / resolution
        new_width = int(width / global_down)
        new_height = int(height / global_down)

    return max(1, new_width), max(1, new_height)


def _pil_rgb_array(
    image: Image.Image,
    size: tuple[int, int] | None = None,
    *,
    resample: int | None = None,
) -> np.ndarray:
    """Load RGB values without compositing or premultiplying alpha."""
    bands = image.getbands()
    if len(bands) >= 3 and bands[:3] == ("R", "G", "B"):
        channels = image.split()[:3]
        if size is not None and image.size != size:
            channels = tuple(channel.resize(size, resample) for channel in channels)
        return np.stack([np.asarray(channel) for channel in channels], axis=-1)
    if len(bands) >= 1:
        gray = image.split()[0]
        if size is not None and image.size != size:
            gray = gray.resize(size, resample)
        return np.repeat(np.asarray(gray)[..., None], 3, axis=-1)

    rgb = image.convert("RGB")
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, resample)
    return np.asarray(rgb)


def _load_image(path: Path | None, size: tuple[int, int] | None = None) -> torch.Tensor | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as image:
        arr = _pil_rgb_array(image, size)
    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr[..., :3]).contiguous()


def _camera_batch_from_w2c(
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    near: float = 0.01,
    far: float = 100.0,
    metadata: dict[str, Any] | None = None,
) -> CameraBatch:
    w2c = w2c.astype(np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    return CameraBatch(
        viewmats=torch.from_numpy(w2c).unsqueeze(0),
        camtoworlds=torch.from_numpy(c2w).unsqueeze(0),
        Ks=torch.from_numpy(K.astype(np.float32)).unsqueeze(0),
        width=int(width),
        height=int(height),
        near=float(near),
        far=float(far),
        metadata=metadata,
    )


def _resolve_image_path(root: Path, image_dir: str, name: str) -> Path | None:
    candidates = [
        root / image_dir / name,
        root / "images" / name,
        root / name,
    ]
    stem = Path(name).stem
    for folder in [root / image_dir, root / "images", root]:
        for suffix in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
            candidates.append(folder / f"{stem}{suffix}")
    for path in candidates:
        if path.exists():
            return path
    return None


def _split_indices(num: int, split: str, eval_every: int) -> list[int]:
    if split in {"all", "full"}:
        return list(range(num))
    if split in {"test", "val", "validation"}:
        return [i for i in range(num) if i % eval_every == 0]
    if split == "train":
        return [i for i in range(num) if i % eval_every != 0]
    raise ValueError(f"Unknown split '{split}'. Expected train/test/all.")


class DatasetBase(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, dataset_path: str | Path, split: str = "train"):
        self.dataset_path = Path(dataset_path)
        self._split = split

    @property
    def split(self) -> str:
        return self._split

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def sample(self, idx: int) -> DatasetSample:
        ...

    def __getitem__(self, idx: int) -> tuple[CameraBatch, torch.Tensor | None]:
        sample = self.sample(idx)
        return sample.camera, sample.image

    def get_name(self, idx: int) -> str:
        return self.sample(idx).name

    def get_all_cameras(self) -> CameraBatch:
        samples = [self.sample(i) for i in range(len(self))]
        return CameraBatch(
            viewmats=torch.cat([s.camera.viewmats for s in samples], dim=0),
            camtoworlds=torch.cat([s.camera.camtoworlds for s in samples], dim=0),
            Ks=torch.cat([s.camera.Ks for s in samples], dim=0),
            width=samples[0].camera.width,
            height=samples[0].camera.height,
            near=samples[0].camera.near,
            far=samples[0].camera.far,
        )


class ColmapDataset(DatasetBase):
    """Dataset loader for COLMAP scenes such as MipNeRF-360 and Tanks & Temples."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        eval_every: int = 8,
        image_dir: str = "images",
        resolution: int = 1,
    ):
        super().__init__(dataset_path, split)
        self.eval_every = eval_every
        self.image_dir = image_dir
        self.resolution = resolution
        self._samples = self._load_samples()

    def _load_samples(self) -> list[DatasetSample]:
        from tribench.vendor.triangle_splatting.scene.colmap_loader import (
            read_extrinsics_binary,
            read_extrinsics_text,
            read_intrinsics_binary,
            read_intrinsics_text,
        )

        sparse_dir = self.dataset_path / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = self.dataset_path / "sparse"
        if not sparse_dir.exists():
            raise FileNotFoundError(f"COLMAP sparse directory not found under {self.dataset_path}")

        try:
            extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
            intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        except Exception:
            extrinsics = read_extrinsics_text(str(sparse_dir / "images.txt"))
            intrinsics = read_intrinsics_text(str(sparse_dir / "cameras.txt"))

        images = sorted(extrinsics.values(), key=lambda im: im.name)
        selected = _split_indices(len(images), self.split, self.eval_every)
        samples: list[DatasetSample] = []
        for out_idx, image_idx in enumerate(selected):
            im = images[image_idx]
            cam = intrinsics[im.camera_id]
            colmap_width, colmap_height = int(cam.width), int(cam.height)
            params = np.asarray(cam.params, dtype=np.float32)
            if cam.model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            image_path = _resolve_image_path(self.dataset_path, self.image_dir, im.name)
            if image_path is not None and image_path.exists():
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            else:
                image_width, image_height = colmap_width, colmap_height

            width, height = _resolve_resolution(image_width, image_height, self.resolution)
            scale_x = width / colmap_width
            scale_y = height / colmap_height
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = im.qvec2rotmat()
            w2c[:3, 3] = im.tvec

            gt = _load_image(image_path, size=(width, height))
            camera = _camera_batch_from_w2c(
                w2c,
                K,
                width,
                height,
                metadata={"image_name": im.name, "index": image_idx, "split_index": out_idx},
            )
            samples.append(DatasetSample(camera=camera, image=gt, name=Path(im.name).stem, image_path=image_path))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, idx: int) -> DatasetSample:
        return self._samples[idx]


class NerfSyntheticDataset(DatasetBase):
    """Dataset loader for NeRF Synthetic/Blender transforms JSON scenes."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        resolution: int = 1,
    ):
        super().__init__(dataset_path, split)
        self.resolution = resolution
        self._samples = self._load_samples()

    def _load_samples(self) -> list[DatasetSample]:
        split_name = "test" if self.split in {"test", "val", "validation"} else self.split
        transform_path = self.dataset_path / f"transforms_{split_name}.json"
        if not transform_path.exists() and self.split == "all":
            paths = sorted(self.dataset_path.glob("transforms_*.json"))
        elif transform_path.exists():
            paths = [transform_path]
        else:
            raise FileNotFoundError(f"Cannot find {transform_path}")

        samples: list[DatasetSample] = []
        for path in paths:
            meta = json.loads(path.read_text())
            camera_angle_x = float(meta["camera_angle_x"])
            for frame_idx, frame in enumerate(meta["frames"]):
                file_path = Path(frame["file_path"])
                if not file_path.suffix:
                    file_path = file_path.with_suffix(".png")
                image_path = self.dataset_path / file_path
                gt = _load_image(image_path)
                if gt is not None:
                    image_height, image_width = gt.shape[:2]
                else:
                    image_width = int(meta.get("w", 800))
                    image_height = int(meta.get("h", 800))
                width, height = _resolve_resolution(image_width, image_height, self.resolution)
                if gt is not None and (gt.shape[1], gt.shape[0]) != (width, height):
                    gt = _load_image(image_path, size=(width, height))
                fx = 0.5 * width / np.tan(0.5 * camera_angle_x)
                fy = fx
                K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float32)
                c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
                # Blender/OpenGL to COLMAP/OpenCV convention.
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                camera = _camera_batch_from_w2c(
                    w2c,
                    K,
                    width,
                    height,
                    metadata={"image_name": image_path.name, "source_json": path.name},
                )
                samples.append(DatasetSample(camera=camera, image=gt, name=image_path.stem, image_path=image_path))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, idx: int) -> DatasetSample:
        return self._samples[idx]


class DTUDataset(DatasetBase):
    """DTU loader using ``cameras.npz`` plus ``images/`` and optional ``mask/``."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        eval_every: int = 8,
        resolution: int = 1,
    ):
        super().__init__(dataset_path, split)
        self.eval_every = eval_every
        self.resolution = resolution
        self._samples = self._load_samples()

    def _load_samples(self) -> list[DatasetSample]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("DTUDataset requires opencv-python for projection matrix decomposition.") from exc

        camera_file = self.dataset_path / "cameras.npz"
        if not camera_file.exists():
            raise FileNotFoundError(f"DTU cameras.npz not found: {camera_file}")
        data = np.load(camera_file)
        image_paths = sorted((self.dataset_path / "images").glob("*.png"))
        if not image_paths:
            image_paths = sorted((self.dataset_path / "image").glob("*.png"))
        selected = _split_indices(len(image_paths), self.split, self.eval_every)
        samples: list[DatasetSample] = []
        for out_idx, image_idx in enumerate(selected):
            image_path = image_paths[image_idx]
            gt = _load_image(image_path)
            if gt is None:
                continue
            image_height, image_width = gt.shape[:2]
            width, height = _resolve_resolution(image_width, image_height, self.resolution)
            if (image_width, image_height) != (width, height):
                gt = _load_image(image_path, size=(width, height))
            world_mat = data[f"world_mat_{image_idx}"].astype(np.float32)
            scale_key = f"scale_mat_{image_idx}"
            scale_mat = (
                data[scale_key].astype(np.float32)
                if scale_key in data.files
                else np.eye(4, dtype=np.float32)
            )
            P = (world_mat @ scale_mat)[:3, :4]
            K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
            K = K / K[2, 2]
            K[:2, :] *= np.array(
                [[width / image_width], [height / image_height]], dtype=np.float32
            )
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = (t[:3] / t[3])[:, 0]
            w2c = np.linalg.inv(c2w)
            mask_path = self.dataset_path / "mask" / image_path.name
            mask = _load_image(mask_path) if mask_path.exists() else None
            camera = _camera_batch_from_w2c(
                w2c,
                K[:3, :3],
                width,
                height,
                metadata={"image_name": image_path.name, "index": image_idx, "split_index": out_idx},
            )
            samples.append(DatasetSample(camera=camera, image=gt, name=image_path.stem, image_path=image_path, mask=mask))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, idx: int) -> DatasetSample:
        return self._samples[idx]


def infer_dataset_type(dataset_path: str | Path, dataset_type: str = "auto") -> str:
    root = Path(dataset_path)
    if dataset_type != "auto":
        return dataset_type
    if (root / "cameras.npz").exists():
        return "dtu"
    if (root / "transforms_train.json").exists() or (root / "transforms_test.json").exists():
        return "blender"
    if (root / "sparse").exists():
        return "colmap"
    raise FileNotFoundError(f"Cannot infer dataset type from {root}")


def load_dataset(
    dataset_path: str | Path,
    *,
    dataset_type: str = "auto",
    split: str = "test",
    eval_every: int = 8,
    image_dir: str = "images",
    resolution: int = 1,
) -> DatasetBase:
    dtype = infer_dataset_type(dataset_path, dataset_type)
    if dtype in {"colmap", "mipnerf360", "tanks", "tanksandtemples", "tankstemple"}:
        return ColmapDataset(
            dataset_path,
            split=split,
            eval_every=eval_every,
            image_dir=image_dir,
            resolution=resolution,
        )
    if dtype in {"blender", "nerf-synthetic", "nerf_synthetic"}:
        return NerfSyntheticDataset(dataset_path, split=split, resolution=resolution)
    if dtype == "dtu":
        return DTUDataset(dataset_path, split=split, eval_every=eval_every, resolution=resolution)
    raise KeyError(f"Unknown dataset type '{dtype}'")


_DATASET_REGISTRY: dict[str, type[DatasetBase]] = {
    "colmap": ColmapDataset,
    "mipnerf360": ColmapDataset,
    "tanks": ColmapDataset,
    "dtu": DTUDataset,
    "blender": NerfSyntheticDataset,
}


def register_dataset(name: str, dataset_cls: type[DatasetBase]) -> None:
    _DATASET_REGISTRY[name] = dataset_cls


def get_dataset(name: str) -> type[DatasetBase]:
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return _DATASET_REGISTRY[name]


def list_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY.keys())
