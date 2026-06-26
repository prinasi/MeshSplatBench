"""Tests for image loading details that affect benchmark metrics."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from tribench.core.datasets import (
    DTUDataset,
    _load_image,
)
from tribench.trainers.triangle_splatting_method import _pil_rgb_uint8
from tribench.vendor.training_images import is_dtu_scene, load_rgba_for_training
from tribench.vendor.triangle_splatting.utils.general_utils import PILtoTorch


def test_load_image_ignores_rgba_alpha_when_resizing(tmp_path):
    path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (200, 100, 50, 0),
            (200, 100, 50, 64),
            (200, 100, 50, 128),
            (200, 100, 50, 255),
        ]
    )
    image.save(path)

    loaded = _load_image(path, size=(1, 1))

    expected = torch.tensor([200 / 255.0, 100 / 255.0, 50 / 255.0])
    assert torch.allclose(loaded[0, 0], expected, atol=1 / 255)


def test_load_image_composites_rgba_when_background_is_given(tmp_path):
    path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (1, 1), (20, 40, 80, 0))
    image.save(path)

    loaded = _load_image(path, bg_color=(1.0, 1.0, 1.0))

    assert torch.allclose(loaded[0, 0], torch.ones(3), atol=1 / 255)


def test_dtu_dataset_composites_mask_to_white(tmp_path):
    root = tmp_path / "scan24"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir()

    image = Image.new("RGB", (4, 4), (255, 0, 0))
    image.save(root / "images" / "0000.png")

    mask = Image.new("L", (4, 4), 0)
    mask.putdata([255, 255, 0, 0] * 4)
    mask.save(root / "mask" / "000.png")

    K = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    world_mat = np.eye(4, dtype=np.float32)
    world_mat[:3, :4] = K @ extrinsic
    np.savez(root / "cameras.npz", world_mat_0=world_mat, scale_mat_0=np.eye(4, dtype=np.float32))

    dataset = DTUDataset(root, split="all", resolution=2)
    sample = dataset.sample(0)

    assert sample.image.shape == (2, 2, 3)
    assert torch.allclose(sample.image[:, 0], torch.tensor([1.0, 0.0, 0.0]).expand(2, 3), atol=1 / 255)
    assert torch.allclose(sample.image[:, 1], torch.ones(2, 3), atol=1 / 255)
    assert sample.mask is not None


def test_uint8_rgb_helper_extracts_rgb_without_compositing_alpha():
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (20, 40, 80, 0),
            (20, 40, 80, 64),
            (20, 40, 80, 128),
            (20, 40, 80, 255),
        ]
    )

    arr = _pil_rgb_uint8(image, (1, 1), resample=Image.Resampling.LANCZOS)

    assert arr.shape == (1, 1, 3)
    assert arr[0, 0].tolist() == [20, 40, 80]


def test_native_rgba_loader_composites_dtu_alpha_to_white():
    image = Image.new("RGBA", (3, 1))
    image.putdata(
        [
            (20, 40, 80, 0),
            (20, 40, 80, 128),
            (20, 40, 80, 255),
        ]
    )

    rgb, alpha = load_rgba_for_training(
        image,
        (3, 1),
        PILtoTorch,
        composite_white=True,
    )

    assert torch.allclose(rgb[:, 0, 0], torch.ones(3), atol=1 / 255)
    assert torch.allclose(rgb[:, 0, 2], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert torch.allclose(alpha[:, 0, 0], torch.zeros(1), atol=1 / 255)


def test_native_rgba_loader_preserves_non_dtu_alpha_rgb():
    image = Image.new("RGBA", (1, 1), (20, 40, 80, 0))

    rgb, alpha = load_rgba_for_training(
        image,
        (1, 1),
        PILtoTorch,
        composite_white=False,
    )

    assert torch.allclose(rgb[:, 0, 0], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert torch.allclose(alpha[:, 0, 0], torch.zeros(1), atol=1 / 255)


def test_is_dtu_scene_tolerates_missing_source_path():
    assert is_dtu_scene(SimpleNamespace()) is False


def test_is_dtu_scene_detects_cameras_npz(tmp_path):
    (tmp_path / "cameras.npz").write_bytes(b"")

    assert is_dtu_scene(SimpleNamespace(source_path=str(tmp_path))) is True
