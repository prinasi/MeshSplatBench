"""Tests for image loading details that affect benchmark metrics."""

from __future__ import annotations

import torch
from PIL import Image

from tribench.core.datasets import (
    _composite_image_with_mask,
    _load_image,
    _load_mask,
    _resolve_dtu_mask_path,
)
from tribench.trainers.triangle_splatting_method import _pil_rgb_uint8


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


def test_mask_composite_uses_white_background(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (2, 1))
    mask.putdata([255, 0])
    mask.save(mask_path)

    image = torch.zeros(1, 2, 3)
    loaded_mask = _load_mask(mask_path)
    composited = _composite_image_with_mask(image, loaded_mask, (1.0, 1.0, 1.0))

    assert torch.allclose(composited[0, 0], torch.zeros(3))
    assert torch.allclose(composited[0, 1], torch.ones(3))


def test_resolve_dtu_mask_path_accepts_three_digit_masks(tmp_path):
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    mask_path = mask_dir / "000.png"
    Image.new("L", (1, 1), 255).save(mask_path)

    resolved = _resolve_dtu_mask_path(mask_dir, tmp_path / "images" / "0000.png", 0)

    assert resolved == mask_path


def test_triangle_splatting_train_loader_ignores_rgba_alpha():
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
