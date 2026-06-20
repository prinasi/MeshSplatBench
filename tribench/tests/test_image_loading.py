"""Tests for image loading details that affect benchmark metrics."""

from __future__ import annotations

import torch
from PIL import Image

from tribench.core.datasets import _load_image
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
