"""Shared native-training image helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def is_dtu_scene(args) -> bool:
    source_path = getattr(args, "source_path", None)
    if source_path is None:
        return False
    return (Path(str(source_path)).expanduser() / "cameras.npz").is_file()


def load_rgba_for_training(image, resolution, pil_to_torch, *, composite_white: bool):
    rgb = torch.cat(
        [pil_to_torch(channel, resolution) for channel in image.split()[:3]],
        dim=0,
    )
    alpha = pil_to_torch(image.split()[3], resolution)
    if composite_white:
        rgb = rgb * alpha + (1.0 - alpha)
    return rgb, alpha
