"""Normalized camera representation for cross-method compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CameraBatch:
    """Normalized camera batch for all triangle splatting methods.
    
    All renderers accept this format at the framework boundary and convert
    to their native camera format internally via each adapter's
    ``_build_native_camera()`` method.
    
    Attributes:
        viewmats: World-to-camera matrices, shape [B, 4, 4].
        camtoworlds: Camera-to-world matrices, shape [B, 4, 4].
        Ks: Intrinsic matrices, shape [B, 3, 3].
        width: Image width in pixels.
        height: Image height in pixels.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        metadata: Optional backend-specific metadata.
    """

    viewmats: torch.Tensor
    camtoworlds: torch.Tensor
    Ks: torch.Tensor
    width: int
    height: int
    near: float = 0.01
    far: float = 100.0
    metadata: dict[str, Any] | None = None

    @property
    def batch_size(self) -> int:
        """Number of cameras in the batch."""
        return self.viewmats.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the camera tensors."""
        return self.viewmats.device

    def to(self, device: torch.device | str) -> CameraBatch:
        """Move all camera tensors to a device."""
        return CameraBatch(
            viewmats=self.viewmats.to(device),
            camtoworlds=self.camtoworlds.to(device),
            Ks=self.Ks.to(device),
            width=self.width,
            height=self.height,
            near=self.near,
            far=self.far,
            metadata=self.metadata,
        )
