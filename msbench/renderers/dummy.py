"""CPU-only dummy renderer for macOS / no-CUDA environments.

Returns placeholder images (gradient + camera info overlay) so the viewer
UI is fully functional even without a CUDA rasterizer backend.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from msbench.core.cameras import CameraBatch
from msbench.renderers.base import RenderOutput, RendererAdapter


class DummyRenderer(RendererAdapter):
    """Placeholder renderer that returns gradient images with pose overlay.

    Used when CUDA backends are unavailable (e.g. macOS). The viewer UI,
    dataset endpoints, point cloud endpoints, and all protocol handling
    work normally — only the rendered image is a placeholder.
    """

    def __init__(self, method_name: str = "dummy") -> None:
        self._method_name = method_name
        self._checkpoint: str | None = None
        self._dataset_path: str | None = None
        self._split = "test"
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def device(self) -> torch.device:
        return self._device

    def load_checkpoint(self, path: str) -> None:
        self._checkpoint = path

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        self._dataset_path = dataset_path
        self._split = split

    def model_stats(self) -> dict[str, Any]:
        return {
            "method": self._method_name,
            "checkpoint": self._checkpoint,
            "device": "cpu (dummy)",
            "num_params": 0,
            "cuda_available": torch.cuda.is_available(),
        }

    def to_primitive(self) -> Any:
        """Return a small dummy triangle soup for geometry endpoints."""
        from msbench.primitives.triangle import IndependentTriangle
        # Create a small set of random triangles
        N = 100
        verts = torch.randn(N, 3, 3) * 0.5
        opacity = torch.ones(N)
        return IndependentTriangle(vertices=verts, opacity=opacity)

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        B = cameras.batch_size
        H = int(cameras.height)
        W = int(cameras.width)

        # Generate a gradient placeholder image
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = np.linspace(20, 60 + c * 20, W, dtype=np.uint8)[None, :]

        # Add text overlay with camera info
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Extract pose info
        try:
            c2w = cameras.camtoworlds[0]
            pos = c2w[:3, 3]
            txt = f"[DummyRenderer] {self._method_name}\n"
            txt += f"pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
            txt += f"res: {W}x{H}  CUDA: {'yes' if torch.cuda.is_available() else 'no'}"
        except Exception:
            txt = f"[DummyRenderer] {self._method_name}"

        # Draw text at center
        y0 = H // 2 - 30
        for line in txt.split("\n"):
            draw.text((W // 2 - 120, y0), line, fill=(255, 255, 255), font=font)
            y0 += 15

        # Draw crosshair
        cx, cy = W // 2, H // 2
        draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(100, 200, 100), width=1)
        draw.line([(cx, cy - 20), (cx, cy + 20)], fill=(100, 200, 100), width=1)

        rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Simple depth: distance from origin
        depth = torch.full((B, 1, H, W), 2.0, dtype=torch.float32)

        # Alpha = 1 everywhere
        alpha = torch.ones(B, 1, H, W, dtype=torch.float32)

        # Normal: facing camera
        normal = torch.zeros(B, 3, H, W, dtype=torch.float32)
        normal[:, 2, :, :] = 1.0

        return RenderOutput(
            rgb=rgb_tensor,
            alpha=alpha,
            depth=depth,
            normal=normal,
        )
