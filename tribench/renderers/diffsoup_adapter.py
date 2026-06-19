"""Adapter for DiffSoup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from tribench.core.cameras import CameraBatch
from tribench.core.registry import register
from tribench.core.stats import compute_triangle_stats
from tribench.renderers.backends import get_backend
from tribench.renderers.base import RendererAdapter, RenderOutput


@register("diffsoup")
class DiffSoupAdapter(RendererAdapter):
    """Renderer adapter for DiffSoup (Direct Differentiable Rasterization of Triangle Soup).
    
    DiffSoup uses:
    - Vertices and faces
    - Multiresolution triangle features
    - Multiresolution opacity
    - ColorMLP for appearance
    - Stochastic opacity masking
    - Depth-tested triangle soup
    
    Integration: Supports as a special backend without forcing into the same
    CUDA renderer abstraction. Normalizes stats and reports through the adapter interface.
    
    Repository: diffsoup/
    """

    def __init__(self, repo_root: str | Path | None = None):
        self._backend = get_backend("diffsoup")
        self.repo_root = self._backend.repo_root(repo_root)
        self._initialized = False
        self._ds = None
        self._checkpoint: dict[str, Any] | None = None
        self._checkpoint_path: str | None = None
        self._dataset_path: str | None = None

    @property
    def name(self) -> str:
        return "diffsoup"

    @property
    def device(self) -> torch.device:
        if self._checkpoint is not None and isinstance(self._checkpoint.get("V"), torch.Tensor):
            return self._checkpoint["V"].device
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        return self._backend.status(self.repo_root)

    def _ensure_imports(self) -> None:
        if self._initialized:
            return

        try:
            import tribench.vendor.diffsoup as ds
        except ImportError as exc:
            raise ImportError(
                "Cannot import bundled DiffSoup renderer. Reinstall TriBench "
                "with its bundled CMake/CUDA extension built."
            ) from exc

        self._ds = ds
        self._initialized = True

    def load_checkpoint(self, path: str) -> None:
        """Load a DiffSoup ``final_params.pt``-style checkpoint."""
        self._ensure_imports()

        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "final_params.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        required = {"V", "F", "feat_acc", "alpha_acc", "Rmax", "feat_dim"}
        missing = sorted(required.difference(ckpt))
        if missing:
            raise KeyError(f"DiffSoup checkpoint missing keys: {', '.join(missing)}")

        self._checkpoint = ckpt
        self._checkpoint_path = str(ckpt_path)

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        raise NotImplementedError(
            "DiffSoupAdapter.load_scene() is not yet implemented. "
            "This should load a scene using diffsoup's data loading utilities."
        )

    def model_stats(self) -> dict[str, Any]:
        if self._checkpoint is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        ckpt = self._checkpoint
        vertices = ckpt["V"].float()
        faces = ckpt["F"].long()
        tri_vertices = vertices[faces]
        alpha_acc = ckpt["alpha_acc"].float()
        tri_opacity = alpha_acc.reshape(alpha_acc.shape[0], -1).mean(dim=1)
        tri_stats = compute_triangle_stats(tri_vertices, tri_opacity)

        mlp_params = 0
        color_mlp = ckpt.get("color_mlp", {})
        if isinstance(color_mlp, dict):
            mlp_params = sum(v.numel() for v in color_mlp.values() if isinstance(v, torch.Tensor))

        ckpt_size_mb = 0.0
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            ckpt_size_mb = os.path.getsize(self._checkpoint_path) / (1024 * 1024)

        return {
            "method": "diffsoup",
            "scene": self._dataset_path or "",
            "primitive_count": int(faces.shape[0]),
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "trainable_param_count": int(
                ckpt["feat_acc"].numel() + ckpt["alpha_acc"].numel() + mlp_params
            ),
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "triangle_area_mean": tri_stats.get("triangle_area_mean", 0.0),
            "triangle_area_median": tri_stats.get("triangle_area_median", 0.0),
            "triangle_area_p95": tri_stats.get("triangle_area_p95", 0.0),
            "opacity_mean": tri_stats.get("opacity_mean", 0.0),
            "opacity_median": tri_stats.get("opacity_median", 0.0),
            "opacity_histogram": tri_stats.get("opacity_histogram", []),
            "backend_extra": {
                "Rmin": int(ckpt.get("Rmin", 0)),
                "Rmax": int(ckpt["Rmax"]),
                "feat_dim": int(ckpt["feat_dim"]),
                "color_mlp_param_count": int(mlp_params),
                "texels_per_face": int(alpha_acc.reshape(alpha_acc.shape[0], -1).shape[1]),
            },
        }

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        raise NotImplementedError(
            "DiffSoupAdapter.render() needs a trained ColorMLP reconstruction path. "
            "The adapter can load/check DiffSoup CUDA modules and report checkpoint "
            "statistics; use the native DiffSoup examples for full rendering until "
            "that path is mapped into TriBench."
        )

    def to_primitive(self) -> Any:
        if self._checkpoint is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        from tribench.primitives.mesh_triangle import IndexedMeshTriangle

        ckpt = self._checkpoint
        vertices = ckpt["V"].float()
        faces = ckpt["F"].long()
        alpha_acc = ckpt["alpha_acc"].float()
        opacity = alpha_acc.reshape(alpha_acc.shape[0], -1).mean(dim=1)
        return IndexedMeshTriangle(vertices=vertices, faces=faces, opacity=opacity)
