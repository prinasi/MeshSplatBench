"""Adapter for MeshSplatting."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tribench.core.cameras import CameraBatch
from tribench.core.registry import register
from tribench.core.stats import compute_triangle_stats
from tribench.renderers.backends import get_backend
from tribench.renderers.base import RendererAdapter, RenderOutput


@register("mesh-splatting")
class MeshSplattingAdapter(RendererAdapter):
    """Renderer adapter for MeshSplatting.
    
    MeshSplatting uses:
    - Shared vertices with index-based topology
    - Triangle indices
    - Vertex weights for displacement
    - Global or model-level sigma
    - Mesh-like gradient scattering
    
    Integration: Preserves native CUDA renderer (diff-triangle-mesh-rasterization).
    Treats shared-vertex topology as first-class primitive type.
    Exposes topology statistics, vertex weight statistics, triangle area,
    and shared vertex gradient diagnostics.
    
    Repository: mesh-splatting/
    """

    def __init__(self, repo_root: str | Path | None = None):
        self._backend = get_backend("mesh-splatting")
        self.repo_root = self._backend.repo_root(repo_root)
        self._initialized = False
        self._model = None
        self._checkpoint_path: str | None = None
        self._dataset_path: str | None = None

        self._TriangleModel = None
        self._render = None
        self._getWorld2View2 = None
        self._getProjectionMatrix = None

    @property
    def name(self) -> str:
        return "mesh-splatting"

    @property
    def device(self) -> torch.device:
        if self._model is not None:
            vertices = getattr(self._model, "vertices", None)
            if isinstance(vertices, torch.Tensor):
                return vertices.device
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        return self._backend.status(self.repo_root)

    def _ensure_imports(self) -> None:
        if self._initialized:
            return

        try:
            from tribench.vendor.mesh_splatting.scene.triangle_model import TriangleModel
            from tribench.vendor.mesh_splatting.triangle_renderer import render
            from tribench.vendor.mesh_splatting.utils.graphics_utils import (
                getProjectionMatrix,
                getWorld2View2,
            )
        except ImportError as exc:
            raise ImportError(
                "Cannot import bundled mesh-splatting renderer. Reinstall "
                "TriBench with its bundled CUDA extensions built."
            ) from exc

        self._TriangleModel = TriangleModel
        self._render = render
        self._getWorld2View2 = getWorld2View2
        self._getProjectionMatrix = getProjectionMatrix
        self._initialized = True

    def load_checkpoint(self, path: str) -> None:
        """Load a mesh-splatting checkpoint directory or ``point_cloud_state_dict.pt``."""
        self._ensure_imports()

        ckpt_path = Path(path)
        ckpt_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
        state_path = ckpt_dir / "point_cloud_state_dict.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {state_path}")

        state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        features_rest = state["features_rest"]
        max_sh_degree = int(math.sqrt(features_rest.shape[1] + 1)) - 1

        model = self._TriangleModel(max_sh_degree)
        model.load_parameters(str(ckpt_dir), device="cuda")

        self._model = model
        self._checkpoint_path = str(state_path)

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        raise NotImplementedError(
            "MeshSplattingAdapter.load_scene() is not yet implemented. "
            "This should load a scene using mesh-splatting's scene/ module."
        )

    def model_stats(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        model = self._model
        vertices = model.get_vertices.detach().cpu()
        faces = model.get_triangle_indices.detach().cpu().long()
        tri_vertices = vertices[faces]
        vertex_weight = model.get_vertex_weight.detach().cpu().squeeze()
        tri_opacity = vertex_weight[faces].mean(dim=1)
        tri_stats = compute_triangle_stats(tri_vertices, tri_opacity)

        num_params = sum(
            p.numel()
            for p in [
                model.vertices,
                model.vertex_weight,
                model._features_dc,
                model._features_rest,
            ]
            if isinstance(p, torch.Tensor)
        )

        ckpt_size_mb = 0.0
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            ckpt_size_mb = os.path.getsize(self._checkpoint_path) / (1024 * 1024)

        weights_np = vertex_weight.numpy()
        return {
            "method": "mesh-splatting",
            "scene": self._dataset_path or "",
            "primitive_count": int(faces.shape[0]),
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "trainable_param_count": int(num_params),
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "triangle_area_mean": tri_stats.get("triangle_area_mean", 0.0),
            "triangle_area_median": tri_stats.get("triangle_area_median", 0.0),
            "triangle_area_p95": tri_stats.get("triangle_area_p95", 0.0),
            "opacity_mean": tri_stats.get("opacity_mean", 0.0),
            "opacity_median": tri_stats.get("opacity_median", 0.0),
            "opacity_histogram": tri_stats.get("opacity_histogram", []),
            "backend_extra": {
                "sigma": float(model.get_sigma),
                "active_sh_degree": int(model.active_sh_degree),
                "max_sh_degree": int(model.max_sh_degree),
                "vertex_weight_mean": float(np.mean(weights_np)),
                "vertex_weight_std": float(np.std(weights_np)),
                "vertex_weight_min": float(np.min(weights_np)),
                "vertex_weight_max": float(np.max(weights_np)),
            },
        }

    def _build_native_camera(self, cameras: CameraBatch, idx: int = 0):
        viewmat = cameras.viewmats[idx].cpu().numpy()
        R = viewmat[:3, :3].T
        T = viewmat[:3, 3]
        K = cameras.Ks[idx]
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        fovX = 2.0 * math.atan(cameras.width / (2.0 * fx))
        fovY = 2.0 * math.atan(cameras.height / (2.0 * fy))

        wvt = torch.tensor(
            self._getWorld2View2(R, T), dtype=torch.float32
        ).transpose(0, 1).cuda()
        proj = self._getProjectionMatrix(
            znear=cameras.near, zfar=cameras.far, fovX=fovX, fovY=fovY
        ).transpose(0, 1).cuda()
        full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        camera_center = wvt.inverse()[3, :3]

        return SimpleNamespace(
            FoVx=fovX,
            FoVy=fovY,
            image_height=cameras.height,
            image_width=cameras.width,
            world_view_transform=wvt,
            full_proj_transform=full_proj,
            camera_center=camera_center,
        )

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        self._ensure_imports()

        native_cam = self._build_native_camera(cameras, idx=0)
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False)
        bg_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        grad_ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

        with grad_ctx:
            rendering = self._render(native_cam, self._model, pipe, bg_color)

        rendered_image = rendering["render"]
        return RenderOutput(
            rgb=rendered_image.permute(1, 2, 0).contiguous(),
            alpha=rendering.get("rend_alpha", torch.empty(0)).squeeze(0)
            if "rend_alpha" in rendering else None,
            depth=rendering.get("surf_depth", torch.empty(0)).squeeze(0)
            if "surf_depth" in rendering else None,
            normal=rendering.get("rend_normal", torch.empty(0)).permute(1, 2, 0).contiguous()
            if "rend_normal" in rendering else None,
            radii=rendering.get("radii"),
            visibility=rendering.get("visibility_filter"),
            extras={
                "scaling": rendering.get("scaling"),
                "max_blending": rendering.get("max_blending"),
                "vertex_rendered": rendering.get("vertex_rendered"),
                "triangle_was_rendered": rendering.get("triangle_was_rendered"),
                "surf_normal": rendering.get("surf_normal"),
            },
        )

    def to_primitive(self) -> Any:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        from tribench.primitives.mesh_triangle import IndexedMeshTriangle

        model = self._model
        vertices = model.get_vertices.detach().clone()
        faces = model.get_triangle_indices.detach().clone().long()
        vertex_weight = model.get_vertex_weight.detach().clone().squeeze()
        face_opacity = vertex_weight[faces].mean(dim=1)

        return IndexedMeshTriangle(
            vertices=vertices,
            faces=faces,
            vertex_weights=vertex_weight,
            opacity=face_opacity,
            sigma=model.get_sigma,
        )
