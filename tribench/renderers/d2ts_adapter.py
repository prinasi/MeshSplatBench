"""Adapter for 2DTS (2D Triangle Splatting)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.special
import torch

from tribench.core.cameras import CameraBatch
from tribench.core.registry import register
from tribench.core.stats import compute_triangle_stats
from tribench.renderers.backends import get_backend
from tribench.renderers.base import RendererAdapter, RenderOutput


@register("2dts")
class D2TSAdapter(RendererAdapter):
    """Renderer adapter for 2D Triangle Splatting (2DTS).
    
    2DTS represents scenes as independent triangle soup with:
    - Vertices: [T, 3, 3] tensor
    - Opacity per triangle
    - SH or feature color
    - Gamma-controlled compactness (gamma annealing)
    - Optional back-face culling and sorting modes
    
    Integration: Preserves native CUDA renderer (diff-triangle-rasterization).
    Exposes gamma, opacity, area, contribution, and screen-space statistics.
    
    Repository: 2dts/
    """

    def __init__(self, repo_root: str | Path | None = None):
        self._backend = get_backend("2dts")
        self.repo_root = self._backend.repo_root(repo_root)
        self._initialized = False
        self._TriangleRenderer = None
        self._Camera = None
        self._model = None
        self._checkpoint_path: str | None = None
        self._dataset_path: str | None = None
        self.gamma = 1.0
        self.max_sh_degree = 0
        self.active_sh_degree = 0
        self.back_culling = False
        self.sort_level = 0
        # Rendering fidelity options (mirror VanillaTSModel.forward).
        # gamma_rescale: shrink triangles to keep opacity integral invariant.
        self.gamma_rescale: bool = False
        # ste_threshold: straight-through opacity binarisation threshold.
        # Triangles with sigmoid(opacity) < threshold are zeroed at eval time.
        self.ste_threshold: float | None = None
        # opacity_floor: minimum opacity after sigmoid mapping [floor, 1].
        self.opacity_floor: float = 0.0
        self._bg_color_str: str = "white"

    @property
    def name(self) -> str:
        return "2dts"

    @property
    def device(self) -> torch.device:
        if self._model is not None:
            vertex = getattr(self._model, "vertex", None)
            if isinstance(vertex, torch.Tensor):
                return vertex.device
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        return self._backend.status(self.repo_root)

    def _ensure_imports(self) -> None:
        """Import bundled 2DTS modules."""
        if self._initialized:
            return

        try:
            from tribench.vendor.d2ts.diff_recon.renderer.triangle_renderer import TriangleRenderer
            from tribench.vendor.d2ts.diff_recon.models.raw_triangle import RawTriangle
            from tribench.vendor.d2ts.diff_recon.utils.camera import Camera
        except ImportError as exc:
            raise ImportError(
                "Cannot import bundled 2DTS renderer. Reinstall TriBench "
                "with its bundled CUDA extensions built."
            ) from exc

        self._TriangleRenderer = TriangleRenderer
        self._RawTriangle = RawTriangle
        self._Camera = Camera
        self._initialized = True

    def load_checkpoint(self, path: str) -> None:
        """Load a 2DTS checkpoint or exported point-cloud PLY."""
        self._ensure_imports()

        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            ckpt_candidates = [
                *(ckpt_path / "ckpt").glob("*.ckpt"),
                *ckpt_path.glob("*.ckpt"),
                *ckpt_path.glob("*.pth"),
                *ckpt_path.glob("*.pt"),
            ]
            candidates = sorted(ckpt_candidates, key=_checkpoint_sort_key)
            if not candidates:
                ply_candidates = [
                    *(ckpt_path / "point_cloud").glob("*.ply"),
                    *ckpt_path.glob("*.ply"),
                ]
                candidates = sorted(ply_candidates, key=_checkpoint_sort_key)
            if not candidates:
                raise FileNotFoundError(f"No .ckpt/.pt/.pth checkpoint or point_cloud/*.ply found under {ckpt_path}")
            ckpt_path = candidates[-1]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if ckpt_path.suffix.lower() == ".ply":
            self._load_ply_checkpoint(ckpt_path)
            return

        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, tuple):
            state_dict = loaded[0]
            if len(loaded) >= 4:
                self.gamma = float(loaded[3])
        elif isinstance(loaded, dict):
            state_dict = loaded
            self.gamma = float(loaded.get("gamma", self.gamma))
        else:
            raise TypeError(f"Unsupported 2DTS checkpoint type: {type(loaded)!r}")

        # Auto-detect gamma_rescale: if gamma was annealed beyond 1.0 the
        # model was almost certainly trained with triangle rescaling enabled.
        if self.gamma > 1.0 and not self.gamma_rescale:
            self.gamma_rescale = True

        required = {"_vertex", "_opacity", "_f_dc", "_f_rest"}
        missing = sorted(required.difference(state_dict))
        if missing:
            raise KeyError(f"2DTS checkpoint missing keys: {', '.join(missing)}")

        device = _checkpoint_device()
        vertex = state_dict["_vertex"].to(device, dtype=torch.float32).detach().clone()
        opacity = state_dict["_opacity"].to(device, dtype=torch.float32).detach().clone()
        f_dc = state_dict["_f_dc"].to(device, dtype=torch.float32).detach().clone()
        f_rest = state_dict["_f_rest"].to(device, dtype=torch.float32).detach().clone()

        coeff_dim = f_rest.shape[-2] + 1
        self.max_sh_degree = int(math.sqrt(coeff_dim) - 1)
        self.active_sh_degree = int(state_dict.get("active_sh_degree", self.max_sh_degree))
        self._model = SimpleNamespace(
            vertex=vertex.requires_grad_(True),
            opacity=opacity.requires_grad_(True),
            f_dc=f_dc.requires_grad_(True),
            f_rest=f_rest.requires_grad_(True),
        )
        self._checkpoint_path = str(ckpt_path)

    def _load_ply_checkpoint(self, path: Path) -> None:
        triangle_model = self._RawTriangle(ply_path=str(path))
        if len(triangle_model) == 0:
            raise ValueError(f"2DTS point-cloud PLY contains no triangles: {path}")

        device = _checkpoint_device()
        vertex = torch.as_tensor(triangle_model.vertex, dtype=torch.float32, device=device)
        opacity = torch.as_tensor(triangle_model.opacity, dtype=torch.float32, device=device)
        shs = torch.as_tensor(triangle_model.shs, dtype=torch.float32, device=device)
        if shs.ndim == 2:
            shs = shs.view(shs.shape[0], -1, 3)
        elif shs.ndim == 3:
            raise ValueError("2DTS vertex-color PLY checkpoints are not supported by this adapter yet.")
        else:
            raise ValueError(f"Unsupported 2DTS SH tensor shape in PLY: {tuple(shs.shape)}")

        coeff_dim = shs.shape[-2]
        self.max_sh_degree = int(math.sqrt(coeff_dim) - 1)
        self.active_sh_degree = self.max_sh_degree
        self._model = SimpleNamespace(
            vertex=vertex.requires_grad_(True),
            opacity=opacity.requires_grad_(True),
            f_dc=shs[:, :1, :].contiguous().requires_grad_(True),
            f_rest=shs[:, 1:, :].contiguous().requires_grad_(True),
        )
        self._checkpoint_path = str(path)

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        raise NotImplementedError(
            "D2TSAdapter.load_scene() is not yet implemented. "
            "This should load a COLMAP or NeRF Synthetic scene using "
            "the 2dts/ dataset loader (src/diff_recon/datasets/)."
        )

    def configure(self, **kwargs: Any) -> None:
        """Set rendering parameters that mirror the original training config.

        Accepted keys:
            gamma_rescale (bool): Enable triangle vertex rescaling. Native
                2DTS applies this whenever configured, even when gamma == 1.
            ste_threshold (float | None): Straight-through estimator
                threshold for opacity binarisation (e.g. 0.3 for DTU).
            opacity_floor (float): Minimum opacity after sigmoid mapping.
            sort_level (int): Depth-sort granularity (0/1/2).
            back_culling (bool): Enable back-face culling.
            bg_color (str): Background colour name ("white"/"black").
        """
        for key, value in kwargs.items():
            if key == "gamma_rescale":
                self.gamma_rescale = bool(value)
            elif key == "ste_threshold":
                self.ste_threshold = float(value) if value is not None else None
            elif key == "opacity_floor":
                self.opacity_floor = float(value)
            elif key == "sort_level":
                self.sort_level = int(value)
            elif key == "back_culling":
                self.back_culling = bool(value)
            elif key == "bg_color":
                self._bg_color_str = str(value)
            else:
                raise KeyError(f"Unknown D2TSAdapter config key: {key!r}")

    def model_stats(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        vertices = self._model.vertex.detach().cpu()
        opacity = torch.sigmoid(self._model.opacity.detach().cpu()).squeeze()
        tri_stats = compute_triangle_stats(vertices, opacity)
        num_params = (
            self._model.vertex.numel()
            + self._model.opacity.numel()
            + self._model.f_dc.numel()
            + self._model.f_rest.numel()
        )
        ckpt_size_mb = 0.0
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            ckpt_size_mb = os.path.getsize(self._checkpoint_path) / (1024 * 1024)

        return {
            "method": "2dts",
            "scene": self._dataset_path or "",
            "primitive_count": int(vertices.shape[0]),
            "vertex_count": int(vertices.shape[0] * 3),
            "face_count": int(vertices.shape[0]),
            "trainable_param_count": int(num_params),
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "triangle_area_mean": tri_stats.get("triangle_area_mean", 0.0),
            "triangle_area_median": tri_stats.get("triangle_area_median", 0.0),
            "triangle_area_p95": tri_stats.get("triangle_area_p95", 0.0),
            "opacity_mean": tri_stats.get("opacity_mean", 0.0),
            "opacity_median": tri_stats.get("opacity_median", 0.0),
            "opacity_histogram": tri_stats.get("opacity_histogram", []),
            "backend_extra": {
                "gamma": float(self.gamma),
                "gamma_rescale": bool(self.gamma_rescale),
                "ste_threshold": self.ste_threshold,
                "opacity_floor": float(self.opacity_floor),
                "max_sh_degree": int(self.max_sh_degree),
                "active_sh_degree": int(self.active_sh_degree),
                "back_culling": bool(self.back_culling),
                "sort_level": int(self.sort_level),
                "bg_color": self._bg_color_str,
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
        return self._Camera(
            R=R,
            T=T,
            FoVx=fovX,
            FoVy=fovY,
            image_width=cameras.width,
            image_height=cameras.height,
            znear=cameras.near,
            zfar=cameras.far,
        ).to(_checkpoint_device())

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        self._ensure_imports()

        native_cam = self._build_native_camera(cameras, idx=0)
        vertex = self._model.vertex
        if vertex.device.type != "cuda":
            raise RuntimeError("2DTS image rendering requires an available CUDA GPU.")
        shs = torch.cat((self._model.f_dc, self._model.f_rest), dim=-2)

        # --- Opacity: sigmoid → floor → optional STE binarisation ----------
        opacity = (
            torch.sigmoid(self._model.opacity) * (1.0 - self.opacity_floor)
            + self.opacity_floor
        )
        if self.ste_threshold is not None:
            opacity = (
                (opacity >= self.ste_threshold).float() - opacity
            ).detach() + opacity

        # --- Gamma rescale: shrink triangles to keep opacity integral -------
        # invariant under gamma annealing.  Must match the training-time
        # rescaling so that the learned geometry renders sharply.
        gamma = self.gamma
        render_vertex = vertex
        if self.gamma_rescale:
            beta = 1.0 / gamma
            rescale_ratio = 1.0 / np.sqrt(
                (2.0 ** beta) * beta * scipy.special.gamma(beta)
            )
            t_center = vertex.mean(dim=1, keepdim=True)
            render_vertex = (vertex - t_center) * rescale_ratio + t_center

        # --- Background colour ---------------------------------------------
        bg_color = _resolve_bg_color(self._bg_color_str, vertex.device)
        bg_depth = (
            native_cam.camera_center.view(1, 1, 3) - vertex
        ).norm(dim=-1).max().item()

        renderer = self._TriangleRenderer(
            native_cam,
            bg_depth=bg_depth,
            bg_color=bg_color,
            sh_degree=min(self.active_sh_degree, self.max_sh_degree),
            gamma=gamma,
            back_culling=self.back_culling,
            rich_info=mode == "train",
            sort_level=self.sort_level,
        )
        grad_ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
        with grad_ctx:
            rendering = renderer.render(render_vertex, shs, None, opacity)

        rendered_image = rendering["render"]
        return RenderOutput(
            rgb=rendered_image.permute(1, 2, 0).contiguous(),
            alpha=rendering.get("alpha_mask"),
            depth=rendering.get("depth"),
            normal=rendering.get("normal").permute(1, 2, 0).contiguous()
            if "normal" in rendering else None,
            radii=rendering.get("radii"),
            extras={
                "num_rendered": rendering.get("num_rendered"),
                "distortion": rendering.get("distortion"),
                "contrib_sum": rendering.get("contrib_sum"),
                "contrib_max": rendering.get("contrib_max"),
                "n_contribs": rendering.get("n_contribs"),
            },
        )

    def to_primitive(self) -> Any:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        from tribench.primitives.triangle import IndependentTriangle

        features = torch.cat((self._model.f_dc.detach(), self._model.f_rest.detach()), dim=-2)
        sh_coeffs = features.reshape(features.shape[0], -1)
        return IndependentTriangle(
            vertices=self._model.vertex.detach().clone(),
            opacity=torch.sigmoid(self._model.opacity.detach()).squeeze(),
            sh_coeffs=sh_coeffs,
            gamma=self.gamma,
            back_culling=self.back_culling,
        )


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return -1, path.name


def _checkpoint_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_BG_COLORS: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}


def _resolve_bg_color(name: str, device: torch.device) -> torch.Tensor:
    """Return a [3] background colour tensor from a colour name."""
    rgb = _BG_COLORS.get(name.lower())
    if rgb is None:
        raise ValueError(
            f"Unknown background colour {name!r}. "
            f"Available: {sorted(_BG_COLORS)}"
        )
    return torch.tensor(rgb, device=device)
