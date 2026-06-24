"""Adapter for DiffSoup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
        self._runtime_device: torch.device | None = None
        self._vertices: torch.Tensor | None = None
        self._faces: torch.Tensor | None = None
        self._features: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._color_mlp: nn.Module | None = None
        self._level: int | None = None
        self._feature_dim: int | None = None
        self._background_color_override: list[float] | None = None

    @property
    def name(self) -> str:
        return "diffsoup"

    @property
    def device(self) -> torch.device:
        if self._runtime_device is not None:
            return self._runtime_device
        if self._checkpoint is not None and torch.cuda.is_available():
            return torch.device("cuda")
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
        self._clear_runtime_state()

    def configure(self, **kwargs: Any) -> None:
        """Set DiffSoup render-time options."""
        for key, value in kwargs.items():
            if key == "bg_color":
                color_name = str(value).lower()
                if color_name in {"white", "1", "1.0"}:
                    self._background_color_override = [1.0, 1.0, 1.0]
                elif color_name in {"black", "0", "0.0"}:
                    self._background_color_override = [0.0, 0.0, 0.0]
                else:
                    raise ValueError("DiffSoupAdapter bg_color must be 'white' or 'black'.")
            else:
                raise KeyError(f"Unknown DiffSoupAdapter config key: {key!r}")

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
        if self._checkpoint is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        self._ensure_runtime()
        assert self._ds is not None
        assert self._runtime_device is not None
        assert self._vertices is not None
        assert self._faces is not None
        assert self._features is not None
        assert self._alpha is not None
        assert self._color_mlp is not None
        assert self._level is not None
        assert self._feature_dim is not None

        device = self._runtime_device
        height = int(cameras.height)
        width = int(cameras.width)
        mvp = self._build_mvp(cameras, device=device).contiguous()
        mvp_inv = torch.inverse(mvp).contiguous()
        background_color = self._background_color(device)
        use_background = bool(torch.any(background_color != 0))
        stochastic = mode != "eval"
        grad_ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

        with grad_ctx:
            clip_vertices = self._project_vertices(self._vertices, mvp)
            raster = self._ds.rasterize_multires_triangle_alpha(
                (height, width),
                clip_vertices,
                self._faces,
                level=self._level,
                alpha_src=self._alpha,
                stochastic=stochastic,
            )
            features = self._ds.multires_triangle_color(
                raster,
                level=self._level,
                feat=self._features,
            ).reshape(-1, height, width, self._feature_dim)
            features = torch.cat(
                [features, self._ds.encode_view_dir_sh2(raster, mvp_inv)],
                dim=-1,
            )
            visible = raster[..., -1] > 0
            color = self._color_mlp(features, mask=visible).reshape(
                -1, height, width, 3,
            )
            if use_background:
                mask = visible.unsqueeze(-1).to(color.dtype)
                color = (
                    mask * color
                    + background_color.view(1, 1, 1, 3) * (1.0 - mask)
                )

        alpha = visible.to(color.dtype)
        depth = torch.where(visible, raster[..., 2], torch.zeros_like(raster[..., 2]))

        if color.shape[0] == 1:
            rgb_out = color[0].contiguous()
            alpha_out = alpha[0].contiguous()
            depth_out = depth[0].contiguous()
        else:
            rgb_out = color.contiguous()
            alpha_out = alpha.contiguous()
            depth_out = depth.contiguous()

        return RenderOutput(
            rgb=rgb_out,
            alpha=alpha_out,
            depth=depth_out,
            extras={
                "background_color": background_color.detach().cpu().tolist(),
                "level": self._level,
                "feature_dim": self._feature_dim,
                "stochastic": stochastic,
            },
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

    def _clear_runtime_state(self) -> None:
        self._runtime_device = None
        self._vertices = None
        self._faces = None
        self._features = None
        self._alpha = None
        self._color_mlp = None
        self._level = None
        self._feature_dim = None

    def _ensure_runtime(self) -> None:
        if self._color_mlp is not None:
            return
        if self._checkpoint is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        if not torch.cuda.is_available():
            raise RuntimeError("DiffSoup rendering requires CUDA.")
        self._ensure_imports()
        assert self._ds is not None

        device = torch.device("cuda")
        ckpt = self._checkpoint
        color_state = ckpt.get("color_mlp")
        if not isinstance(color_state, dict):
            raise KeyError("DiffSoup checkpoint missing color_mlp state dict.")

        self._vertices = ckpt["V"].to(device=device, dtype=torch.float32).contiguous()
        self._faces = ckpt["F"].to(device=device, dtype=torch.int32).contiguous()
        self._features = ckpt["feat_acc"].to(device=device, dtype=torch.float32).contiguous()
        self._alpha = ckpt["alpha_acc"].to(device=device, dtype=torch.float32).contiguous()
        self._level = int(ckpt["Rmax"])
        self._feature_dim = int(ckpt["feat_dim"])
        self._color_mlp = self._build_color_mlp(color_state, device)
        self._runtime_device = device

    def _build_color_mlp(self, state_dict: dict[str, Any], device: torch.device) -> nn.Module:
        assert self._ds is not None
        weight_items = []
        for key, value in state_dict.items():
            if key.endswith(".weight") and isinstance(value, torch.Tensor):
                try:
                    layer_idx = int(key.split(".")[1])
                except (IndexError, ValueError):
                    layer_idx = len(weight_items)
                weight_items.append((layer_idx, value))
        weight_items.sort(key=lambda item: item[0])
        if len(weight_items) < 2:
            raise KeyError("DiffSoup color_mlp state dict does not contain enough layers.")

        weights = [value for _, value in weight_items]
        input_dim = int(weights[0].shape[1])
        hidden_dim = int(weights[0].shape[0])
        n_layers = len(weights) - 1
        output_dim = int(weights[-1].shape[0])
        color_mlp = self._ds.ColorMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=output_dim,
        ).to(device)
        color_mlp.load_state_dict(state_dict)
        color_mlp.eval()
        return color_mlp

    def _background_color(self, device: torch.device) -> torch.Tensor:
        if self._background_color_override is not None:
            color = self._background_color_override
        else:
            ckpt = self._checkpoint or {}
            dataset_type = str(ckpt.get("dataset_type", "")).lower()
            white_background = dataset_type == "dtu" or "flip_z" not in ckpt
            color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
        return torch.tensor(color, dtype=torch.float32, device=device)

    def _build_mvp(self, cameras: CameraBatch, *, device: torch.device) -> torch.Tensor:
        ckpt = self._checkpoint or {}
        viewmats = cameras.viewmats.to(device=device, dtype=torch.float32)
        Ks = cameras.Ks.to(device=device, dtype=torch.float32)
        if Ks.shape[0] == 1 and viewmats.shape[0] > 1:
            Ks = Ks.expand(viewmats.shape[0], -1, -1)

        synthetic = "flip_z" not in ckpt
        if synthetic:
            viewmats = self._synthetic_viewmat_to_native(viewmats)
            z_near, z_far = 0.1, 10.0
        else:
            z_near = self._near_plane(cameras)
            z_far = float(cameras.far)

        projections = torch.stack(
            [
                self._opengl_projection_from_K(
                    Ks[i],
                    cameras.height,
                    cameras.width,
                    z_near,
                    z_far,
                )
                for i in range(viewmats.shape[0])
            ],
            dim=0,
        )

        flip_z = bool(ckpt.get("flip_z", False))
        if flip_z:
            zf = torch.eye(4, dtype=viewmats.dtype, device=device)
            zf[2, 2] = -1.0
            viewmats = zf.unsqueeze(0) @ viewmats
        return projections @ viewmats

    def _near_plane(self, cameras: CameraBatch) -> float:
        ckpt = self._checkpoint or {}
        dataset_type = str(ckpt.get("dataset_type", "")).lower()
        metadata = cameras.metadata or {}
        split = str(metadata.get("split", "")).lower()
        colmap_like = dataset_type in {"colmap", "mip360", "mipnerf360", "dtu"}
        if colmap_like and split in {"test", "val", "validation"}:
            return 0.5
        return float(cameras.near)

    @staticmethod
    def _opengl_projection_from_K(
        K: torch.Tensor,
        height: int,
        width: int,
        z_near: float,
        z_far: float,
    ) -> torch.Tensor:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        projection = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
        projection[0, 0] = 2.0 * fx / float(width)
        projection[1, 1] = 2.0 * fy / float(height)
        projection[0, 2] = 1.0 - 2.0 * (cx / float(width))
        projection[1, 2] = 2.0 * (cy / float(height)) - 1.0
        projection[2, 2] = (z_far + z_near) / (z_near - z_far)
        projection[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
        projection[3, 2] = -1.0
        return projection

    @staticmethod
    def _project_vertices(verts: torch.Tensor, mvp: torch.Tensor) -> torch.Tensor:
        verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
        return torch.einsum("bij,nj->bni", mvp, verts_h).contiguous()

    @staticmethod
    def _synthetic_viewmat_to_native(viewmats: torch.Tensor) -> torch.Tensor:
        coordinate_flip = torch.eye(4, dtype=viewmats.dtype, device=viewmats.device)
        coordinate_flip[1, 1] = -1.0
        coordinate_flip[2, 2] = -1.0
        return coordinate_flip.unsqueeze(0) @ viewmats
