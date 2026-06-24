"""Adapter for MeshSplatting."""

from __future__ import annotations

import json
import math
import os
import re
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
        self._checkpoint_dir: Path | None = None
        self._model_path: Path | None = None
        self._loaded_iteration: int | None = None
        self._dataset_path: str | None = None
        self._background_color = [0.0, 0.0, 0.0]
        self._background_color_override: list[float] | None = None
        self._render_scaling = 4

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

    def configure(self, **kwargs: Any) -> None:
        """Set render-time options used by TriBench evaluation.

        Accepted keys:
            render_scaling (int): MeshSplatting triangle scaling used by the
                native render.py script. Defaults to 4.
            scaling (int): Alias for render_scaling.
            bg_color (str): Background colour name ("white"/"black").
        """
        for key, value in kwargs.items():
            if key in {"render_scaling", "scaling"}:
                self._render_scaling = int(value)
                if self._model is not None:
                    self._model.scaling = self._render_scaling
            elif key == "bg_color":
                color_name = str(value).lower()
                if color_name in {"white", "1", "1.0"}:
                    self._background_color = [1.0, 1.0, 1.0]
                elif color_name in {"black", "0", "0.0"}:
                    self._background_color = [0.0, 0.0, 0.0]
                else:
                    raise ValueError(
                        "MeshSplattingAdapter bg_color must be 'white' or 'black'."
                    )
                self._background_color_override = list(self._background_color)
            else:
                raise KeyError(f"Unknown MeshSplattingAdapter config key: {key!r}")

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
        model.scaling = self._render_scaling

        self._model = model
        self._checkpoint_path = str(state_path)
        self._checkpoint_dir = ckpt_dir
        self._model_path, self._loaded_iteration = self._infer_native_run_checkpoint(ckpt_dir)
        if self._background_color_override is None:
            self._background_color = self._load_background_color(ckpt_dir)
        else:
            self._background_color = list(self._background_color_override)

    def _infer_native_run_checkpoint(self, ckpt_dir: Path) -> tuple[Path | None, int | None]:
        match = re.fullmatch(r"iteration_(\d+)", ckpt_dir.name)
        if match and ckpt_dir.parent.name == "point_cloud":
            return ckpt_dir.parent.parent, int(match.group(1))
        return None, None

    def _load_background_color(self, ckpt_dir: Path) -> list[float]:
        metadata_path = ckpt_dir / "tribench_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                color = metadata.get("background_color")
                if isinstance(color, list) and len(color) == 3:
                    return [float(v) for v in color]
                if metadata.get("white_background"):
                    return [1.0, 1.0, 1.0]
            except (OSError, ValueError, TypeError):
                pass

        cfg_args = ckpt_dir.parent.parent / "cfg_args"
        if cfg_args.exists():
            try:
                match = re.search(
                    r"white_background=(True|False)",
                    cfg_args.read_text(encoding="utf-8"),
                )
                if match and match.group(1) == "True":
                    return [1.0, 1.0, 1.0]
            except OSError:
                pass
        return [0.0, 0.0, 0.0]

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
        bg_color = torch.tensor(self._background_color, dtype=torch.float32, device="cuda")
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
                "background_color": list(self._background_color),
                "scaling": rendering.get("scaling"),
                "max_blending": rendering.get("max_blending"),
                "vertex_rendered": rendering.get("vertex_rendered"),
                "triangle_was_rendered": rendering.get("triangle_was_rendered"),
                "surf_normal": rendering.get("surf_normal"),
            },
        )

    def export_mesh(
        self,
        path: str | Path,
        *,
        dataset_path: str,
        split: str = "train",
        image_dir: str = "images",
        resolution: int = 1,
        eval_every: int = 8,
        voxel_size: float = 0.004,
        sdf_trunc: float = 0.016,
        depth_trunc: float = 3.0,
        num_cluster: int = 1,
        depth_ratio: float = 1.0,
        eval_split: bool = False,
        render_scaling: int = 1,
    ) -> Path:
        """Export a TSDF-fused mesh following the native MeshSplatting mesh.py path."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        if self._model_path is None or self._loaded_iteration is None:
            raise RuntimeError(
                "Native MeshSplatting mesh export requires a checkpoint directory "
                "like point_cloud/iteration_<N>."
            )

        self._ensure_imports()
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("MeshSplatting TSDF mesh export requires open3d.") from exc

        from tribench.vendor.mesh_splatting.scene import Scene
        from tribench.vendor.mesh_splatting.utils.mesh_utils import (
            GaussianExtractor,
            post_process_mesh,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = path.with_name(path.name.replace("_post.ply", ".ply")) if path.name.endswith("_post.ply") else path

        native_args = self._native_scene_args(
            dataset_path=dataset_path,
            image_dir=image_dir,
            resolution=resolution,
            eval_split=eval_split,
        )
        triangles = self._TriangleModel(native_args.sh_degree)
        triangles.scaling = int(render_scaling)
        scene = Scene(
            native_args,
            triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=self._loaded_iteration,
            shuffle=False,
        )

        previous_sh_degree = triangles.active_sh_degree
        previous_scaling = triangles.scaling
        triangles.active_sh_degree = 0
        triangles.scaling = int(render_scaling)
        pipe = SimpleNamespace(
            debug=False,
            convert_SHs_python=False,
            compute_cov3D_python=False,
            depth_ratio=float(depth_ratio),
        )
        bg_color = [1.0, 1.0, 1.0] if native_args.white_background else [0.0, 0.0, 0.0]
        extractor = GaussianExtractor(triangles, self._render, pipe, bg_color=bg_color)
        try:
            # Match native mesh.py: it always reconstructs from getTrainCameras().
            # With eval_split=False, those "train" cameras are the full DTU image set.
            extractor.reconstruction(scene.getTrainCameras())
            mesh = extractor.extract_mesh_bounded(
                voxel_size=float(voxel_size),
                sdf_trunc=float(sdf_trunc),
                depth_trunc=float(depth_trunc),
            )
            o3d.io.write_triangle_mesh(str(raw_path), mesh)
            mesh_post = post_process_mesh(mesh, cluster_to_keep=int(num_cluster))
            o3d.io.write_triangle_mesh(str(path), mesh_post)
            self._write_mesh_export_metadata(
                path,
                eval_split=eval_split,
                render_scaling=render_scaling,
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
                num_cluster=num_cluster,
                depth_ratio=depth_ratio,
            )
        finally:
            triangles.active_sh_degree = previous_sh_degree
            triangles.scaling = previous_scaling

        return path

    def _write_mesh_export_metadata(
        self,
        path: Path,
        *,
        eval_split: bool,
        render_scaling: int,
        voxel_size: float,
        sdf_trunc: float,
        depth_trunc: float,
        num_cluster: int,
        depth_ratio: float,
    ) -> None:
        metadata = {
            "tribench_mesh_export": {
                "method": "mesh-splatting",
                "pipeline": "native_tsdf",
                "version": 1,
                "checkpoint": str(self._checkpoint_dir) if self._checkpoint_dir is not None else None,
                "model_path": str(self._model_path) if self._model_path is not None else None,
                "iteration": self._loaded_iteration,
                "eval_split": bool(eval_split),
                "render_scaling": int(render_scaling),
                "voxel_size": float(voxel_size),
                "sdf_trunc": float(sdf_trunc),
                "depth_trunc": float(depth_trunc),
                "num_cluster": int(num_cluster),
                "depth_ratio": float(depth_ratio),
            }
        }
        metadata_path = path.with_suffix(path.suffix + ".tribench.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _native_scene_args(
        self,
        *,
        dataset_path: str,
        image_dir: str,
        resolution: int,
        eval_split: bool,
    ) -> SimpleNamespace:
        cfg = self._read_native_cfg_args()
        white_background = bool(getattr(cfg, "white_background", False))
        data_device = str(getattr(cfg, "data_device", "cuda"))
        sh_degree = int(getattr(cfg, "sh_degree", 3))
        return SimpleNamespace(
            sh_degree=sh_degree,
            source_path=os.path.abspath(str(dataset_path)),
            model_path=str(self._model_path),
            images=str(image_dir),
            resolution=int(resolution),
            white_background=white_background,
            data_device=data_device,
            eval=bool(eval_split),
        )

    def _read_native_cfg_args(self) -> SimpleNamespace:
        if self._model_path is None:
            return SimpleNamespace()
        cfg_args = self._model_path / "cfg_args"
        if not cfg_args.exists():
            return SimpleNamespace()
        try:
            namespace = {"Namespace": SimpleNamespace}
            parsed = eval(cfg_args.read_text(encoding="utf-8"), namespace)
            return parsed if isinstance(parsed, SimpleNamespace) else SimpleNamespace()
        except Exception:
            return SimpleNamespace()

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
