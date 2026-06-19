"""Adapter for Triangle Splatting.

Connects to the triangle-splatting repository's native CUDA renderer
(diff-triangle-rasterization) for benchmarking and evaluation.

Requires:
    - TriBench installed with its bundled CUDA extensions built.
"""

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
import torch.nn as nn

from tribench.core.cameras import CameraBatch
from tribench.core.registry import register
from tribench.core.stats import compute_triangle_stats
from tribench.renderers.backends import get_backend
from tribench.renderers.base import RendererAdapter, RenderOutput


# Default path to the triangle-splatting repository
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent.parent / "triangle-splatting"


@register("triangle-splatting")
class TriangleSplattingAdapter(RendererAdapter):
    """Renderer adapter for Triangle Splatting.

    Triangle Splatting uses:
    - Triangle points: [T, 3, 3] (T triangles, 3 vertices each, xyz)
    - Per-triangle sigma for smooth convex coverage
    - Per-triangle opacity (sigmoid-activated)
    - SH coefficients for view-dependent color
    - Per-triangle mask for soft pruning

    Integration: Preserves native CUDA renderer (diff-triangle-rasterization).
    Exposes sigma, primitive visibility, max blending, and density statistics.

    Repository: triangle-splatting/
    """

    def __init__(self, repo_root: str | Path | None = None):
        """Initialize the adapter.

        Args:
            repo_root: Path to the triangle-splatting repository root.
                       Defaults to ../triangle-splatting relative to this file.
        """
        self._backend = get_backend("triangle-splatting")
        self.repo_root = self._backend.repo_root(repo_root or _DEFAULT_REPO_ROOT)
        self._initialized = False
        self._model = None
        self._cameras: list | None = None
        self._dataset_path: str | None = None
        self._checkpoint_path: str | None = None
        self._background_color = [0.0, 0.0, 0.0]

        # Lazy-imported modules
        self._TriangleModel = None
        self._ts_render = None
        self._getWorld2View2 = None
        self._getProjectionMatrix = None

    @property
    def name(self) -> str:
        return "triangle-splatting"

    @property
    def device(self) -> torch.device:
        if self._model is not None:
            return self._model._triangles_points.device
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        return self._backend.status(self.repo_root)

    def _ensure_imports(self) -> None:
        """Import bundled triangle-splatting modules."""
        if self._initialized:
            return

        try:
            from tribench.vendor.triangle_splatting.scene.triangle_model import TriangleModel
            from tribench.vendor.triangle_splatting.triangle_renderer import render as ts_render
            from tribench.vendor.triangle_splatting.utils.graphics_utils import (
                getProjectionMatrix,
                getWorld2View2,
            )
        except ImportError as exc:
            raise ImportError(
                "Cannot import bundled triangle-splatting renderer. Reinstall "
                "TriBench with its bundled CUDA extensions built."
            ) from exc

        self._TriangleModel = TriangleModel
        self._ts_render = ts_render
        self._getWorld2View2 = getWorld2View2
        self._getProjectionMatrix = getProjectionMatrix
        self._initialized = True

    # ---- Checkpoint loading ----

    def _load_background_color(self, checkpoint_path: Path) -> list[float]:
        """Infer eval background from TriBench metadata or native cfg_args."""
        search_dirs = [
            checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        ]
        search_dirs.extend(search_dirs[0].parents)

        for directory in search_dirs:
            metadata_path = directory / "tribench_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                color = metadata.get("background_color")
                if color is not None:
                    return [float(v) for v in color]
                if metadata.get("white_background"):
                    return [1.0, 1.0, 1.0]
                return [0.0, 0.0, 0.0]

            cfg_path = directory / "cfg_args"
            if cfg_path.exists():
                match = re.search(
                    r"white_background=(True|False)",
                    cfg_path.read_text(encoding="utf-8"),
                )
                if match:
                    return [1.0, 1.0, 1.0] if match.group(1) == "True" else [0.0, 0.0, 0.0]

        return [0.0, 0.0, 0.0]

    def load_checkpoint(self, path: str) -> None:
        """Load a Triangle Splatting checkpoint.

        Accepts either:
        - A directory containing ``point_cloud_state_dict.pt``
        - A direct path to a ``.pt`` file

        Args:
            path: Path to checkpoint directory or file.
        """
        self._ensure_imports()

        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            state_path = ckpt_path / "point_cloud_state_dict.pt"
        else:
            state_path = ckpt_path

        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {state_path}")

        self._checkpoint_path = str(state_path)
        self._background_color = self._load_background_color(ckpt_path)

        # Load state dict
        state_dict = torch.load(str(state_path), map_location="cpu", weights_only=False)

        # Infer SH degree from features_rest shape
        features_rest = state_dict["features_rest"]
        num_rest_coeffs = features_rest.shape[1]  # e.g. 15 for degree 3
        max_sh_degree = int(math.sqrt(num_rest_coeffs + 1)) - 1

        # Create model with inferred SH degree (__init__ just sets empty tensors)
        model = self._TriangleModel(max_sh_degree)

        # Load all parameters from checkpoint onto CUDA
        model._triangles_points = (
            state_dict["triangles_points"].to("cuda").to(torch.float32)
            .detach().clone().requires_grad_(True)
        )
        model._sigma = (
            state_dict["sigma"].to("cuda").to(torch.float32)
            .detach().clone().requires_grad_(True)
        )
        model._features_dc = (
            state_dict["features_dc"].to("cuda").to(torch.float32)
            .detach().clone().requires_grad_(True)
        )
        model._features_rest = (
            state_dict["features_rest"].to("cuda").to(torch.float32)
            .detach().clone().requires_grad_(True)
        )
        model._opacity = (
            state_dict["opacity"].to("cuda").to(torch.float32)
            .detach().clone().requires_grad_(True)
        )
        model.active_sh_degree = state_dict["active_sh_degree"]

        # Mask is not saved in checkpoint — default to all-ones
        N = model._triangles_points.size(0)
        model._mask = nn.Parameter(
            torch.ones((N, 1), device="cuda").requires_grad_(True)
        )

        # Compute indexing buffers (always 3 points per triangle)
        num_per_tri = torch.full((N,), 3, dtype=torch.int, device="cuda")
        cumsum = torch.cumsum(
            torch.nn.functional.pad(num_per_tri, (1, 0), value=0), 0, dtype=torch.int
        )[:-1]
        model._num_points_per_triangle = num_per_tri
        model._cumsum_of_points_per_triangle = cumsum
        model._number_of_points = N
        model.nb_points = 3

        # Initialize tracking buffers (used during training, harmless for eval)
        model.max_radii2D = torch.zeros(N, device="cuda")
        model.max_density_factor = torch.zeros(N, device="cuda")
        model.max_scaling = torch.zeros(N, device="cuda")
        model.denom = torch.zeros(N, 1, device="cuda")
        model.triangle_area = torch.zeros(N, device="cuda")
        model.image_size = torch.zeros(N, device="cuda")
        model.importance_score = torch.zeros(N, device="cuda")
        model.spatial_lr_scale = 0.0

        # Create a placeholder optimizer (not used during eval)
        model.optimizer = torch.optim.Adam([
            {"params": [model._features_dc], "lr": 1e-5, "name": "f_dc"},
            {"params": [model._features_rest], "lr": 5e-7, "name": "f_rest"},
            {"params": [model._opacity], "lr": 1e-5, "name": "opacity"},
            {"params": [model._triangles_points], "lr": 1e-5, "name": "triangles_points"},
            {"params": [model._sigma], "lr": 1e-5, "name": "sigma"},
            {"params": [model._mask], "lr": 1e-5, "name": "mask"},
        ], lr=0.0, eps=1e-15)

        self._model = model
        print(f"[TriBench] Loaded {N} triangles from {state_path}")

    # ---- Scene loading ----

    def load_scene(
        self,
        dataset_path: str,
        split: str = "test",
        *,
        image_dir: str = "images",
        resolution: int = 1,
        eval_every: int = 8,
    ) -> None:
        """Load a scene's cameras from a dataset directory.

        Supports COLMAP and Blender (NeRF Synthetic) formats.

        Args:
            dataset_path: Path to the dataset root directory.
            split: Dataset split ('train' or 'test').
            image_dir: Image folder inside the dataset root.
            resolution: Native triangle-splatting resolution argument.
            eval_every: Holdout stride for train/test splits.
        """
        self._ensure_imports()

        dataset_path = Path(dataset_path)
        self._dataset_path = str(dataset_path)

        from tribench.vendor.triangle_splatting.scene.colmap_loader import (
            read_extrinsics_binary,
            read_extrinsics_text,
            read_intrinsics_binary,
            read_intrinsics_text,
        )
        from tribench.vendor.triangle_splatting.scene.dataset_readers import (
            readColmapCameras,
            readNerfSyntheticInfo,
        )
        from tribench.vendor.triangle_splatting.utils.camera_utils import (
            cameraList_from_camInfos,
        )

        sparse_dir = dataset_path / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = dataset_path / "sparse"

        if sparse_dir.exists():
            # COLMAP format
            try:
                cameras_extrinsic = read_extrinsics_binary(
                    str(sparse_dir / "images.bin")
                )
                cameras_intrinsic = read_intrinsics_binary(
                    str(sparse_dir / "cameras.bin")
                )
            except Exception:
                cameras_extrinsic = read_extrinsics_text(
                    str(sparse_dir / "images.txt")
                )
                cameras_intrinsic = read_intrinsics_text(
                    str(sparse_dir / "cameras.txt")
                )

            cam_infos_unsorted = readColmapCameras(
                cam_extrinsics=cameras_extrinsic,
                cam_intrinsics=cameras_intrinsic,
                images_folder=str(dataset_path / image_dir),
            )
            # Sort by image name for deterministic ordering
            cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

            # Split into train/test (every 8th for test)
            if split == "test":
                cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_every == 0]
            elif split == "train":
                cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_every != 0]

            # Build Camera objects
            model_args = SimpleNamespace(data_device="cuda", resolution=resolution)
            self._cameras = cameraList_from_camInfos(cam_infos, 1.0, model_args)

        elif (dataset_path / "transforms_test.json").exists():
            # Blender format
            transform_file = dataset_path / f"transforms_{split}.json"
            if not transform_file.exists():
                transform_file = dataset_path / "transforms_test.json"

            scene_info = readNerfSyntheticInfo(
                str(dataset_path),
                self._background_color == [1.0, 1.0, 1.0],
                True,
            )
            self._cameras = scene_info.test_cameras if split == "test" else scene_info.train_cameras

        else:
            raise FileNotFoundError(
                f"Cannot detect scene type at {dataset_path}. "
                f"Expected sparse/ (COLMAP) or transforms_test.json (Blender)."
            )

        print(f"[TriBench] Loaded {len(self._cameras)} cameras for split '{split}'")

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
    ) -> Path:
        """Export a TSDF-fused mesh following the native DTU mesh pipeline."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        self._ensure_imports()
        self.load_scene(
            dataset_path,
            split=split,
            image_dir=image_dir,
            resolution=resolution,
            eval_every=eval_every,
        )
        if not self._cameras:
            raise RuntimeError(f"No cameras loaded for mesh export from {dataset_path!r}.")

        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Triangle-splatting TSDF mesh export requires open3d.") from exc

        from tribench.vendor.triangle_splatting.utils.mesh_utils import (
            GaussianExtractor,
            post_process_mesh,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        previous_sh_degree = self._model.active_sh_degree
        self._model.active_sh_degree = 0
        pipe = SimpleNamespace(
            debug=False,
            convert_SHs_python=False,
            depth_ratio=float(depth_ratio),
        )
        extractor = GaussianExtractor(
            self._model,
            self._ts_render,
            pipe,
            bg_color=self._background_color,
        )
        try:
            extractor.reconstruction(self._cameras)
            mesh = extractor.extract_mesh_bounded(
                voxel_size=float(voxel_size),
                sdf_trunc=float(sdf_trunc),
                depth_trunc=float(depth_trunc),
            )
            if int(num_cluster) > 0:
                mesh = post_process_mesh(mesh, cluster_to_keep=int(num_cluster))
            o3d.io.write_triangle_mesh(str(path), mesh)
        finally:
            self._model.active_sh_degree = previous_sh_degree

        return path

    # ---- Model statistics ----

    def model_stats(self) -> dict[str, Any]:
        """Compute model statistics from the loaded checkpoint."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        model = self._model
        vertices = model._triangles_points.detach().cpu()
        sigma = model.get_sigma.detach().cpu().squeeze()
        opacity = model.get_opacity.detach().cpu().squeeze()

        # Triangle area and opacity statistics
        tri_stats = compute_triangle_stats(vertices, opacity)

        # Trainable parameter count
        num_params = sum(p.numel() for p in [
            model._triangles_points, model._sigma,
            model._features_dc, model._features_rest,
            model._opacity, model._mask,
        ])

        # Checkpoint file size
        ckpt_size_mb = 0.0
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            ckpt_size_mb = os.path.getsize(self._checkpoint_path) / (1024 * 1024)

        # Sigma statistics for backend_extra
        sigma_np = sigma.numpy()

        stats = {
            "method": "triangle-splatting",
            "scene": self._dataset_path or "",
            "primitive_count": vertices.shape[0],
            "vertex_count": vertices.shape[0] * 3,
            "face_count": vertices.shape[0],
            "trainable_param_count": num_params,
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "triangle_area_mean": tri_stats.get("triangle_area_mean", 0.0),
            "triangle_area_median": tri_stats.get("triangle_area_median", 0.0),
            "triangle_area_p95": tri_stats.get("triangle_area_p95", 0.0),
            "opacity_mean": tri_stats.get("opacity_mean", 0.0),
            "opacity_median": tri_stats.get("opacity_median", 0.0),
            "opacity_histogram": tri_stats.get("opacity_histogram", []),
            "backend_extra": {
                "sigma_mean": float(np.mean(sigma_np)),
                "sigma_std": float(np.std(sigma_np)),
                "sigma_min": float(np.min(sigma_np)),
                "sigma_max": float(np.max(sigma_np)),
                "max_sh_degree": model.max_sh_degree,
                "active_sh_degree": model.active_sh_degree,
            },
        }
        return stats

    # ---- Rendering ----

    def _build_native_camera(self, cameras: CameraBatch, idx: int = 0):
        """Build a triangle-splatting native camera from CameraBatch.

        Uses the native getWorld2View2 and getProjectionMatrix utilities
        to ensure exact matrix convention compatibility.

        Args:
            cameras: CameraBatch from TriBench.
            idx: Camera index in the batch.

        Returns:
            SimpleNamespace with all camera fields expected by ts render().
        """
        viewmat = cameras.viewmats[idx].cpu().numpy()  # [4, 4] world-to-camera

        # Decompose viewmat into R and T
        # viewmat = [R.T | T; 0 0 0 1], so R = viewmat[:3,:3].T
        R = viewmat[:3, :3].T  # [3, 3] rotation (column-major)
        T = viewmat[:3, 3]     # [3] translation

        # Compute FoV from intrinsics
        K = cameras.Ks[idx]
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        fovX = 2.0 * math.atan(cameras.width / (2.0 * fx))
        fovY = 2.0 * math.atan(cameras.height / (2.0 * fy))

        # Use native utilities for exact convention match
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
        """Render using the native triangle-splatting CUDA renderer.

        Args:
            cameras: CameraBatch with camera parameters.
            mode: 'eval' for no gradients, 'train' for gradient computation.

        Returns:
            RenderOutput with rendered RGB image and auxiliary outputs.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        self._ensure_imports()

        native_cam = self._build_native_camera(cameras, idx=0)

        # Pipeline config: SH evaluated in CUDA (not Python), no debug
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=0.0)
        bg_color = torch.tensor(
            self._background_color, dtype=torch.float32, device="cuda"
        )

        grad_ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

        with grad_ctx:
            rendering = self._ts_render(
                native_cam, self._model, pipe, bg_color
            )

        # Extract outputs
        rendered_image = rendering["render"]          # [3, H, W]
        radii = rendering["radii"]                     # [N]
        visibility = rendering["visibility_filter"]    # [N] bool
        rend_alpha = rendering["rend_alpha"]           # [1, H, W]
        rend_normal = rendering["rend_normal"]         # [3, H, W] world-space
        surf_depth = rendering["surf_depth"]           # [1, H, W]

        return RenderOutput(
            rgb=rendered_image.permute(1, 2, 0).contiguous(),   # [H, W, 3]
            alpha=rend_alpha.squeeze(0),                         # [H, W]
            depth=surf_depth.squeeze(0),                         # [H, W]
            normal=rend_normal.permute(1, 2, 0).contiguous(),   # [H, W, 3]
            radii=radii,
            visibility=visibility,
            extras={
                "scaling": rendering.get("scaling"),
                "density_factor": rendering.get("density_factor"),
                "max_blending": rendering.get("max_blending"),
                "rend_dist": rendering.get("rend_dist"),
                "surf_normal": rendering.get("surf_normal"),
            },
        )

    # ---- Primitive extraction ----

    def to_primitive(self) -> Any:
        """Extract the model as a ConvexTriangle primitive.

        Returns:
            ConvexTriangle with the model's geometry, sigma, opacity, and SH.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        from tribench.primitives.convex_triangle import ConvexTriangle

        model = self._model
        vertices = model._triangles_points.detach().clone()   # [N, 3, 3]
        points = vertices.mean(dim=1)                          # [N, 3] centroids
        sigma = model.get_sigma.detach().squeeze()             # [N] activated
        opacity = model.get_opacity.detach().squeeze()         # [N] sigmoid
        features = model.get_features.detach()                 # [N, (deg+1)^2, 3]
        # Flatten SH coefficients: [N, C] where C = (deg+1)^2 * 3
        sh_coeffs = features.reshape(features.shape[0], -1)

        return ConvexTriangle(
            points=points,
            vertices=vertices,
            sigma=sigma,
            opacity=opacity,
            sh_coeffs=sh_coeffs,
        )
