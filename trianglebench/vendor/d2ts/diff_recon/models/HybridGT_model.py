import math
import os
import numpy as np
import trimesh

import torch
import torch.nn.functional as F

from .Base_model import BaseModel
from .model_utils import get_color_tensor
from .raw_gaussian import RawGaussian
from .raw_triangle import RawTriangle
from ..renderer.hybrid_renderer import HybridRenderer
from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.logger import Logger


class HybridGTModel(BaseModel):
    def __init__(self, config: Config = None, logger: Logger = None, device: torch.device | int = None):
        super().__init__(config, logger, device)

        config = self.config
        self.tri_gamma = config.tri_gamma if config.tri_gamma is not None else 100.0
        self.gau_gamma = config.gau_gamma if config.gau_gamma is not None else 1.0
        self.back_culling = config.back_culling if config.back_culling is not None else False
        self.sort_level = config.sort_level if config.sort_level is not None else 2
        self.tri_sh_degree = config.tri_sh_degree if config.tri_sh_degree is not None else 0
        self.gau_sh_degree = config.gau_sh_degree if config.gau_sh_degree is not None else 0
        self.ambient_intensity = config.ambient_intensity if getattr(config, "ambient_intensity", None) is not None else 1.0
        self.light_color = config.light_color if getattr(config, "light_color", None) is not None else [1.0, 1.0, 1.0]
        self.light_dir = config.light_dir if getattr(config, "light_dir", None) is not None else [0.0, 0.0, -1.0]
        self.light_intensity = config.light_intensity if getattr(config, "light_intensity", None) is not None else 0.0

        self.register_buffer("tri_vertex", torch.empty((0, 3, 3), dtype=torch.float32))
        self.register_buffer("tri_shs", torch.empty((0,), dtype=torch.float32))
        self.register_buffer("tri_uv", torch.empty((0, 3, 2), dtype=torch.float32))
        self.register_buffer("tri_texture", torch.empty((0, 0, 3), dtype=torch.float32))
        self.register_buffer("tri_opacity", torch.empty((0, 1), dtype=torch.float32))

        self.register_buffer("gau_xyz", torch.empty((0, 3), dtype=torch.float32))
        self.register_buffer("gau_shs", torch.empty((0,), dtype=torch.float32))
        self.register_buffer("gau_opacity", torch.empty((0, 1), dtype=torch.float32))
        self.register_buffer("gau_scaling", torch.empty((0, 3), dtype=torch.float32))
        self.register_buffer("gau_rot", torch.empty((0, 4), dtype=torch.float32))

    def _load_raw_gaussian(self, gaussian: RawGaussian | str) -> RawGaussian:
        if isinstance(gaussian, RawGaussian):
            return gaussian
        if isinstance(gaussian, str):
            return RawGaussian(ply_path=gaussian)
        raise TypeError(f"Unsupported gaussian input type: {type(gaussian)}")

    def _load_raw_triangle(self, triangle: RawTriangle | str) -> RawTriangle:
        if isinstance(triangle, RawTriangle):
            return triangle
        if isinstance(triangle, str):
            ext = os.path.splitext(triangle)[1].lower()
            if ext in [".glb", ".gltf"]:
                return RawTriangle(glb_path=triangle)
            return RawTriangle(ply_path=triangle)
        raise TypeError(f"Unsupported triangle input type: {type(triangle)}")

    def load_gaussian(self, gaussian: RawGaussian | str) -> "HybridGTModel":
        raw_gaussian = self._load_raw_gaussian(gaussian)

        if len(raw_gaussian) == 0:
            self.gau_xyz = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            self.gau_shs = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.gau_opacity = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            self.gau_scaling = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            self.gau_rot = torch.empty((0, 4), dtype=torch.float32, device=self.device)
            self.gau_sh_degree = 0
            return self

        self.gau_xyz = torch.tensor(raw_gaussian.xyz, dtype=torch.float32, device=self.device)
        self.gau_shs = torch.tensor(raw_gaussian.shs, dtype=torch.float32, device=self.device).view(raw_gaussian.xyz.shape[0], -1, 3)
        self.gau_opacity = torch.sigmoid(torch.tensor(raw_gaussian.opacity, dtype=torch.float32, device=self.device))
        self.gau_scaling = torch.exp(torch.tensor(raw_gaussian.scale, dtype=torch.float32, device=self.device))
        self.gau_rot = F.normalize(torch.tensor(raw_gaussian.rot, dtype=torch.float32, device=self.device), dim=1)

        self.gau_sh_degree = max(int(math.sqrt(self.gau_shs.shape[1]) - 1), 0)
        return self

    def load_triangle(self, triangle: RawTriangle | str) -> "HybridGTModel":
        raw_triangle = self._load_raw_triangle(triangle)

        if len(raw_triangle) == 0:
            self.tri_vertex = torch.empty((0, 3, 3), dtype=torch.float32, device=self.device)
            self.tri_shs = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.tri_uv = torch.empty((0, 3, 2), dtype=torch.float32, device=self.device)
            self.tri_texture = torch.empty((0, 0, 3), dtype=torch.float32, device=self.device)
            self.tri_opacity = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            self.tri_sh_degree = 0
            return self

        tri_vertex = np.ascontiguousarray(raw_triangle.vertex)
        self.tri_vertex = torch.tensor(tri_vertex, dtype=torch.float32, device=self.device)
        if raw_triangle.hasTexture():
            texture = raw_triangle.texture[..., :3] if raw_triangle.texture.shape[-1] == 4 else raw_triangle.texture
            self.tri_uv = torch.tensor(np.ascontiguousarray(raw_triangle.uv), dtype=torch.float32, device=self.device)
            self.tri_texture = torch.tensor(np.ascontiguousarray(texture), dtype=torch.float32, device=self.device)
            self.tri_shs = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.tri_sh_degree = 0
        else:
            if raw_triangle.shs.ndim == 3:
                self.tri_shs = torch.tensor(np.ascontiguousarray(raw_triangle.shs), dtype=torch.float32, device=self.device).view(raw_triangle.vertex.shape[0], 3, -1, 3)
                self.tri_sh_degree = max(int(math.sqrt(self.tri_shs.shape[-2]) - 1), 0)
            else:
                self.tri_shs = torch.tensor(np.ascontiguousarray(raw_triangle.shs), dtype=torch.float32, device=self.device).view(raw_triangle.vertex.shape[0], -1, 3)
                self.tri_sh_degree = max(int(math.sqrt(self.tri_shs.shape[1]) - 1), 0)
            self.tri_uv = torch.empty((0, 3, 2), dtype=torch.float32, device=self.device)
            self.tri_texture = torch.empty((0, 0, 3), dtype=torch.float32, device=self.device)
        self.tri_opacity = torch.sigmoid(torch.tensor(np.ascontiguousarray(raw_triangle.opacity), dtype=torch.float32, device=self.device))

        return self

    def forward(
        self,
        camera: Camera,
        background: str = "black",
        tri_gamma: float = None,
        gau_gamma: float = None,
        back_culling: bool = None,
        sort_level: int = None,
        bg_depth: float = None,
        ambient_intensity: float = None,
        light_color: torch.Tensor = None,
        light_dir: torch.Tensor = None,
        light_intensity: float = None,
        debug: bool = False,
        rich_info: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.tri_vertex.numel() == 0 and self.gau_xyz.numel() == 0:
            raise ValueError("No geometry loaded. Call load_triangle and/or load_gaussian before forward.")

        tri_gamma = tri_gamma if tri_gamma is not None else self.tri_gamma
        gau_gamma = gau_gamma if gau_gamma is not None else self.gau_gamma
        back_culling = back_culling if back_culling is not None else self.back_culling
        sort_level = sort_level if sort_level is not None else self.sort_level
        ambient_intensity = ambient_intensity if ambient_intensity is not None else self.ambient_intensity
        light_color = light_color if light_color is not None else torch.tensor(self.light_color, dtype=torch.float32, device=camera.device)
        light_dir = light_dir if light_dir is not None else torch.tensor(self.light_dir, dtype=torch.float32, device=camera.device)
        light_intensity = light_intensity if light_intensity is not None else self.light_intensity

        tri_vertex = self.tri_vertex.to(camera.device) if self.tri_vertex.numel() > 0 else None
        tri_shs = self.tri_shs.to(camera.device) if self.tri_shs.numel() > 0 else None
        tri_uv = self.tri_uv.to(camera.device) if self.tri_uv.numel() > 0 else None
        tri_texture = self.tri_texture.to(camera.device) if self.tri_texture.numel() > 0 else None
        tri_opacity = self.tri_opacity.to(camera.device).view(-1) if self.tri_opacity.numel() > 0 else None

        gau_xyz = self.gau_xyz.to(camera.device) if self.gau_xyz.numel() > 0 else None
        gau_shs = self.gau_shs.to(camera.device) if self.gau_shs.numel() > 0 else None
        gau_opacity = self.gau_opacity.to(camera.device) if self.gau_opacity.numel() > 0 else None
        gau_scaling = self.gau_scaling.to(camera.device) if self.gau_scaling.numel() > 0 else None
        gau_rot = self.gau_rot.to(camera.device) if self.gau_rot.numel() > 0 else None

        if bg_depth is None:
            bg_depth_candidates = []
            if tri_vertex is not None:
                bg_depth_candidates.append((camera.camera_center.view(1, 1, 3) - tri_vertex).norm(dim=-1).max())
            if gau_xyz is not None:
                bg_depth_candidates.append((camera.camera_center.view(1, 3) - gau_xyz).norm(dim=-1).max())
            bg_depth = torch.stack(bg_depth_candidates).max().item()

        bg_color = get_color_tensor(background).to(camera.device)

        renderer = HybridRenderer(
            cam=camera,
            bg_depth=bg_depth,
            bg_color=bg_color,
            tri_sh_degree=self.tri_sh_degree,
            gau_sh_degree=self.gau_sh_degree,
            tri_gamma=tri_gamma,
            gau_gamma=gau_gamma,
            ambient_intensity=ambient_intensity,
            light_color=light_color,
            light_dir=light_dir,
            light_intensity=light_intensity,
            back_culling=back_culling,
            rich_info=rich_info,
            sort_level=sort_level,
            debug=debug,
        )

        use_tri_texture = tri_uv is not None and tri_texture is not None and tri_uv.numel() > 0 and tri_texture.numel() > 0

        return renderer.render(
            tri_vertex=tri_vertex,
            tri_shs=None if use_tri_texture else tri_shs,
            tri_color=None,
            tri_uv=tri_uv,
            tri_texture=tri_texture,
            tri_opacity=tri_opacity,
            gau_xyz=gau_xyz,
            gau_shs=gau_shs,
            gau_color=None,
            gau_opacity=gau_opacity,
            gau_scaling=gau_scaling,
            gau_rot=gau_rot,
        )