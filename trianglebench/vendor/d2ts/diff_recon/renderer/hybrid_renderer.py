import torch
from typing import Optional

from hybrid_rasterization import TriangleRasterizationSettings, TriangleRasterizer

from ..utils.camera import Camera


class HybridRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_depth: float = 0.0,
        bg_color: torch.Tensor = torch.Tensor([0, 0, 0]),
        tri_sh_degree: int = 0,
        gau_sh_degree: int = 0,
        tri_gamma: float = 1.0,
        gau_gamma: float = 1.0,
        ambient_intensity: float = 1.0,
        light_color: Optional[torch.Tensor] = None,
        light_dir: Optional[torch.Tensor] = None,
        light_intensity: float = 0.0,
        back_culling: bool = False,
        rich_info: bool = False,
        sort_level: int = 0,
        gaussian_prefiltered: bool = False,
        debug: bool = False,
    ):
        if light_color is None:
            light_color = torch.tensor([1.0, 1.0, 1.0], dtype=bg_color.dtype, device=cam.device)
        if light_dir is None:
            light_dir = torch.tensor([0.0, 0.0, -1.0], dtype=bg_color.dtype, device=cam.device)

        raster_settings = TriangleRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=cam.tan_fovx,
            tanfovy=cam.tan_fovy,
            viewmatrix=cam.world_view_transform.contiguous(),
            projmatrix=cam.projection_matrix.contiguous(),
            full_projmatrix=cam.full_proj_transform.contiguous(),
            campos=cam.camera_center.contiguous(),
            tri_sh_degree=tri_sh_degree,
            gau_sh_degree=gau_sh_degree,
            tri_gamma=tri_gamma,
            gau_gamma=gau_gamma,
            ambient_intensity=ambient_intensity,
            light_color=light_color.to(cam.device),
            light_dir=light_dir.to(cam.device),
            light_intensity=light_intensity,
            background_depth=bg_depth,
            background=bg_color.to(cam.device),
            back_culling=back_culling,
            rich_info=rich_info,
            sort_level=sort_level,
            gaussian_prefiltered=gaussian_prefiltered,
            debug=debug,
        )

        self.cam = cam
        self.rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    def render(
        self,
        tri_vertex: torch.Tensor = None,
        tri_shs: torch.Tensor = None,
        tri_color: torch.Tensor = None,
        tri_uv: torch.Tensor = None,
        tri_texture: torch.Tensor = None,
        tri_opacity: torch.Tensor = None,
        gau_xyz: torch.Tensor = None,
        gau_shs: torch.Tensor = None,
        gau_color: torch.Tensor = None,
        gau_opacity: torch.Tensor = None,
        gau_scaling: torch.Tensor = None,
        gau_rot: torch.Tensor = None,
        gau_cov3d_precomp: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        has_triangles = tri_vertex is not None and tri_vertex.numel() > 0
        has_gaussians = gau_xyz is not None and gau_xyz.numel() > 0

        if not has_triangles and not has_gaussians:
            raise ValueError("At least one of triangle inputs or gaussian inputs must be provided.")

        ref_tensor = tri_vertex if has_triangles else gau_xyz
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        if has_triangles:
            if tri_opacity is None or tri_opacity.numel() == 0:
                raise ValueError("tri_opacity is required when triangle inputs are provided.")
            has_tri_shs = tri_shs is not None and tri_shs.numel() > 0
            has_tri_color = tri_color is not None and tri_color.numel() > 0
            has_tri_texture = tri_uv is not None and tri_uv.numel() > 0 and tri_texture is not None and tri_texture.numel() > 0
            if int(has_tri_shs) + int(has_tri_color) + int(has_tri_texture) != 1:
                raise ValueError("Provide exactly one of tri_shs, tri_color, or tri_uv+tri_texture when triangle inputs are provided.")
        else:
            tri_vertex = torch.empty((0, 3, 3), device=device, dtype=dtype)
            tri_opacity = torch.empty((0,), device=device, dtype=dtype)
            tri_shs = None
            tri_color = torch.empty((0, 3), device=device, dtype=dtype)
            tri_uv = torch.empty((0, 3, 2), device=device, dtype=dtype)
            tri_texture = torch.empty((0, 0, 3), device=device, dtype=dtype)

        if has_gaussians:
            if gau_opacity is None or gau_opacity.numel() == 0:
                raise ValueError("gau_opacity is required when gaussian inputs are provided.")
            has_gau_shs = gau_shs is not None and gau_shs.numel() > 0
            has_gau_color = gau_color is not None and gau_color.numel() > 0
            if has_gau_shs == has_gau_color:
                raise ValueError("Provide exactly one of gau_shs or gau_color when gaussian inputs are provided.")

            has_cov = gau_cov3d_precomp is not None and gau_cov3d_precomp.numel() > 0
            has_scale_rot = (
                gau_scaling is not None
                and gau_scaling.numel() > 0
                and gau_rot is not None
                and gau_rot.numel() > 0
            )
            if has_cov == has_scale_rot:
                raise ValueError("Provide exactly one of gau_cov3d_precomp or gau_scaling+gau_rot when gaussian inputs are provided.")
        else:
            gau_xyz = torch.empty((0, 3), device=device, dtype=dtype)
            gau_opacity = torch.empty((0,), device=device, dtype=dtype)
            gau_shs = None
            gau_color = torch.empty((0, 3), device=device, dtype=dtype)
            gau_scaling = torch.empty((0, 3), device=device, dtype=dtype)
            gau_rot = torch.empty((0, 4), device=device, dtype=dtype)
            gau_cov3d_precomp = None

        grad_holder = torch.zeros((tri_vertex.shape[0]), device=device, dtype=dtype, requires_grad=True)

        if gau_cov3d_precomp is None:
            gau_cov3d_precomp = torch.empty((0, 6), device=device, dtype=dtype)

        output_tuple = self.rasterizer.forward(
            vertex=tri_vertex,
            grad_holder=grad_holder,
            opacity=tri_opacity,
            shs=tri_shs,
            feature=tri_color,
            tri_uv=tri_uv,
            tri_texture=tri_texture,
            means3D=gau_xyz,
            gaussian_opacity=gau_opacity,
            gaussian_shs=gau_shs,
            gaussian_colors=gau_color,
            gaussian_scales=gau_scaling,
            gaussian_rotations=gau_rot,
            gaussian_cov3D_precomp=gau_cov3d_precomp,
        )

        if self.rasterizer.raster_settings.rich_info:
            rendered_image, radii, depth, normal, distortion, contrib_sum, contrib_max, n_contribs, alpha_mask, num_rendered = output_tuple
            output_pkg = {
                "render": rendered_image,
                "radii": radii,
                "num_rendered": num_rendered,
                "grad_holder": grad_holder,
                "depth": depth,
                "normal": normal,
                "distortion": distortion,
                "contrib_sum": contrib_sum,
                "contrib_max": contrib_max,
                "n_contribs": n_contribs,
                "alpha_mask": alpha_mask,
            }
        else:
            rendered_image, radii = output_tuple
            output_pkg = {
                "render": rendered_image,
                "radii": radii,
                "grad_holder": grad_holder,
            }
        return output_pkg
