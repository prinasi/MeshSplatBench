import torch

from trianglebench.vendor._cmod.d2ts_rasterization import TriangleRasterizationSettings, TriangleRasterizer

from ..utils.camera import Camera


class TriangleRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_depth: float = 0.0,
        bg_color: torch.Tensor = torch.Tensor([0, 0, 0]),
        sh_degree: int = 0,
        gamma: float = 1.0,
        back_culling: bool = False,
        rich_info: bool = False,
        sort_level: int = 0,
        debug: bool = False,
    ):
        raster_settings = TriangleRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=cam.tan_fovx,
            tanfovy=cam.tan_fovy,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.projection_matrix,
            campos=cam.camera_center,
            sh_degree=sh_degree,
            gamma=gamma,
            background_depth=bg_depth,
            background=bg_color.to(cam.device),
            back_culling=back_culling,
            rich_info=rich_info,
            sort_level=sort_level,
            debug=debug,
        )

        self.cam = cam
        self.rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    def render(
        self,
        vertex: torch.Tensor,
        shs: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Create placeholder tensor to store gradients of the 2D (screen-space) center of each triangle.
        grad_holder = torch.zeros((vertex.shape[0]), device=vertex.device, dtype=vertex.dtype, requires_grad=True)

        output_tuple = self.rasterizer.forward(
            vertex=vertex,
            grad_holder=grad_holder,
            opacity=opacity,
            shs=shs,
            feature=color,
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
            }
        return output_pkg
