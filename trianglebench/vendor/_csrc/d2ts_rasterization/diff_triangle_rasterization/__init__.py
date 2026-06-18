from typing import NamedTuple
import torch.nn as nn
import torch
from typing import Callable, Tuple

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def debug_run(func: Callable, *args, debug: bool = False):
    if debug:
        func_name = func.__name__
        cpu_args = cpu_deep_copy_tuple(args)
        try:
            return func(*args)
        except Exception as ex:
            torch.save(cpu_args, f"snapshot_{func_name}.dump")
            print(f"\nAn error occured in {func_name}. Writing snapshot_{func_name}.dump for debugging.")
            raise ex
    else:
        return func(*args)


class TriangleRasterizationSettings(NamedTuple):
    # Camera settings
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    # Geometry settings
    sh_degree: int
    gamma: float
    background_depth: float
    background: torch.Tensor
    # Rasterization settings
    back_culling: bool
    rich_info: bool
    sort_level: int
    debug: bool


class _RasterizeTriangles(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertex: torch.Tensor,
        grad_holder: torch.Tensor,
        shs: torch.Tensor,
        feature: torch.Tensor,
        opacity: torch.Tensor,
        raster_settings: TriangleRasterizationSettings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.viewmatrix.contiguous(),
            raster_settings.projmatrix.contiguous(),
            raster_settings.campos.contiguous(),
            raster_settings.sh_degree,
            raster_settings.gamma,
            raster_settings.background_depth,
            raster_settings.background.contiguous(),
            vertex,
            shs,
            feature,
            opacity,
            raster_settings.back_culling,
            raster_settings.rich_info,
            raster_settings.sort_level,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        (
            num_rendered,
            out_feature,
            radii,
            depth,
            normal,
            distortion,
            contrib_sum,
            contrib_max,
            n_contribs,
            final_Ts,
            geometryBuffer,
            binningBuffer,
            imageBuffer,
        ) = debug_run(_C.rasterize_triangles, *args, debug=raster_settings.debug)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(vertex, shs, feature, opacity, radii, out_feature, depth, normal, distortion, geometryBuffer, binningBuffer, imageBuffer)

        alpha_mask = 1 - final_Ts
        if (n_contribs < 0).any():
            raise ValueError("n_contribs has negative values, indicating an integer overflow.")

        if raster_settings.rich_info:
            return out_feature, radii, depth, normal, distortion, contrib_sum, contrib_max, n_contribs, alpha_mask, num_rendered
        else:
            return out_feature, radii

    @staticmethod
    def backward(ctx, *grads_out: torch.Tensor):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        vertex, shs, feature, opacity, radii, out_feature, depth, normal, distortion, geometryBuffer, binningBuffer, imageBuffer = ctx.saved_tensors

        if raster_settings.rich_info:
            grad_out_feature, _, grad_out_depth, grad_out_normal, grad_out_distortion, _, _, _, _, _ = grads_out
        else:
            raise ValueError("Rich info must be enabled for backward pass.")

        # Restructure args as C++ method expects them
        args = (
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.viewmatrix.contiguous(),
            raster_settings.projmatrix.contiguous(),
            raster_settings.campos.contiguous(),
            raster_settings.sh_degree,
            raster_settings.gamma,
            raster_settings.background_depth,
            raster_settings.background.contiguous(),
            vertex,
            shs,
            feature,
            opacity,
            num_rendered,
            radii,
            out_feature,
            depth,
            normal,
            distortion,
            geometryBuffer,
            binningBuffer,
            imageBuffer,
            grad_out_feature,
            grad_out_depth,
            grad_out_normal,
            grad_out_distortion,
            raster_settings.back_culling,
            raster_settings.rich_info,
            raster_settings.sort_level,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        (
            grad_vertex,
            grad_v_norm,
            grad_shs,
            grad_feature,
            grad_opacity,
        ) = debug_run(_C.rasterize_triangles_backward, *args, debug=raster_settings.debug)

        grads = (
            grad_vertex,
            grad_v_norm,
            grad_shs,
            grad_feature,
            grad_opacity,
            None,
        )
        return grads


class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings: TriangleRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        vertex: torch.Tensor,
        grad_holder: torch.Tensor,
        opacity: torch.Tensor,
        shs: torch.Tensor = None,
        feature: torch.Tensor = None,
    ):
        if (shs is None and feature is None) or (shs is not None and feature is not None):
            raise Exception("Please provide excatly one of either SHs or feature!")

        shs = torch.Tensor([]) if shs is None else shs
        feature = torch.Tensor([]) if feature is None else feature

        # Invoke C++/CUDA rasterization routine
        return _RasterizeTriangles.apply(vertex, grad_holder, shs, feature, opacity, self.raster_settings)
