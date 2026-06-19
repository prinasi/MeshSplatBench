import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.special
from pathlib import Path
from copy import deepcopy
import cv2

from .point_cloud import PointCloud
from .raw_triangle import RawTriangle
from .model_utils import *
from .Base_model import BaseModel
from .point_cloud import PointCloud
from ..renderer.triangle_renderer import TriangleRenderer
from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.scheduler import exponential_scheduler, exponential_step_scheduler, linear_scheduler
from ..utils.sh_utils import RGB2SH, SH2RGB
from ..utils.vis_utils import save_image_tensor, plot_distribution
from ..trainers.trainer_utils import ScharrFilter


class VanillaTSModel(BaseModel):
    def __init__(self, config: Config = None, logger: Logger = None, device: torch.device | int = None):
        super().__init__(config, logger, device)

        config = self.config
        self.max_sh_degree = config.max_sh_degree if config.max_sh_degree is not None else 0
        self.use_color_affine = config.use_color_affine if config.use_color_affine is not None else False
        self.back_culling = config.back_culling if config.back_culling is not None else False
        self.back_culling_prob = config.back_culling_prob if config.back_culling_prob is not None else 1.0
        self.ste_threshold = config.ste_threshold if config.ste_threshold is not None else None
        self.gamma_rescale = config.gamma_rescale if config.gamma_rescale is not None else False
        self.render_spp = config.render_spp if config.render_spp is not None else None
        self.use_vertex_color = config.use_vertex_color if config.use_vertex_color is not None else False
        self.sort_level = config.sort_level if config.sort_level is not None else 0

        self.active_sh_degree = 0
        self.gamma = 1.0
        self.opacity_floor = 0.0
        self.optimizer = None
        self.scene_bbox = None
        self.initialized = False

    @property
    def get_vertex(self):
        return self._vertex

    @property
    def get_xyz(self):
        return self._vertex.mean(dim=1)

    # @property
    # def get_scaling(self):
    #     """
    #     calculate the radius of the smallest enclosing circle of each triangle
    #     """
    #     side_1 = self._vertex[:, 2] - self._vertex[:, 1]
    #     side_2 = self._vertex[:, 0] - self._vertex[:, 2]
    #     side_3 = self._vertex[:, 1] - self._vertex[:, 0]
    #     l_side1 = side_1.norm(dim=1)
    #     l_side2 = side_2.norm(dim=1)
    #     l_side3 = side_3.norm(dim=1)
    #     sq_side1 = l_side1**2
    #     sq_side2 = l_side2**2
    #     sq_side3 = l_side3**2
    #     is_obtuse = torch.stack((sq_side1 + sq_side2 < sq_side3, sq_side2 + sq_side3 < sq_side1, sq_side3 + sq_side1 < sq_side2), dim=1).any(dim=1)

    #     area = torch.cross(side_1, side_2, dim=1).norm(dim=1) / 2 + 1e-10
    #     circumradius = l_side1 * l_side2 * l_side3 / (4 * area)
    #     l_side_max = torch.stack((l_side1, l_side2, l_side3), dim=1).max(dim=1).values
    #     circumradius[is_obtuse] = l_side_max[is_obtuse] / 2

    #     return circumradius

    @property
    def get_scaling(self):
        l_side1 = (self._vertex[:, 2] - self._vertex[:, 1]).norm(dim=1)
        l_side2 = (self._vertex[:, 0] - self._vertex[:, 2]).norm(dim=1)
        l_side3 = (self._vertex[:, 1] - self._vertex[:, 0]).norm(dim=1)
        return torch.stack((l_side1, l_side2, l_side3), dim=1).mean(dim=1)

    @property
    def get_area(self):
        side_1 = self._vertex[:, 1] - self._vertex[:, 0]
        side_2 = self._vertex[:, 2] - self._vertex[:, 0]
        return torch.cross(side_1, side_2, dim=1).norm(dim=1) / 2

    @property
    def get_features(self):
        return torch.cat((self._f_dc, self._f_rest), dim=-2)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity) * (1.0 - self.opacity_floor) + self.opacity_floor

    def set_opacity(self, opacity: float):
        self._opacity.data.fill_(inverse_sigmoid(opacity))

    def setup_color_affine(self, view_count):
        if not self.use_color_affine:
            return

        self._color_affine_weight = nn.Parameter(torch.zeros((view_count, 3, 3)).float().to(self.device), requires_grad=True)
        self._color_affine_weight.data[:, 0, 0] = 1
        self._color_affine_weight.data[:, 1, 1] = 1
        self._color_affine_weight.data[:, 2, 2] = 1
        self._color_affine_bias = nn.Parameter(torch.zeros((view_count, 3)).float().to(self.device), requires_grad=True)

    def setup_scene_info(self, scene_info: dict = None):
        if scene_info is None:
            return

        self.scene_bbox = scene_info["bbox_xyz"]

    def _setup_parameters(self, point_count: int):
        self._vertex = nn.Parameter(torch.empty(point_count, 3, 3).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.empty(point_count, 1).float().to(self.device), requires_grad=True)
        if self.use_vertex_color:
            self._f_dc = nn.Parameter(torch.empty(point_count, 3, 1, 3).float().to(self.device), requires_grad=True)
            self._f_rest = nn.Parameter(torch.empty(point_count, 3, (self.max_sh_degree + 1) ** 2 - 1, 3).float().to(self.device), requires_grad=True)
        else:
            self._f_dc = nn.Parameter(torch.empty(point_count, 1, 3).float().to(self.device), requires_grad=True)
            self._f_rest = nn.Parameter(torch.empty(point_count, (self.max_sh_degree + 1) ** 2 - 1, 3).float().to(self.device), requires_grad=True)

    def _setup_optimizer(self):
        if self.config.optimizer is None:
            return

        l = [
            {"params": [self._vertex], "lr": 0, "name": "vertex"},
            {"params": [self._opacity], "lr": 0, "name": "opacity"},
            {"params": [self._f_dc], "lr": 0, "name": "f_dc"},
            {"params": [self._f_rest], "lr": 0, "name": "f_rest"},
        ]
        if self.use_color_affine:
            l += [
                {"params": [self._color_affine_weight], "lr": 0, "name": "color_affine_weight"},
                {"params": [self._color_affine_bias], "lr": 0, "name": "color_affine_bias"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _setup_lr_scheduler(self):
        args = self.config.optimizer
        if args is None:
            return

        v_scheduler_ = exponential_scheduler(**vars(args.vertex))
        if args.vertex_scale_up_iter is not None and args.vertex_scale_up is not None:
            v_scheduler = lambda iteration: v_scheduler_(iteration) * (1.0 if iteration <= args.vertex_scale_up_iter else args.vertex_scale_up)
        else:
            v_scheduler = v_scheduler_

        self.lr_schedulers = {
            "vertex": v_scheduler,
            "opacity": exponential_scheduler(**vars(args.opacity)),
            "f_dc": exponential_scheduler(**vars(args.f_dc)),
            "f_rest": exponential_scheduler(**vars(args.f_rest)),
        }

        if self.use_color_affine:
            self.lr_schedulers.update(
                {
                    "color_affine_weight": exponential_scheduler(**vars(args.color_affine)),
                    "color_affine_bias": exponential_scheduler(**vars(args.color_affine)),
                }
            )

    def _setup_model_update_utils(self):
        args = self.config.model_update
        if args is None:
            return

        if args.densification is not None and args.densification.grad_threshold_init is not None and args.densification.grad_threshold_final is not None:
            self.grad_threshold_scheduler = exponential_scheduler(
                v_init=args.densification.grad_threshold_init,
                v_final=args.densification.grad_threshold_final,
                max_steps=args.densification.end_iter - args.densification.start_iter,
            )
        if args.opacity_pruning is not None:
            self.opacity_pruning_scheduler = exponential_scheduler(
                v_init=args.opacity_pruning.opacity_threshold_init,
                v_final=args.opacity_pruning.opacity_threshold_final,
                max_steps=args.opacity_pruning.end_iter - args.opacity_pruning.start_iter,
            )
        if args.opacity_clipping is not None:
            self.opacity_clipping_scheduler = exponential_scheduler(
                v_init=args.opacity_clipping.opacity_threshold_init,
                v_final=args.opacity_clipping.opacity_threshold_final,
                max_steps=args.opacity_clipping.end_iter - args.opacity_clipping.start_iter,
            )
        if args.scale_clipping is not None:
            self.scale_max_scheduler = exponential_scheduler(
                v_init=args.scale_clipping.scale_max_init,
                v_final=args.scale_clipping.scale_max_final,
                max_steps=args.scale_clipping.end_iter - args.scale_clipping.start_iter,
            )
        if args.gamma_schedule is not None:
            if args.gamma_schedule.step_scheduler:
                self.gamma_scheduler = exponential_step_scheduler(
                    v_init=args.gamma_schedule.gamma_init,
                    v_final=args.gamma_schedule.gamma_final,
                    max_steps=args.gamma_schedule.end_iter - args.gamma_schedule.start_iter,
                    n_stage=args.gamma_schedule.n_stage,
                )
            else:
                self.gamma_scheduler = exponential_scheduler(
                    v_init=args.gamma_schedule.gamma_init,
                    v_final=args.gamma_schedule.gamma_final,
                    max_steps=args.gamma_schedule.end_iter - args.gamma_schedule.start_iter,
                )
        if args.opacity_scheduler is not None:
            self.opacity_scheduler = linear_scheduler(
                v_init=args.opacity_scheduler.opacity_init,
                v_final=args.opacity_scheduler.opacity_final,
                max_steps=args.opacity_scheduler.end_iter - args.opacity_scheduler.start_iter,
            )

        self.gradient_accum = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.gradient_denom = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.max_radii2D = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_sum = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_max = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_denom = torch.zeros((self._vertex.shape[0]), device=self.device)

    def _training_setup(self):
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_model_update_utils()
        self.initialized = True

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.lr_schedulers:
                param_group["lr"] = self.lr_schedulers[param_group["name"]](iteration)

    def _prune_points_update_states(self, mask: torch.Tensor):
        # update optimizer and parameter after runing _prune_points
        for group in self.optimizer.param_groups:
            if group["name"] in ["color_affine_weight", "color_affine_bias"]:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(param[mask], requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _prune_points(self, prune_mask: torch.Tensor):
        self.gradient_accum = self.gradient_accum[~prune_mask]
        self.gradient_denom = self.gradient_denom[~prune_mask]
        self.max_radii2D = self.max_radii2D[~prune_mask]
        self.contrib_sum = self.contrib_sum[~prune_mask]
        self.contrib_max = self.contrib_max[~prune_mask]
        self.contrib_denom = self.contrib_denom[~prune_mask]
        self._prune_points_update_states(~prune_mask)

    def _grow_points_update_states(self, tensors_dict: dict[str, torch.Tensor]):
        # update optimizer and parameter after runing _grow_points
        for group in self.optimizer.param_groups:
            if group["name"] in ["color_affine_weight", "color_affine_bias"]:
                continue

            param = group["params"][0]
            extension_tensor = tensors_dict[group["name"]]
            new_param = nn.Parameter(torch.cat((param, extension_tensor), dim=0), requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _grow_points(self, grow_mask: torch.Tensor):
        args = self.config.model_update.densification
        split_num = args.split_num
        split_scale_threshold = args.split_scale_threshold if args.split_scale_threshold is not None else 0.0
        split_radii_threshold = args.split_radii_threshold if args.split_radii_threshold is not None else 0.0

        large_point_mask = self.get_scaling > split_scale_threshold
        large_point_mask &= self.max_radii2D >= split_radii_threshold
        clone_mask = grow_mask & ~large_point_mask  # clone small points
        split_mask = grow_mask & large_point_mask  # split large points

        # generate clone points
        clone_vertex = self._vertex[clone_mask]
        # clone_opacity = self._opacity[clone_mask]
        clone_opacity = self.get_opacity[clone_mask]
        clone_opacity = inverse_sigmoid(1 - (1 - clone_opacity) ** 0.5)  # keep the contribution decay constant after cloning
        clone_f_dc = self._f_dc[clone_mask]
        clone_f_rest = self._f_rest[clone_mask]

        clone_normal = torch.cross(clone_vertex[:, 1] - clone_vertex[:, 0], clone_vertex[:, 2] - clone_vertex[:, 0], dim=1)
        clone_normal = F.normalize(clone_normal, dim=1).unsqueeze(1)  # (N, 1, 3)
        clone_scaling = self.get_scaling[clone_mask].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        clone_offset = torch.randn_like(clone_vertex) * clone_scaling  # (N, 3, 3)
        clone_offset = clone_offset - (clone_offset * clone_normal).sum(dim=-1, keepdim=True) * clone_normal  # project to the plane
        clone_vertex = clone_vertex + clone_offset

        # generate split points
        if split_num == 2:  # cut from the center of the longest edge to the opposite vertex
            split_vertex_ori = self._vertex[split_mask]
            side1 = split_vertex_ori[:, 2] - split_vertex_ori[:, 1]
            side2 = split_vertex_ori[:, 0] - split_vertex_ori[:, 2]
            side3 = split_vertex_ori[:, 1] - split_vertex_ori[:, 0]
            l_side = torch.argmax(torch.stack((side1.norm(dim=1), side2.norm(dim=1), side3.norm(dim=1)), dim=1), dim=1)  # longest side
            l_side_p1 = (l_side + 1) % 3
            l_side_p2 = (l_side + 2) % 3
            r = torch.arange(split_vertex_ori.shape[0], device=self.device)
            l_side_center = (split_vertex_ori[r, l_side_p1] + split_vertex_ori[r, l_side_p2]) / 2  # center point of the longest side
            split_triangle_1 = torch.stack((split_vertex_ori[r, l_side], split_vertex_ori[r, l_side_p1], l_side_center), dim=1)
            split_triangle_2 = torch.stack((split_vertex_ori[r, l_side], l_side_center, split_vertex_ori[r, l_side_p2]), dim=1)
            split_vertex = torch.cat((split_triangle_1, split_triangle_2), dim=0)
        elif split_num == 3:  # cut from the barycenter to each vertex
            split_vertex_ori = self._vertex[split_mask]
            barycenter = split_vertex_ori.mean(dim=1)
            split_triangle_1 = torch.stack((split_vertex_ori[:, 0], split_vertex_ori[:, 1], barycenter), dim=1)
            split_triangle_2 = torch.stack((split_vertex_ori[:, 1], split_vertex_ori[:, 2], barycenter), dim=1)
            split_triangle_3 = torch.stack((split_vertex_ori[:, 2], split_vertex_ori[:, 0], barycenter), dim=1)
            split_vertex = torch.cat((split_triangle_1, split_triangle_2, split_triangle_3), dim=0)
        elif split_num == 4:  # cut among the center of each edge
            split_vertex_ori = self._vertex[split_mask]
            center1 = (split_vertex_ori[:, 1] + split_vertex_ori[:, 2]) / 2
            center2 = (split_vertex_ori[:, 0] + split_vertex_ori[:, 2]) / 2
            center3 = (split_vertex_ori[:, 0] + split_vertex_ori[:, 1]) / 2
            split_triangle_1 = torch.stack((split_vertex_ori[:, 0], center3, center2), dim=1)
            split_triangle_2 = torch.stack((split_vertex_ori[:, 1], center1, center3), dim=1)
            split_triangle_3 = torch.stack((split_vertex_ori[:, 2], center2, center1), dim=1)
            split_triangle_4 = torch.stack((center1, center2, center3), dim=1)
            split_vertex = torch.cat((split_triangle_1, split_triangle_2, split_triangle_3, split_triangle_4), dim=0)
        else:
            raise ValueError(f"Unsupported split_num: {split_num}, only support 2, 3, or 4 splits")
        split_opacity = self._opacity[split_mask].repeat(split_num, 1)
        split_f_dc = self._f_dc[split_mask].repeat(split_num, 1, 1, 1) if self.use_vertex_color else self._f_dc[split_mask].repeat(split_num, 1, 1)
        split_f_rest = self._f_rest[split_mask].repeat(split_num, 1, 1, 1) if self.use_vertex_color else self._f_rest[split_mask].repeat(split_num, 1, 1)

        # combine clone and split points
        new_vertex = torch.cat((clone_vertex, split_vertex), dim=0)
        new_opacity = torch.cat((clone_opacity, split_opacity), dim=0)
        new_f_dc = torch.cat((clone_f_dc, split_f_dc), dim=0)
        new_f_rest = torch.cat((clone_f_rest, split_f_rest), dim=0)

        new_points = {
            "vertex": new_vertex,
            "opacity": new_opacity,
            "f_dc": new_f_dc,
            "f_rest": new_f_rest,
        }

        self._prune_points(split_mask)

        new_points_count = new_vertex.shape[0]
        self.gradient_accum = F.pad(self.gradient_accum, (0, new_points_count), value=0)
        self.gradient_denom = F.pad(self.gradient_denom, (0, new_points_count), value=0)
        self.max_radii2D = F.pad(self.max_radii2D, (0, new_points_count), value=0)
        self.contrib_sum = F.pad(self.contrib_sum, (0, new_points_count), value=0)
        self.contrib_max = F.pad(self.contrib_max, (0, new_points_count), value=0)
        self.contrib_denom = F.pad(self.contrib_denom, (0, new_points_count), value=0)
        self._grow_points_update_states(new_points)
        return clone_mask.sum().item(), split_mask.sum().item()

    def _reset_opacity_update_states(self, tensor_dict: torch.Tensor):
        # update optimizer and parameter after runing _opacity_reset
        for group in self.optimizer.param_groups:
            if not group["name"] in tensor_dict:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(tensor_dict[group["name"]], requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"] = torch.zeros_like(new_param)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _clipping_update_states(self, clip_mask: torch.Tensor, clip_value: float, name: str):
        # update optimizer and parameter after runing _scale_clipping or _opacity_clipping
        for group in self.optimizer.param_groups:
            if not group["name"] == name:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(param, requires_grad=True)
            new_param[clip_mask] = clip_value

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"][clip_mask] = 0.0
                stored_state["exp_avg_sq"][clip_mask] = 0.0
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{name}", new_param)

    def _training_statistic(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
        args = self.config.model_update.statistic
        if args is None or not (args.start_iter < iteration <= args.end_iter) or render_pkg is None:
            return

        grad = render_pkg["grad"]
        radii = render_pkg["radii"]
        contrib_sum = render_pkg["contrib_sum"]
        contrib_max = render_pkg["contrib_max"]

        if grad is not None:
            self.gradient_accum += grad
            self.gradient_denom[contrib_max > 0] += 1
        self.contrib_sum = torch.max(self.contrib_sum, contrib_sum)
        self.contrib_max = torch.max(self.contrib_max, contrib_max)
        self.contrib_denom[radii > 0] += 1
        self.max_radii2D = torch.max(self.max_radii2D, radii)

    def _densification(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
        args = self.config.model_update.densification
        if args is None or not (args.start_iter < iteration <= args.end_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        start_iter = args.start_iter
        min_view_count = args.min_view_count
        target_point_num = args.target_point_num
        grow_ratio = args.grow_ratio
        max_grow_ratio = args.max_grow_ratio
        random_select = args.random_select if args.random_select is not None else False

        select_mask = self.gradient_denom >= min_view_count
        select_count = select_mask.sum().item()
        grad = self.gradient_accum / self.gradient_denom.clamp(min=1)
        opacity = self.get_opacity.squeeze(-1)
        grad *= opacity  # prioritize points with high opacity
        grow_mask = select_mask.clone()
        grad_threshold = 0.0

        if hasattr(self, "grad_threshold_scheduler") and not random_select:
            grad_threshold = self.grad_threshold_scheduler(iteration - start_iter)
            grad_mask = grad >= grad_threshold
            grow_mask &= grad_mask

        if target_point_num is not None and grow_ratio is not None and max_grow_ratio is not None:
            grow_count = self._get_diff_point_count(target_point_num, grow_ratio, "grow")
            grow_count = min(grow_count, select_count * max_grow_ratio)

            if random_select:
                log_contrib = torch.log1p(self.contrib_sum)
                log_contrib_min, log_contrib_max = log_contrib.quantile(0.1), log_contrib.quantile(0.9)
                contrib_prob = ((log_contrib - log_contrib_min) / (log_contrib_max - log_contrib_min + 1e-10)).clamp(0.0, 1.0)
                contribution_mask = contrib_prob > torch.rand_like(contrib_prob)
                opacity_mask = opacity > torch.rand_like(opacity) if self.ste_threshold is None else opacity > self.ste_threshold
                select_idx = torch.argwhere(select_mask & opacity_mask & contribution_mask).squeeze(1)
                grow_idx = select_idx[torch.randperm(select_idx.shape[0])[: int(grow_count)]]
            else:
                select_idx = torch.argwhere(select_mask).squeeze(1)
                grow_idx = select_idx[torch.topk(grad[select_mask], k=int(grow_count)).indices]
                grad_threshold = max(grad[grow_idx[-1]] if len(grow_idx) > 0 else 0.0, grad_threshold)

            grow_mask_2 = torch.zeros_like(grow_mask, dtype=torch.bool)
            grow_mask_2[grow_idx] = True
            grow_mask &= grow_mask_2

        # visualize densified triangles
        if "densification_img" in render_pkg:
            shs = self.get_features
            if self.use_vertex_color:
                shs[grow_mask, :, 0] = RGB2SH(torch.tensor([1.0, 0.0, 0.0], device=self.device))  # set triangles with high gradients to red
                shs[grow_mask, :, 1:] = 0.0  # set other SH coefficients to zero
            else:
                shs[grow_mask, 0] = RGB2SH(torch.tensor([1.0, 0.0, 0.0], device=self.device))  # set triangles with high gradients to red
                shs[grow_mask, 1:] = 0.0  # set other SH coefficients to zero
            with torch.no_grad():
                img = self.forward(render_pkg["camera"], "black", False, shs=shs)["render"]
            render_pkg["densification_img"] = img

        self.gradient_accum[select_mask] = 0
        self.gradient_denom[select_mask] = 0
        n_clone, n_split = self._grow_points(grow_mask)
        self.logger.info(
            f"[ITER {iteration}, densification] Cloning {n_clone} points, splitting {n_split} points, "
            + f"{select_count} points selected by gradient_denom >= {min_view_count}, grad threshold: {grad_threshold:.2e}"
        )

    def _opacity_pruning(self, iteration: int):
        args = self.config.model_update.opacity_pruning
        if args is None or not (args.start_iter < iteration <= args.hold_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        start_iter = args.start_iter

        opacity_threshold = self.opacity_pruning_scheduler(iteration - start_iter)
        prune_mask = (self.get_opacity < opacity_threshold).squeeze(-1)

        self._prune_points(prune_mask)
        self.logger.info(f"[ITER {iteration}, opacity pruning] Pruning {prune_mask.sum().item()} points, opacity threshold: {opacity_threshold:.5f}")

    def _opacity_clipping(self, iteration: int):
        args = self.config.model_update.opacity_clipping
        if args is None or not (args.start_iter < iteration <= args.hold_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        opacity_threshold = self.opacity_clipping_scheduler(iteration - args.start_iter)
        clip_mask = (self.get_opacity > opacity_threshold).squeeze(-1)
        clip_count = clip_mask.sum().item()

        if clip_count > 0:
            self._clipping_update_states(clip_mask, 10.0, "opacity")
        self.logger.info(f"[ITER {iteration}, opacity clipping] Clipping {clip_count} points, opacity threshold: {opacity_threshold:.5f}")

    def _scale_pruning(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
        args = self.config.model_update.scale_pruning
        if args is None or not (args.start_iter < iteration <= args.end_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        radii_threshold = args.radii_threshold
        scale_threshold = args.scale_threshold
        min_area_threshold = args.min_area_threshold

        width, height = render_pkg["camera"].image_width, render_pkg["camera"].image_height
        if radii_threshold == -1:
            radii_threshold = max(width, height)
        elif isinstance(radii_threshold, float) and 0 < radii_threshold < 1:
            radii_threshold = int(max(width, height) * radii_threshold)
        elif radii_threshold is None:
            assert isinstance(radii_threshold, int) and radii_threshold > 0, "radii_threshold should be a positive float or int"

        radii_prune_mask = self.max_radii2D >= radii_threshold if radii_threshold is not None else torch.zeros_like(self.max_radii2D, dtype=torch.bool)
        scale_prune_mask = self.get_scaling >= scale_threshold if scale_threshold is not None else torch.zeros_like(self.max_radii2D, dtype=torch.bool)
        area_prune_mask = self.get_area <= min_area_threshold if min_area_threshold is not None else torch.zeros_like(self.max_radii2D, dtype=torch.bool)
        prune_mask = radii_prune_mask | scale_prune_mask | area_prune_mask

        self._prune_points(prune_mask)
        self.logger.info(
            f"[ITER {iteration}, scale pruning] Pruning {prune_mask.sum().item()} points, "
            + f"{radii_prune_mask.sum().item()} by radii, {scale_prune_mask.sum().item()} by scale, {area_prune_mask.sum().item()} by area, "
            + f"radii threshold: {radii_threshold}, scale threshold: {scale_threshold}, area threshold: {min_area_threshold}"
        )

    def _rescale_triangles(self, rescale_ratio: torch.Tensor | float, rescale_mask: torch.Tensor = None):
        """
        rescale triangles by rescale_ratio
        args:
            rescale_ratio: (M, ) tensor or a float, rescale ratio for each triangle
            rescale_mask: (N, ) tensor, mask for triangles to rescale
        return:
            rescaled_vertex: (N, 3, 3) tensor, rescaled vertex
        """
        vertex = self._vertex[rescale_mask] if rescale_mask is not None else self._vertex
        if isinstance(rescale_ratio, torch.Tensor):
            assert rescale_ratio.dim() == 1 and rescale_ratio.size(0) == vertex.size(0)
            rescale_ratio = rescale_ratio.unsqueeze(1).unsqueeze(1)

        t_center = vertex.mean(dim=1, keepdim=True)
        rescaled_vertex = (vertex - t_center) * rescale_ratio + t_center
        return rescaled_vertex

    def _scale_clipping(self, iteration: int):
        args = self.config.model_update.scale_clipping
        if args is None or not (args.start_iter < iteration <= args.hold_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        scale_max = self.scale_max_scheduler(iteration - args.start_iter)
        scaling = self.get_scaling
        clip_mask = scaling > scale_max

        rescale_ratio = scale_max / scaling[clip_mask]
        rescaled_vertex = self._rescale_triangles(rescale_ratio, clip_mask)

        clip_count = clip_mask.sum().item()

        if clip_count > 0:
            self._clipping_update_states(clip_mask, rescaled_vertex, "vertex")
        self.logger.info(f"[ITER {iteration}, scale clipping] Clipping {clip_count} points, scale max: {scale_max:.5f}")

    def _get_diff_point_count(self, target_point_num: int, ratio: float, type: str) -> int:
        """
        Calculate the number of points to be pruned or grown based on the target point number.
        Considers the current number of valid points inside the scene bounding box and above the STE threshold.
        Args:
            target_point_num: The target number of points in the scene.
            ratio: The ratio to adjust the number of points to be pruned or grown.
            type: The type of operation, either "grow" or "prune".
        Returns:
            diff_point_count: The number of points to be pruned or grown.
        """
        total_point_count = self._vertex.shape[0]
        inside_mask = get_inside_mask(self.get_xyz, self.scene_bbox)
        ste_mask = (self.get_opacity > self.ste_threshold if self.ste_threshold is not None else torch.ones_like(self.get_opacity, dtype=torch.bool)).squeeze()
        valid_point_count = (inside_mask & ste_mask).sum().item()
        if valid_point_count == 0:
            self.logger.warning("No valid points found inside the scene bounding box or above the STE threshold.")
            return 0
        target_point_num = target_point_num * total_point_count / valid_point_count

        if type == "grow":
            diff_point_count = max(0, (target_point_num * 1.01 - total_point_count) * ratio)
        elif type == "prune":
            diff_point_count = max(0, (total_point_count - target_point_num * 0.99) * ratio)
        else:
            raise ValueError(f"Unsupported type: {type}, only support 'grow' or 'prune'")
        return int(diff_point_count)

    def _contribution_pruning(self, iteration: int):
        args = self.config.model_update.contribution_pruning
        if args is None or not (args.start_iter < iteration <= args.end_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        min_view_count = args.min_view_count
        target_point_num = args.target_point_num
        prune_ratio = args.prune_ratio
        max_prune_ratio = args.max_prune_ratio
        contrib_max_ratio = args.contrib_max_ratio
        sparsity_retain_ratio = args.sparsity_retain_ratio
        downsample_iteration = args.downsample_iteration
        downsample_point_num = args.downsample_point_num
        contrib_max_threshold = args.contrib_max_threshold
        contrib_sum_threshold = args.contrib_sum_threshold

        select_mask = self.contrib_denom >= min_view_count
        select_count = select_mask.sum().item()
        prune_mask = select_mask.clone()

        # use contribution threshold to control pruning
        if contrib_max_threshold is not None or contrib_sum_threshold is not None:
            contrib_max_mask = (
                self.contrib_max <= contrib_max_threshold if contrib_max_threshold is not None else torch.zeros_like(self.contrib_max, dtype=torch.bool)
            )
            contrib_sum_mask = (
                self.contrib_sum <= contrib_sum_threshold if contrib_sum_threshold is not None else torch.zeros_like(self.contrib_sum, dtype=torch.bool)
            )
            prune_mask &= contrib_max_mask | contrib_sum_mask

        # use target point number to control pruning
        if target_point_num is not None and prune_ratio is not None and max_prune_ratio is not None:
            if downsample_iteration is not None and downsample_point_num is not None:
                for iter, point_num in zip(downsample_iteration, downsample_point_num):
                    if iteration > iter:
                        target_point_num = point_num
                        contrib_max_ratio *= 0.5
                        new_sparsity_retain_ratio = sparsity_retain_ratio + (0.8 - sparsity_retain_ratio) * 0.5
                        prune_ratio *= (1 - sparsity_retain_ratio) / (1 - new_sparsity_retain_ratio)
                        sparsity_retain_ratio = new_sparsity_retain_ratio

            prune_count = self._get_diff_point_count(target_point_num, prune_ratio, "prune")
            prune_count = min(prune_count, select_count * max_prune_ratio)
            contrib_max_prune_count = int(prune_count * contrib_max_ratio)
            contrib_sum_prune_count = int(prune_count * (1 - contrib_max_ratio))

            select_idx = torch.argwhere(select_mask).squeeze(1)
            contrib_max_prune_idx = torch.argsort(self.contrib_max[select_mask])[:contrib_max_prune_count]
            contrib_max_prune_idx = select_idx[contrib_max_prune_idx]
            contrib_sum_prune_idx = torch.argsort(self.contrib_sum[select_mask])[:contrib_sum_prune_count]
            contrib_sum_prune_idx = select_idx[contrib_sum_prune_idx]
            prune_idx = torch.cat((contrib_max_prune_idx, contrib_sum_prune_idx)).unique()

            prune_mask_2 = torch.zeros_like(select_mask, dtype=torch.bool)
            prune_mask_2[prune_idx] = True
            prune_mask &= prune_mask_2

        # use distance to the nearest point to retain sparse points
        retain_point_count = 0
        if sparsity_retain_ratio is not None and sparsity_retain_ratio > 0:
            prune_idx = torch.argwhere(prune_mask).squeeze(1)
            retain_point_count = int(sparsity_retain_ratio * len(prune_idx))
            dist = inter_point_distance(self.get_xyz)
            retain_idx = prune_idx[torch.argsort(dist[prune_idx], descending=True)[:retain_point_count]]
            prune_mask[retain_idx] = False

        max_pruned_contrib_max = self.contrib_max[prune_mask].max().item() if prune_mask.any() else 0.0
        max_pruned_contrib_sum = self.contrib_sum[prune_mask].max().item() if prune_mask.any() else 0.0

        self.contrib_sum[select_mask] = 0
        self.contrib_max[select_mask] = 0
        self.contrib_denom[select_mask] = 0
        self._prune_points(prune_mask)
        self.logger.info(
            f"[ITER {iteration}, contribution pruning] Pruning {prune_mask.sum().item()} points, "
            + f"{select_count} points selected by contrib_denom >= {min_view_count}, "
            + f"max pruned contrib_max: {max_pruned_contrib_max:.5f}, max pruned contrib_sum: {max_pruned_contrib_sum:.5f}"
        )

    def _opacity_reset(self, iteration: int):
        args = self.config.model_update.opacity_reset
        if args is None or not (args.start_iter < iteration <= args.end_iter and (iteration - args.start_iter) % args.interval_iter == 0):
            return

        reset_value = args.reset_value

        opacity_old = self.get_opacity
        opacity_reset = torch.ones_like(opacity_old) * reset_value
        reset_mask = opacity_old > reset_value

        opacity_new = inverse_sigmoid(torch.min(opacity_old, opacity_reset))
        self._reset_opacity_update_states({"opacity": opacity_new})
        self.logger.info(f"[ITER {iteration}, opacity reset] Reset opacity of {reset_mask.sum().item()} points to {reset_value}")

    def _set_gamma(self, iteration: int):
        args = self.config.model_update.gamma_schedule
        if args is None or not (args.start_iter < iteration <= args.end_iter):
            return

        self.gamma = self.gamma_scheduler(iteration - args.start_iter)

    def _set_opacity_floor(self, iteration: int):
        args = self.config.model_update.opacity_scheduler
        if args is None or not (args.start_iter < iteration <= args.end_iter):
            return

        self.opacity_floor = self.opacity_scheduler(iteration - args.start_iter)

    def _set_sh_degree(self, iteration: int):
        args = self.config.model_update.sh_schedule
        if args is None:
            return

        active_sh_degree = 0
        for iter in args.one_up_iters:
            if iteration > iter:
                active_sh_degree += 1
        self.active_sh_degree = min(active_sh_degree, self.max_sh_degree)

    def state_update(self, iteration: int):
        if self.config.model_update is None:
            return

        self._set_gamma(iteration)
        self._set_opacity_floor(iteration)
        self._set_sh_degree(iteration)

    @torch.no_grad()
    def model_update(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
        if self.config.model_update is None:
            return

        self._training_statistic(iteration, render_pkg)
        self._opacity_pruning(iteration)
        self._opacity_clipping(iteration)
        self._scale_pruning(iteration, render_pkg)
        self._scale_clipping(iteration)
        self._contribution_pruning(iteration)
        self._densification(iteration, render_pkg)
        self._opacity_reset(iteration)

        self.state_update(iteration)

    def forward(
        self,
        camera: Camera,
        background: str = None,
        is_training: bool = True,
        color_affine: bool = None,
        back_culling: bool = None,
        sh_degree: int = None,
        gamma: float = None,
        ste_threshold: float = None,
        shs: torch.Tensor = None,
        sort_level: int = None,
    ) -> dict[str, torch.Tensor]:
        """
        Render the scene.
        """
        color_affine = color_affine if color_affine is not None else self.use_color_affine
        # back_culling = back_culling if back_culling is not None else self.back_culling
        sh_degree = sh_degree if sh_degree is not None else self.active_sh_degree
        gamma = gamma if gamma is not None else self.gamma
        sort_level = sort_level if sort_level is not None else self.sort_level
        if back_culling is None:
            if not is_training:
                back_culling = self.back_culling
            elif self.back_culling and torch.rand(1).item() < self.back_culling_prob:
                back_culling = True
            else:
                back_culling = False

        bg_color = get_color_tensor(background).to(self.device) if background is not None else camera.bg_color
        if bg_color is None:
            raise ValueError("Background color must be specified or camera must have a background color.")

        vertex = self.get_vertex
        shs = self.get_features if shs is None else shs
        opacity = self.get_opacity

        # rescale triangles to keep the integration of triangle opacity invariant under different gamma
        if self.gamma_rescale:
            beta = 1 / gamma
            rescale_ratio = 1 / np.sqrt(2**beta * beta * scipy.special.gamma(beta))
            vertex_rescale = self._rescale_triangles(rescale_ratio)

        ste_threshold = ste_threshold if ste_threshold is not None else self.ste_threshold
        if ste_threshold is not None:
            opacity_ste = ((opacity >= ste_threshold).float() - opacity).detach() + opacity

        bg_depth = (camera.camera_center.view(1, 1, 3) - vertex).norm(dim=-1).max().item()
        supersampling = self.render_spp is not None and self.render_spp > 1

        if supersampling:
            assert isinstance(self.render_spp, int)
            w, h = camera.image_width, camera.image_height
            camera = deepcopy(camera)
            camera.image_width = w * self.render_spp
            camera.image_height = h * self.render_spp

        renderer = TriangleRenderer(
            camera,
            bg_depth=bg_depth,
            bg_color=bg_color,
            sh_degree=min(sh_degree, self.max_sh_degree),
            gamma=gamma,
            back_culling=back_culling,
            rich_info=is_training,
            sort_level=sort_level,
        )
        output_pkg = renderer.render(
            vertex_rescale if self.gamma_rescale else vertex,
            shs,
            None,
            opacity_ste if ste_threshold is not None else opacity,
        )

        if supersampling:
            output_pkg["render"] = F.interpolate(output_pkg["render"].unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
            if "radii" in output_pkg:
                output_pkg["radii"] = output_pkg["radii"] // self.render_spp
            if "depth" in output_pkg:
                output_pkg["depth"] = F.interpolate(output_pkg["depth"].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            if "normal" in output_pkg:
                output_pkg["normal"] = F.interpolate(output_pkg["normal"].unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
            if "distortion" in output_pkg:
                output_pkg["distortion"] = F.interpolate(output_pkg["distortion"].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            if "alpha_mask" in output_pkg:
                output_pkg["alpha_mask"] = F.interpolate(output_pkg["alpha_mask"].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)

        if is_training:
            render_pkg = {
                "camera": camera,
                "scaling": self.get_scaling,
                "opacity": opacity,
                "vertex": vertex,
            }
            render_pkg.update(output_pkg)
        else:
            render_pkg = {
                "render": output_pkg["render"],
            }

        if self.use_color_affine and color_affine:
            image = render_pkg["render"]
            uid = camera.uid
            image_transformed = (image.permute(1, 2, 0) @ self._color_affine_weight[uid] + self._color_affine_bias[uid]).permute(2, 0, 1)
            render_pkg["render"] = image_transformed.clamp(0, 1)
            render_pkg["render_original"] = image

        return render_pkg

    def render_points(
        self,
        camera: Camera,
        render_pkg: dict = None,
        alpha_thres: float = 0.99,
        depth_grad_quantile: float = 0.95,
        ret_pointcloud: bool = True,
    ) -> PointCloud | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a point cloud from the given viewpoint.
        """
        if render_pkg is None:
            render_pkg = self.forward(camera, "black", True, False)

        xyz = camera.get_xyz_from_depth(render_pkg["depth"])  # (H, W, 3), xyz in world space
        color = render_pkg["render"].permute(1, 2, 0).clip(0, 1)  # (H, W, 3)
        normal = render_pkg["normal"].permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T  # (H, W, 3), normal in world space

        mask = render_pkg["alpha_mask"] > alpha_thres

        # erode the mask to remove boundary points
        # mask = mask.cpu().numpy()
        # mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1).astype(bool)
        # mask = torch.tensor(mask, device=self.device)

        if depth_grad_quantile is not None and 0 <= depth_grad_quantile < 1:
            depth_grad = ScharrFilter()(render_pkg["depth"].unsqueeze(0).unsqueeze(0), ret_norm=True).squeeze()
            depth_grad_mask = depth_grad < torch.quantile(depth_grad, depth_grad_quantile)
            mask &= depth_grad_mask

        mask = mask.view(-1)
        xyz = xyz.view(-1, 3)[mask]
        color = color.view(-1, 3)[mask]
        normal = normal.view(-1, 3)[mask]
        if ret_pointcloud:
            return PointCloud(*to_numpy(xyz, color, normal))
        else:
            pixels = camera.get_pixels()  # (H, W, 2)
            pixels = pixels.view(-1, 2)[mask]
            return xyz, color, normal, pixels

    def savePLY(self, ply_path: str, bbox_filtering: bool = True):
        self.logger.info(f"Saving triangles to {ply_path}")
        triangle_model = self.toRawTriangle(bbox_filtering)
        triangle_model.savePLY(ply_path, save_extra=True)

    def loadPLY(self, ply_path: str) -> "VanillaTSModel":
        self.logger.info(f"Loading triangles from {ply_path}")
        triangle_model = RawTriangle(ply_path=ply_path)
        return self.fromRawTriangle(triangle_model)

    @torch.no_grad()
    def toRawTriangle(self, bbox_filtering: bool = True) -> RawTriangle:
        vertex = self._vertex
        opacity = self._opacity
        shs = self.get_features.view(vertex.shape[0], 3, -1) if self.use_vertex_color else self.get_features.view(vertex.shape[0], -1)

        if bbox_filtering and self.scene_bbox is not None:
            mask = get_inside_mask(self.get_xyz, self.scene_bbox)
            vertex = vertex[mask]
            opacity = opacity[mask]
            shs = shs[mask]

        if self.ste_threshold is not None:
            ste_mask = torch.sigmoid(opacity).squeeze() > self.ste_threshold
            vertex = vertex[ste_mask]
            opacity = ste_mask[ste_mask].unsqueeze(-1).float() * 10
            shs = shs[ste_mask]
        return RawTriangle(*to_numpy(vertex, opacity, shs))

    def fromRawTriangle(self, triangle_model: RawTriangle) -> "VanillaTSModel":
        np = triangle_model.vertex.shape[0]
        use_vertex_color = len(triangle_model.shs.shape) == 3
        if use_vertex_color != self.use_vertex_color:
            raise ValueError(f"Model use_vertex_color {self.use_vertex_color} does not match triangle_model use_vertex_color {use_vertex_color}")

        shs = torch.tensor(triangle_model.shs).float().to(self.device)
        shs = shs.view(np, 3, -1, 3) if self.use_vertex_color else shs.view(np, -1, 3)
        features = torch.zeros((np, (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        if self.use_vertex_color:
            features = features.unsqueeze(1).repeat(1, 3, 1, 1)
        if shs.shape[-2] > features.shape[-2]:
            features = shs[..., : features.shape[-2], :]
        else:
            features[..., : shs.shape[-2], :] = shs

        self._vertex = nn.Parameter(torch.tensor(triangle_model.vertex).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.tensor(triangle_model.opacity).float().to(self.device), requires_grad=True)
        self._f_dc = nn.Parameter(features[..., :1, :].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[..., 1:, :].contiguous(), requires_grad=True)

        self._training_setup()
        return self

    def saveGLB(self, glb_path: str, bbox_filtering: bool = True, process: bool = False):
        self.logger.info(f"Saving triangles to {glb_path}")
        triangle_model = self.toRawTriangle(bbox_filtering)
        triangle_model.saveGLB(glb_path, save_back=not self.back_culling, process=process)

    def loadGLB(self, glb_path: str) -> "VanillaTSModel":
        self.logger.info(f"Loading triangles from {glb_path}")
        triangle_model = RawTriangle(glb_path=glb_path)
        return self.fromRawTriangle(triangle_model)

    def save_ckpt(self, ckpt_path: str):
        self.logger.info(f"Saving checkpoint to {ckpt_path}")
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        save_items = (self.state_dict(), self.optimizer.state_dict(), self.scene_bbox, self.gamma)
        torch.save(save_items, open(ckpt_path, "wb"))

    def load_ckpt(self, ckpt_path: str) -> "VanillaTSModel":
        (params_state_dict, optimizer_state_dict, self.scene_bbox, self.gamma) = torch.load(open(ckpt_path, "rb"))

        point_count = params_state_dict["_vertex"].shape[0]
        self._setup_parameters(point_count)
        self.load_state_dict(params_state_dict)

        self._training_setup()
        self.optimizer.load_state_dict(optimizer_state_dict)
        return self

    def _sample_points(
        self,
        points: torch.Tensor,
        shs: torch.Tensor,
        normals: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config.sampling
        sample_method = args.sample_method
        n_sample_inside = args.n_sample_inside
        n_sample_outside = args.n_sample_outside
        grid_size_inside = args.grid_size_inside
        grid_size_outside = args.grid_size_outside

        if name == "inside":
            n_sample = n_sample_inside
            grid_size = grid_size_inside
        elif name == "outside":
            n_sample = n_sample_outside
            grid_size = grid_size_outside
        else:
            raise ValueError(f"Unknown sampling name: {name}")

        self.logger.info(f"Running {name} sampling")
        if sample_method == "random":
            if n_sample > points.shape[0] or n_sample <= 0:
                self.logger.warning(f"target sample number {n_sample} is invalid, using all points")
                return points, shs, normals
            sample_idx = torch.randperm(points.shape[0])[:n_sample]
            sampled_points, sampled_shs, sampled_normals = points[sample_idx], shs[sample_idx], normals[sample_idx]
        elif sample_method == "grid":
            grid_size = grid_size_search(points, n_sample) if grid_size is None else grid_size
            sampled_points, sampled_shs, sampled_normals = grid_sampling(points, shs, normals, grid_size=grid_size)
            sampled_normals = sampled_normals / sampled_normals.norm(dim=1, keepdim=True)
        elif sample_method == "direct":
            sampled_points, sampled_shs, sampled_normals = points, shs, normals
        else:
            raise ValueError(f"Unknown sampling method: {sample_method}")

        self.logger.info(f"Sampled {sampled_points.shape[0]} points from {points.shape[0]} points using {sample_method} sampling")
        if sample_method == "grid":
            self.logger.info(f"Grid size: {grid_size:.5f}")

        return sampled_points, sampled_shs, sampled_normals

    def random_pcd(self) -> PointCloud:
        config = self.config.random_init
        if config is None:
            raise ValueError("Random initialization config is not provided")

        bbox_list = config.bbox_list
        point_num_list = config.point_num_list
        normal_list = config.normal_list

        pcd = PointCloud()
        for bbox, point_num, normal in zip(bbox_list, point_num_list, normal_list):
            bbox = np.array(bbox, dtype=np.float32)
            points = np.random.rand(point_num, 3).astype(np.float32) * (bbox[3:] - bbox[:3]) + bbox[:3]
            colors = np.random.rand(point_num, 3).astype(np.float32)
            if normal == "random":
                normals = np.random.randn(point_num, 3).astype(np.float32)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            else:
                normals = np.tile(np.array(normal, dtype=np.float32), (point_num, 1))
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            pcd += PointCloud(points=points, colors=colors, normals=normals)

        return pcd

    def create_from_pcd(self, pcd: PointCloud = None):
        if pcd is None or len(pcd) == 0:
            pcd = self.random_pcd()

        args = self.config.sampling
        if args is None:
            raise ValueError("Sampling config is not provided")

        init_opacity = args.init_opacity if args.init_opacity is not None else 0.1
        duplicate_count = args.duplicate_count if args.duplicate_count is not None else 1

        points = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        shs = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        normals = torch.tensor(np.asarray(pcd.normals)).float().to(self.device)
        if not normals.any():
            normals = torch.randn_like(points)
        normals = normals / normals.norm(dim=1, keepdim=True)

        inside_mask = get_inside_mask(points, self.scene_bbox)

        inside_points = points[inside_mask]
        inside_shs = shs[inside_mask]
        inside_normals = normals[inside_mask]
        inside_points, inside_shs, inside_normals = self._sample_points(inside_points, inside_shs, inside_normals, "inside")

        outside_points = points[~inside_mask]
        outside_shs = shs[~inside_mask]
        outside_normals = normals[~inside_mask]
        outside_points, outside_shs, outside_normals = self._sample_points(outside_points, outside_shs, outside_normals, "outside")

        points = torch.cat((inside_points, outside_points), dim=0)
        shs = torch.cat((inside_shs, outside_shs), dim=0)
        normals = torch.cat((inside_normals, outside_normals), dim=0)
        scaling = inter_point_distance(points)[..., None]

        if init_opacity == "random":
            opacities = inverse_sigmoid(torch.rand((points.shape[0], 1)).float().to(self.device))
        else:
            opacities = inverse_sigmoid(torch.ones((points.shape[0], 1)).float().to(self.device) * init_opacity)
        features = torch.zeros((shs.shape[0], (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        features[:, 0, :] = shs

        # repeat points
        if duplicate_count > 1:
            self.logger.info(f"Duplicate points {duplicate_count} times")
            points_random = [points]
            for i in range(duplicate_count - 1):
                random_offset = torch.rand((scaling.shape[0], 3)).float().to(self.device)
                random_offset = (random_offset * 2 - 1) * 0.5 * scaling
                points_random.append(points + random_offset.to(points.device))

            points = torch.cat(points_random, dim=0)
            opacities = opacities.repeat(duplicate_count, 1)
            features = features.repeat(duplicate_count, 1, 1)
            normals = normals.repeat(duplicate_count, 1)
            scaling = inter_point_distance(points)[..., None]

            # pcd_test = PointCloud(points=points.cpu().numpy(), colors=(SH2RGB(features[:, 0, :]).squeeze(1).cpu().numpy() * 255.0).astype(np.uint8), normals=normals.cpu().numpy())
            # pcd_test.storePly("test.ply")

        # make equilateral triangles
        up = torch.tensor([0, 0, 1]).float().to(self.device).repeat(points.shape[0], 1)
        u_dir = torch.cross(up, normals, dim=1)
        u_dir[u_dir.norm(dim=1) < 1e-10] = torch.tensor([1, 0, 0]).float().to(self.device)
        u_dir = u_dir / u_dir.norm(dim=1, keepdim=True)
        v_dir = torch.cross(normals, u_dir, dim=1)
        v_dir[v_dir.norm(dim=1) < 1e-10] = torch.tensor([0, 1, 0]).float().to(self.device)
        v_dir = v_dir / v_dir.norm(dim=1, keepdim=True)

        v1 = points + u_dir * scaling
        v2 = points + (-1 / 2 * u_dir + np.sqrt(3) / 2 * v_dir) * scaling
        v3 = points + (-1 / 2 * u_dir - np.sqrt(3) / 2 * v_dir) * scaling
        vertex = torch.stack((v1, v2, v3), dim=1)

        if self.back_culling:
            vertex_back = torch.stack((v3, v2, v1), dim=1)
            vertex = torch.cat((vertex, vertex_back), dim=0)
            opacities = torch.cat((opacities, opacities), dim=0)
            features = torch.cat((features, features), dim=0)

        self.logger.info(f"Number of points at initialisation: {vertex.shape[0]}")

        self._vertex = nn.Parameter(vertex, requires_grad=True)
        self._opacity = nn.Parameter(opacities, requires_grad=True)
        if self.use_vertex_color:
            features = features.unsqueeze(1).repeat(1, 3, 1, 1)
        self._f_dc = nn.Parameter(features[..., :1, :].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[..., 1:, :].contiguous(), requires_grad=True)

        self._training_setup()
