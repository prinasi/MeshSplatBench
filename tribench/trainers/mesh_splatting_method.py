"""MeshSplatting training method for the generic TrainingLoop."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from tribench.core.cameras import CameraBatch
from tribench.renderers.base import RenderOutput
from tribench.trainers.checkpoints import find_latest_point_cloud_checkpoint
from tribench.trainers.hooks import TrainingMethod
from tribench.trainers.registry import register_training_method


def _seed_native_state(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device("cuda:0"))


class _MeshSplattingScene:
    """Small scene wrapper matching the upstream MeshSplatting Scene API."""

    def __init__(
        self,
        args: SimpleNamespace,
        triangles,
        init_weight: float,
        set_sigma: float,
        *,
        shuffle: bool = True,
        resolution_scales: list[float] | None = None,
    ) -> None:
        from tribench.vendor.mesh_splatting.scene.dataset_readers import (
            sceneLoadTypeCallbacks,
        )
        from tribench.vendor.mesh_splatting.utils.camera_utils import (
            cameraList_from_camInfos,
            camera_to_JSON,
        )

        self.model_path = args.model_path
        self.triangles = triangles
        self.loaded_iter = None
        self.train_cameras: dict[float, list[Any]] = {}
        self.test_cameras: dict[float, list[Any]] = {}

        source_path = Path(args.source_path)
        if (source_path / "cameras.npz").exists():
            scene_info = sceneLoadTypeCallbacks["DTU"](
                args.source_path, args.images, args.eval
            )
        elif (source_path / "sparse").exists():
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif (source_path / "transforms_train.json").exists():
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            raise FileNotFoundError(
                f"Cannot recognize MeshSplatting dataset layout: {source_path}"
            )

        output_path = Path(self.model_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if scene_info.ply_path and Path(scene_info.ply_path).exists():
            shutil.copyfile(scene_info.ply_path, output_path / "input.ply")

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for idx, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(idx, cam))
        (output_path / "cameras.json").write_text(
            json.dumps(json_cams, indent=2), encoding="utf-8"
        )

        if shuffle:
            random.seed(7)
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales or [1.0]:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if scene_info.point_cloud is None:
            raise RuntimeError(f"No point cloud available for {source_path}")
        self.triangles.create_from_pcd(scene_info.point_cloud, init_weight, set_sigma)

    def save(self, iteration: int) -> str:
        point_cloud_path = Path(self.model_path) / "point_cloud" / f"iteration_{iteration}"
        self.triangles.save_parameters(str(point_cloud_path))
        return str(point_cloud_path / "point_cloud_state_dict.pt")

    def getTrainCameras(self, scale: float = 1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0):
        return self.test_cameras[scale]


@register_training_method("mesh-splatting")
class MeshSplattingTrainingMethod(TrainingMethod):
    """Training method implementation for the bundled MeshSplatting backend."""

    def __init__(
        self,
        dataset: str | Path,
        output_dir: str | Path,
        *,
        max_steps: int = 30_000,
        images: str = "images",
        llffhold: int = 8,
        resolution: int = -1,
        eval_split: bool = True,
        white_background: bool = False,
        dtu_eval_mode: str = "full",
        indoor: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.max_steps = max_steps
        self.images_dir = images
        self.llffhold = llffhold
        self.resolution = resolution
        self.eval_split = eval_split
        self.white_background = white_background
        self.dtu_eval_mode = str((extra_args or {}).get("dtu_eval_mode", dtu_eval_mode)).lower()
        self.indoor = indoor
        self.extra_args = extra_args or {}

        self._model = None
        self._optimizer = None
        self._scene = None
        self._bg_color: torch.Tensor | None = None
        self._opt = None
        self._pipe = None
        self._render = None
        self._viewpoint_stack: list[Any] = []
        self._train_cameras: list[Any] = []
        self._test_cameras: list[Any] = []
        self._initialized = False
        self._step = 0
        self._need_delaunay = False
        self._prune_threshold = 0.0
        self._final_cleaned = False
        self._depth_l1_weight = None
        self._last_native_cam = None

    def _build_dataset_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            sh_degree=3,
            source_path=str(self.dataset_path),
            model_path=str(self.output_dir),
            images=self.images_dir,
            resolution=self.resolution,
            white_background=self.white_background,
            dtu_eval_mode=self.dtu_eval_mode,
            data_device=str(self.extra_args.get("data_device", "cuda")),
            eval=self.eval_split,
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        seed = self.extra_args.get("seed", 0)
        if seed is not None:
            _seed_native_state(int(seed))

        from tribench.vendor.mesh_splatting.scene.triangle_model import TriangleModel
        from tribench.vendor.mesh_splatting.triangle_renderer import render
        from tribench.vendor.mesh_splatting.utils.general_utils import get_expon_lr_func

        self._render = render

        opt_defaults = dict(
            iterations=self.max_steps,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=max(self.max_steps, 30_000),
            lambda_dssim=0.2,
            densification_interval=500,
            densify_from_iter=500,
            densify_until_iter=10_000,
            run_restricted_delaunay=-1,
            random_background=False,
            feature_lr=0.0016,
            max_points=4_000_000,
            max_primitives=None,
            set_weight=0.28,
            weight_lr=0.03,
            lambda_weight=1.9e-06,
            iteration_mesh=5_000,
            lambda_normals=0.00005,
            lambda_normals_super=0.01,
            add_percentage=1.23,
            set_sigma=1.0,
            intervall_add_triangles=500,
            prune_triangles_threshold=0.235,
            lr_triangles_points_init=0.0015,
            start_opacity_floor=5_000,
            start_pruning=4_000,
            sigma_until=30_000,
            final_opacity_iter=24_000,
            sigma_start=0,
            splitt_large_triangles=100,
            start_upsampling=20_000,
            upscaling_factor=2,
            size_probs_zero=7.5e-05,
            size_probs_zero_image_space=0.0,
            prune_size=1400,
            lambda_vertex=0.00025,
            max_diff_threshold=0.5,
            start_vertex_opt=12_000,
            lamba_depth=0.05,
            depth_lambda_init=0.01,
            depth_lambda_final=0.001,
        )
        opt_defaults["iterations"] = self.max_steps
        if self.indoor:
            opt_defaults.update(
                add_percentage=1.27,
                densify_from_iter=1_000,
                densify_until_iter=10_000,
                feature_lr=0.004,
                size_probs_zero=0.0,
                splitt_large_triangles=500,
                start_pruning=3_000,
                weight_lr=0.05,
                lambda_weight=0.0,
                lambda_normals=0.00001,
                lambda_normals_super=0.01,
                prune_size=1300,
                lambda_vertex=0.00025,
                depth_lambda_init=0.0,
                depth_lambda_final=0.0,
                iteration_mesh=12_000,
            )
        opt_defaults.update(self.extra_args)
        max_primitives = opt_defaults.get("max_primitives")
        if max_primitives is not None:
            cap = int(max_primitives)
            if cap <= 0:
                raise ValueError(f"max_primitives must be positive, got {cap!r}")
            opt_defaults["max_primitives"] = cap
        opt_defaults["iterations"] = self.max_steps
        self._opt = SimpleNamespace(**opt_defaults)

        self._pipe = SimpleNamespace(
            debug=False,
            convert_SHs_python=False,
            compute_cov3D_python=False,
            depth_ratio=float(self.extra_args.get("depth_ratio", 1.0)),
        )
        self._depth_l1_weight = get_expon_lr_func(
            self._opt.depth_lambda_init,
            self._opt.depth_lambda_final,
            max_steps=self.max_steps,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_args = self._build_dataset_args()
        (self.output_dir / "cfg_args").write_text(
            str(SimpleNamespace(**vars(dataset_args))), encoding="utf-8"
        )

        self._model = TriangleModel(dataset_args.sh_degree)
        self._scene = _MeshSplattingScene(
            dataset_args,
            self._model,
            self._opt.set_weight,
            self._opt.set_sigma,
            shuffle=bool(self.extra_args.get("shuffle", True)),
        )
        self._model.training_setup(
            self._opt,
            self._opt.feature_lr,
            self._opt.weight_lr,
            self._opt.lr_triangles_points_init,
        )
        self._model.add_percentage = self._opt.add_percentage
        self._model.size_probs_zero = self._opt.size_probs_zero
        self._model.size_probs_zero_image_space = self._opt.size_probs_zero_image_space
        if self._opt.max_primitives is not None:
            self._model.enforce_max_primitives(self._opt.max_primitives)
        self._optimizer = self._model.optimizer

        self._train_cameras = self._scene.getTrainCameras().copy()
        self._test_cameras = self._scene.getTestCameras().copy()
        self._viewpoint_stack = self._train_cameras.copy()
        self._prune_threshold = float(self._opt.prune_triangles_threshold)

        bg_val = 1.0 if self.white_background else 0.0
        self._bg_color = torch.tensor([bg_val] * 3, dtype=torch.float32, device="cuda")

    def _update_sigma_and_sampling(self, iteration: int) -> None:
        opt = self._opt
        if self._need_delaunay:
            with torch.no_grad():
                self._model.run_restricted_delaunay()
                if self._opt.max_primitives is not None:
                    self._model.enforce_max_primitives(self._opt.max_primitives)
            self._need_delaunay = False

        if iteration == opt.start_upsampling:
            self._model.scaling = opt.upscaling_factor
        if iteration == opt.start_upsampling + 5000:
            self._model.scaling = 4

        self._model.update_learning_rate(iteration)

        initial_sigma = opt.set_sigma
        final_sigma = 0.0001
        if iteration < opt.sigma_start:
            current_sigma = initial_sigma
        else:
            denom = max(1, opt.sigma_until - opt.sigma_start)
            progress = min((iteration - opt.sigma_start) / denom, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        self._model.set_sigma(current_sigma)

        if iteration % 1000 == 0:
            self._model.oneupSHdegree()

    def sample_batch(self) -> tuple[CameraBatch, torch.Tensor]:
        self._ensure_initialized()
        self._update_sigma_and_sampling(self._step)

        if not self._viewpoint_stack or len(self._train_cameras) + self._step == self.max_steps:
            self._viewpoint_stack = self._train_cameras.copy()
            if len(self._train_cameras) + self._step == self.max_steps:
                self._model.importance_score = torch.zeros(
                    (self._model._triangle_indices.shape[0]),
                    dtype=torch.float,
                    device="cuda",
                )

        stack_pos = random.randint(0, len(self._viewpoint_stack) - 1)
        cam = self._viewpoint_stack.pop(stack_pos)
        gt = cam.original_image.permute(1, 2, 0).contiguous()
        camera = CameraBatch(
            viewmats=torch.eye(4).unsqueeze(0),
            camtoworlds=torch.eye(4).unsqueeze(0),
            Ks=torch.eye(3).unsqueeze(0),
            width=cam.image_width,
            height=cam.image_height,
            metadata={"native_cam": cam},
        )
        return camera, gt.unsqueeze(0)

    def render_train(self, cameras: CameraBatch) -> RenderOutput:
        self._ensure_initialized()
        native_cam = cameras.metadata["native_cam"]
        self._last_native_cam = native_cam
        bg = (
            torch.rand((3), device="cuda")
            if self._opt.random_background
            else self._bg_color
        )
        rendering = self._render(native_cam, self._model, self._pipe, bg)

        with torch.no_grad():
            image_size = rendering["scaling"].detach()
            mask = image_size > self._model.image_size
            self._model.image_size[mask] = image_size[mask]

            importance_score = rendering["max_blending"].detach()
            mask = importance_score > self._model.importance_score
            self._model.importance_score[mask] = importance_score[mask]

            pixel_count = rendering["triangle_was_rendered"].detach()
            mask = pixel_count > self._model.pixel_count
            self._model.pixel_count[mask] = pixel_count[mask]

        rgb = rendering["render"].permute(1, 2, 0).contiguous()
        return RenderOutput(
            rgb=rgb,
            extras={
                "expected_depth": rendering.get("expected_depth"),
                "image_2D": rendering.get("image_2D"),
                "rend_normal": rendering.get("rend_normal"),
                "surf_depth": rendering.get("surf_depth"),
                "surf_normal": rendering.get("surf_normal"),
                "vertex_depth_out": rendering.get("vertex_depth_out"),
                "vertex_rendered": rendering.get("vertex_rendered"),
            },
        )

    def compute_loss(
        self, output: RenderOutput, gt_images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        from tribench.vendor.mesh_splatting.utils.loss_utils import (
            l1_loss,
            ssim,
            vertex_depth_loss_hr,
        )

        native_cam = output.extras.get("native_cam") if output.extras else None
        pred = output.rgb.permute(2, 0, 1).contiguous()
        gt = gt_images.squeeze(0).permute(2, 0, 1).contiguous()
        pixel_loss = l1_loss(pred, gt)
        ssim_value = ssim(pred, gt)
        image_loss = (1.0 - self._opt.lambda_dssim) * pixel_loss + self._opt.lambda_dssim * (
            1.0 - ssim_value
        )
        total = image_loss
        losses = {"pixel": pixel_loss, "image": image_loss}

        iteration = self._step
        vertex_weights = self._model.get_vertex_weight[
            : self._model.vertices.shape[0]
        ][self._model._triangle_indices.long()]
        lambda_weight = self._opt.lambda_weight if iteration < self._opt.start_opacity_floor else 0
        if lambda_weight > 0:
            weight_loss = lambda_weight * vertex_weights.mean()
            total = total + weight_loss
            losses["weight"] = weight_loss

        extras = output.extras or {}
        native_cam = getattr(self, "_last_native_cam", None)
        lambda_vertex = self._opt.lambda_vertex if iteration > self._opt.start_vertex_opt else 0
        if lambda_vertex > 0 and all(
            extras.get(key) is not None
            for key in ("surf_depth", "vertex_depth_out", "image_2D", "vertex_rendered")
        ):
            vertex_depth = vertex_depth_loss_hr(
                extras["vertex_depth_out"],
                extras["image_2D"],
                extras["vertex_rendered"],
                extras["surf_depth"],
                max_diff_threshold=self._opt.max_diff_threshold,
            )
            vertex_loss = lambda_vertex * vertex_depth
            total = total + vertex_loss
            losses["vertex_depth"] = vertex_loss

        if (
            self._depth_l1_weight is not None
            and native_cam is not None
            and self._depth_l1_weight(iteration) > 0
            and getattr(native_cam, "invdepthmap", None) is not None
            and extras.get("expected_depth") is not None
        ):
            inv_depth = 1.0 / (extras["expected_depth"] + 1e-6)
            mono_invdepth = native_cam.invdepthmap.cuda()
            depth_mask = native_cam.depth_mask.cuda()
            depth_loss = (
                self._depth_l1_weight(iteration)
                * torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
            )
            total = total + depth_loss
            losses["depth"] = depth_loss

        rend_normal = extras.get("rend_normal")
        surf_normal = extras.get("surf_normal")
        lambda_normal = self._opt.lambda_normals if iteration > self._opt.iteration_mesh else 0
        if lambda_normal > 0 and rend_normal is not None and surf_normal is not None:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * normal_error.mean()
            total = total + normal_loss
            losses["normal"] = normal_loss

        if native_cam is not None and getattr(native_cam, "normal_map", None) is not None:
            lambda_super = (
                self._opt.lambda_normals_super
                if iteration > self._opt.iteration_mesh
                else 0
            )
            if lambda_super > 0 and rend_normal is not None:
                gt_normal = native_cam.normal_map.cuda()
                seg_hr = gt_normal.unsqueeze(0)
                gt_normal = F.interpolate(
                    seg_hr,
                    size=(gt.shape[1], gt.shape[2]),
                    mode="area",
                ).squeeze(0)
                normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
                supervised_normal = lambda_super * normal_error.mean()
                total = total + supervised_normal
                losses["normal_supervised"] = supervised_normal

        losses["total"] = total
        return losses

    def optimizer_step(self) -> None:
        self._ensure_initialized()
        if self._step < self.max_steps:
            self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

    def _prune_unused_vertices(self) -> None:
        device = self._model.vertices.device
        used_vertex_mask = torch.zeros(
            self._model.vertices.shape[0], dtype=torch.bool, device=device
        )
        if self._model._triangle_indices.numel() > 0:
            used_vertex_mask[self._model._triangle_indices.flatten().long()] = True
        weight_mask = (
            self._model.get_vertex_weight.squeeze() >= self._prune_threshold
        )
        vertex_mask = weight_mask[: self._model.vertices.shape[0]] | used_vertex_mask
        self._model._prune_vertices(vertex_mask)

    def update_structure(self, step: int) -> dict[str, Any] | None:
        self._ensure_initialized()
        opt = self._opt
        before = int(self._model._triangle_indices.shape[0])
        update_type: list[str] = []
        run_restricted_delaunay = (
            opt.run_restricted_delaunay
            if opt.run_restricted_delaunay >= 0
            else opt.densify_until_iter + 1000
        )

        with torch.no_grad():
            if step % 500 == 0 and step < run_restricted_delaunay:
                triangle_vertex_weights = self._model.opacity_activation(
                    self._model.vertex_weight[self._model._triangle_indices.long()]
                )
                min_weights = triangle_vertex_weights.min(dim=1).values
                mask_opacity = (min_weights <= self._prune_threshold).squeeze()
                mask_importance = (
                    self._model.importance_score <= self._prune_threshold
                ).squeeze()
                mask_size = (self._model.image_size > opt.prune_size).squeeze()
                delete_mask = mask_opacity | mask_size
                if len(self._train_cameras) < 500:
                    delete_mask = delete_mask | mask_importance

                if step > opt.start_pruning:
                    self._model.prune_triangles(~delete_mask)
                    update_type.append("prune")

                self._prune_unused_vertices()

                current_primitives = int(self._model._triangle_indices.shape[0])
                max_split_candidates = None
                if opt.max_primitives is not None:
                    primitive_budget = int(opt.max_primitives) - current_primitives
                    max_split_candidates = max(0, primitive_budget // 3)

                needs_densification = (
                    step < opt.densify_until_iter
                    and step % opt.densification_interval == 0
                    and step > opt.densify_from_iter
                    and self._model.vertices.shape[0] < opt.max_points
                    and (opt.max_primitives is None or max_split_candidates > 0)
                    and self._model.importance_score.numel() > 0
                    and torch.sum(self._model.importance_score) > 0
                )
                if needs_densification:
                    self._model.add_new_gs(
                        step,
                        cap_max=opt.max_points,
                        splitt_large_triangles=opt.splitt_large_triangles,
                        max_split_candidates=max_split_candidates,
                    )
                    if opt.max_primitives is not None:
                        self._model.enforce_max_primitives(opt.max_primitives)
                    update_type.append("densify")

                if step > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = opt.final_opacity_iter
                    ratio = min(
                        1.0,
                        max(0.0, (step - start_iter) / max(1, end_iter - start_iter)),
                    )
                    current_opacity = min(0.1 + (0.9999 - 0.1) * ratio, 0.9999)
                    self._model.update_min_weight(current_opacity)
                    self._prune_threshold += 0.01

            elif step == run_restricted_delaunay:
                self._need_delaunay = True
                update_type.append("schedule-delaunay")

            elif step % 500 == 0 and step > run_restricted_delaunay + 1000:
                if step > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = opt.final_opacity_iter
                    ratio = min(
                        1.0,
                        max(0.0, (step - start_iter) / max(1, end_iter - start_iter)),
                    )
                    current_opacity = min(0.1 + (0.9999 - 0.1) * ratio, 0.9999)
                    self._model.update_min_weight(current_opacity)
                    self._prune_threshold += 0.01
                    update_type.append("opacity-floor")

        after = int(self._model._triangle_indices.shape[0])
        self._optimizer = self._model.optimizer
        if not update_type and before == after:
            return None
        return {
            "type": "+".join(update_type) if update_type else "structure",
            "triangles_before": before,
            "triangles_after": after,
            "delta": after - before,
            "vertices": int(self._model.vertices.shape[0]),
            "max_points": int(opt.max_points),
            "max_primitives": None if opt.max_primitives is None else int(opt.max_primitives),
        }

    def _final_cleanup(self) -> None:
        if self._final_cleaned or not self.extra_args.get("final_cleanup", True):
            return
        self._final_cleaned = True
        if not self._train_cameras:
            return

        bg = self._bg_color
        self._model.importance_score = torch.zeros(
            (self._model._triangle_indices.shape[0]), dtype=torch.float, device="cuda"
        )
        for viewpoint_cam in self._train_cameras:
            render_pkg = self._render(viewpoint_cam, self._model, self._pipe, bg)
            importance_score = render_pkg["max_blending"].detach()
            mask = importance_score > self._model.importance_score
            self._model.importance_score[mask] = importance_score[mask]

        mask_importance = (self._model.importance_score <= 0.5).squeeze()
        self._model.prune_triangles(~mask_importance)
        device = self._model.vertices.device
        used_vertex_mask = torch.zeros(
            self._model.vertices.shape[0], dtype=torch.bool, device=device
        )
        if self._model._triangle_indices.numel() > 0:
            used_vertex_mask[self._model._triangle_indices.flatten().long()] = True
        self._model._prune_vertices(used_vertex_mask)
        self._optimizer = self._model.optimizer

    def on_step_end(self, step: int, losses: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        self._ensure_initialized()
        if epoch >= self.max_steps:
            self._final_cleanup()
        if self._opt.max_primitives is not None:
            self._model.enforce_max_primitives(self._opt.max_primitives)
            self._optimizer = self._model.optimizer
        if self._scene is not None:
            self._scene.save(epoch)
        metadata = {
            "white_background": self.white_background,
            "background_color": self._bg_color.detach().cpu().tolist(),
            "indoor": self.indoor,
            "images": self.images_dir,
            "resolution": self.resolution,
        }
        ckpt_dir = self.output_dir / "point_cloud" / f"iteration_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "tribench_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def resume_from_checkpoint(self, output_dir: str | Path, max_steps: int) -> int:
        latest = find_latest_point_cloud_checkpoint(output_dir)
        if latest is None:
            return 0
        step, ckpt_dir = latest
        if step <= 0:
            return 0

        self._ensure_initialized()
        self._model.load_parameters(str(ckpt_dir))
        self._model.training_setup(
            self._opt,
            self._opt.feature_lr,
            self._opt.weight_lr,
            self._opt.lr_triangles_points_init,
        )
        self._model.add_percentage = self._opt.add_percentage
        self._model.size_probs_zero = self._opt.size_probs_zero
        self._model.size_probs_zero_image_space = self._opt.size_probs_zero_image_space
        if self._opt.max_primitives is not None:
            self._model.enforce_max_primitives(self._opt.max_primitives)
        self._optimizer = self._model.optimizer
        self._viewpoint_stack = self._train_cameras.copy()
        self.set_step(step)
        print(f"Loaded checkpoint: {ckpt_dir}")
        return step

    def get_lr(self) -> dict[str, float]:
        if self._optimizer is None:
            return {}
        return {
            pg["name"]: pg["lr"]
            for pg in self._optimizer.param_groups
            if "name" in pg
        }
