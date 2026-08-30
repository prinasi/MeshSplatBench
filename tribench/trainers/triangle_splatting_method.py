"""Triangle-splatting training method for the generic TrainingLoop.

This wraps the native triangle-splatting training pipeline (model,
optimizer, densification, scene cameras) behind the ``TrainingMethod``
hook interface so that the generic ``TrainingLoop`` orchestrator can
drive it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from tribench.core.cameras import CameraBatch
from tribench.renderers.base import RenderOutput
from tribench.trainers.checkpoints import find_latest_point_cloud_checkpoint
from tribench.trainers.hooks import TrainingMethod
from tribench.trainers.registry import register_training_method


def _pil_rgb_uint8(
    image,
    size: tuple[int, int] | None = None,
    *,
    resample=None,
):
    """Match native triangle-splatting RGB loading while ignoring alpha."""
    import numpy as np

    bands = image.getbands()
    if len(bands) >= 3 and bands[:3] == ("R", "G", "B"):
        channels = image.split()[:3]
        if size is not None and image.size != size:
            channels = tuple(channel.resize(size, resample) for channel in channels)
        return np.stack([np.asarray(channel) for channel in channels], axis=-1)
    if len(bands) >= 1:
        gray = image.split()[0]
        if size is not None and image.size != size:
            gray = gray.resize(size, resample)
        return np.repeat(np.asarray(gray)[..., None], 3, axis=-1)
    rgb = image.convert("RGB")
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, resample)
    return np.asarray(rgb)


def _seed_native_state(seed: int) -> None:
    """Match triangle-splatting's deterministic training state."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device("cuda:0"))


@register_training_method("triangle-splatting")
class TriangleSplattingTrainingMethod(TrainingMethod):
    """Training method implementation for the triangle-splatting backend.

    Parameters
    ----------
    dataset:
        Path to the dataset root directory.
    output_dir:
        Directory for checkpoints and logs.
    max_steps:
        Maximum number of training iterations.
    images:
        Image subfolder name inside the dataset root.
    resolution:
        Image resolution control matching the native triangle-splatting
        loader: ``1/2/4/8`` are downscale factors, other positive values
        are target widths, and ``-1`` caps very large images at 1600 px.
    eval_split:
        Whether to hold out every 8th image for testing.
    white_background:
        Use white instead of black background.
    dtu_eval_mode:
        DTU image loading mode. ``foreground`` composites PNG alpha
        consistently with a white DTU foreground benchmark target.
    extra_args:
        Additional keyword arguments forwarded to the native training
        configuration (e.g. ``densify_until_iter``, ``test_iterations``).
    """

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
        self.extra_args = extra_args or {}
        self.dtu_eval_mode = str(self.extra_args.get("dtu_eval_mode", dtu_eval_mode)).lower()

        # Lazy-initialized state
        self._model = None
        self._optimizer = None
        self._scene = None
        self._bg_color: torch.Tensor | None = None
        self._opt = None
        self._viewpoint_stack: list[Any] = []
        self._new_round = False
        self._removed_them = False
        self._opacity_now = True
        self._last_structure_update: dict[str, Any] | None = None
        self._initialized = False
        self._step = 0

    def _build_dataset_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            sh_degree=3,
            source_path=str(self.dataset_path),
            model_path=str(self.output_dir),
            images=self.images_dir,
            resolution=self.resolution,
            white_background=self.white_background,
            dtu_eval_mode=self.dtu_eval_mode,
            data_device="cuda",
            eval=self.eval_split,
        )

    def _resolve_resolution(self, width: int, height: int) -> tuple[int, int, float, float]:
        """Match the native triangle-splatting resolution argument semantics."""
        if self.resolution in {1, 2, 4, 8}:
            new_width = round(width / self.resolution)
            new_height = round(height / self.resolution)
        else:
            if self.resolution == -1:
                global_down = width / 1600 if width > 1600 else 1
            else:
                global_down = width / self.resolution
            new_width = int(width / global_down)
            new_height = int(height / global_down)

        new_width = max(1, new_width)
        new_height = max(1, new_height)
        return new_width, new_height, new_width / width, new_height / height

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        seed = self.extra_args.get("seed", 0)
        if seed is not None:
            _seed_native_state(int(seed))

        from tribench.vendor.triangle_splatting.scene import Scene, TriangleModel
        from tribench.vendor.triangle_splatting.triangle_renderer import (
            render as ts_render,
        )
        from tribench.vendor.triangle_splatting.utils.graphics_utils import (
            getProjectionMatrix,
            getWorld2View2,
        )

        self._ts_render = ts_render
        self._getWorld2View2 = getWorld2View2
        self._getProjectionMatrix = getProjectionMatrix

        opt_defaults = dict(
            split_size=24.0,
            start_lr_sigma=0,
            max_noise_factor=1.5,
            add_shape=1.3,
            depth_ratio=1.0,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=max(self.max_steps, 30_000),
            set_opacity=0.28,
            triangle_size=2.23,
            nb_points=3,
            set_sigma=1.16,
            feature_lr=0.0025,
            opacity_lr=0.014,
            lambda_dssim=0.2,
            densification_interval=500,
            densify_from_iter=500,
            densify_until_iter=25_000,
            random_background=False,
            lr_mask=0.01,
            lambda_normals=0.0001,
            lambda_dist=0.0,
            lambda_opacity=0.0055,
            lambda_size=0.00000001,
            opacity_dead=0.014,
            importance_threshold=0.022,
            iteration_mesh=5000,
            lr_sigma=0.0008,
            lr_triangles_points_init=0.0018,
            proba_distr=2,
            max_shapes=3_000_000,
            no_dome=False,
            outdoor=False,
        )
        opt_defaults.update(self.extra_args)
        max_primitives = opt_defaults.get("max_primitives")
        if max_primitives is not None:
            cap = int(max_primitives)
            if cap <= 0:
                raise ValueError(f"max_primitives must be positive, got {cap!r}")
            opt_defaults["max_primitives"] = cap
            opt_defaults["max_shapes"] = min(int(opt_defaults.get("max_shapes", cap)), cap)
        self._opt = SimpleNamespace(**opt_defaults)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_args = self._build_dataset_args()
        self._model = TriangleModel(dataset_args.sh_degree)
        self._scene = Scene(
            dataset_args,
            self._model,
            self._opt.set_opacity,
            self._opt.triangle_size,
            self._opt.nb_points,
            self._opt.set_sigma,
            bool(self._opt.no_dome),
            shuffle=bool(self.extra_args.get("shuffle", True)),
        )

        self._train_cameras = self._scene.getTrainCameras().copy()
        self._test_cameras = self._scene.getTestCameras().copy()

        # -- Optimizer via training_setup ---------------------------------
        self._model.training_setup(
            self._opt,
            lr_mask=self._opt.lr_mask,
            lr_features=self._opt.feature_lr,
            lr_opacity=self._opt.opacity_lr,
            lr_sigma=self._opt.lr_sigma,
            lr_triangles_points_init=self._opt.lr_triangles_points_init,
        )
        self._optimizer = self._model.optimizer
        self._viewpoint_stack = self._train_cameras.copy()

        # -- Background color ---------------------------------------------
        bg_val = 1.0 if self.white_background else 0.0
        self._bg_color = torch.tensor([bg_val] * 3, dtype=torch.float32, device="cuda")
        self._enforce_primitive_cap()

    # ------------------------------------------------------------------
    # TrainingMethod hooks
    # ------------------------------------------------------------------

    def sample_batch(self) -> tuple[CameraBatch, torch.Tensor]:
        self._ensure_initialized()
        self._model.update_learning_rate(self._step)
        if not self._viewpoint_stack:
            self._viewpoint_stack = self._train_cameras.copy()
            if not self._new_round and self._removed_them:
                self._new_round = True
                self._removed_them = False
            else:
                self._new_round = False

        stack_pos = random.randint(0, len(self._viewpoint_stack) - 1)
        cam = self._viewpoint_stack.pop(stack_pos)
        gt = cam.original_image.permute(1, 2, 0).contiguous()

        # Build a CameraBatch wrapping this single camera
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
        from types import SimpleNamespace

        iteration = self._step
        if iteration % 1000 == 0:
            self._model.oneupSHdegree()

        pipe = SimpleNamespace(
            debug=False,
            convert_SHs_python=False,
            depth_ratio=float(self._opt.depth_ratio),
        )
        bg = (
            torch.rand((3), device="cuda")
            if self._opt.random_background
            else self._bg_color
        )
        rendering = self._ts_render(native_cam, self._model, pipe, bg)

        with torch.no_grad():
            triangle_area = rendering["density_factor"].detach()
            image_size = rendering["scaling"].detach()
            importance_score = rendering["max_blending"].detach()

            if self._new_round:
                mask = triangle_area > 1
                self._model.triangle_area[mask] += 1

            mask = image_size > self._model.image_size
            self._model.image_size[mask] = image_size[mask]
            mask = importance_score > self._model.importance_score
            self._model.importance_score[mask] = importance_score[mask]

        rgb = rendering["render"].permute(1, 2, 0)  # [H, W, 3]
        return RenderOutput(
            rgb=rgb,
            extras={
                "rend_dist": rendering.get("rend_dist"),
                "rend_normal": rendering.get("rend_normal"),
                "surf_normal": rendering.get("surf_normal"),
            },
        )

    def compute_loss(
        self, output: RenderOutput, gt_images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        from tribench.vendor.triangle_splatting.utils.loss_utils import (
            l1_loss,
            l2_loss,
            ssim,
        )

        pred = output.rgb.permute(2, 0, 1).contiguous()
        gt = gt_images.squeeze(0).permute(2, 0, 1).contiguous()

        opt = self._opt
        iteration = self._step
        pixel_loss = (
            l2_loss(pred, gt)
            if self._model.large and opt.outdoor
            else l1_loss(pred, gt)
        )
        image_loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (
            1.0 - ssim(pred, gt)
        )
        opacity_loss = torch.abs(self._model.get_opacity).mean() * opt.lambda_opacity
        total = image_loss + opacity_loss
        losses = {
            "pixel": pixel_loss,
            "image": image_loss,
            "opacity": opacity_loss,
        }

        if iteration > opt.iteration_mesh and output.extras:
            rend_dist = output.extras.get("rend_dist")
            rend_normal = output.extras.get("rend_normal")
            surf_normal = output.extras.get("surf_normal")
            if rend_dist is not None and opt.lambda_dist:
                dist_loss = opt.lambda_dist * rend_dist.mean()
                total = total + dist_loss
                losses["dist"] = dist_loss
            if rend_normal is not None and surf_normal is not None and opt.lambda_normals:
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = opt.lambda_normals * normal_error.mean()
                total = total + normal_loss
                losses["normal"] = normal_loss

        if iteration < opt.densify_until_iter:
            from tribench.vendor.triangle_splatting.utils.loss_utils import (
                equilateral_regularizer,
            )

            mean_area = equilateral_regularizer(
                self._model.get_triangles_points
            ).mean().clamp_min(torch.finfo(output.rgb.dtype).eps)
            size_loss = opt.lambda_size / mean_area
            total = total + size_loss
            losses["size"] = size_loss

        losses["total"] = total
        return losses

    def optimizer_step(self) -> None:
        self._ensure_initialized()
        self._enforce_primitive_cap()
        if self._step < self.max_steps:
            self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

    def _primitive_cap(self) -> int | None:
        cap = getattr(self._opt, "max_primitives", None)
        if cap is None:
            cap = getattr(self._opt, "max_shapes", None)
        if cap is None:
            return None
        cap = int(cap)
        return cap if cap > 0 else None

    def _enforce_primitive_cap(self) -> int:
        cap = self._primitive_cap()
        if cap is None or self._model is None or not hasattr(self._model, "enforce_max_primitives"):
            return 0
        removed = int(self._model.enforce_max_primitives(cap))
        self._optimizer = self._model.optimizer
        return removed

    def _build_dead_mask(
        self,
        *,
        include_area: bool,
        include_image_size: bool,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        opt = self._opt
        importance_dead = (
            self._model.importance_score < opt.importance_threshold
        ).squeeze()
        opacity_dead = (self._model.get_opacity <= opt.opacity_dead).squeeze()
        area_dead = (self._model.triangle_area < 2).squeeze()
        image_size_dead = (self._model.image_size > 1400).squeeze()

        if len(self._train_cameras) < 250 or not self._new_round:
            dead_mask = torch.logical_or(importance_dead, opacity_dead)
        else:
            dead_mask = opacity_dead

        if include_area and not self._new_round:
            dead_mask = torch.logical_or(dead_mask, area_dead)
            if include_image_size:
                dead_mask = torch.logical_or(dead_mask, image_size_dead)

        if dead_mask.all():
            dead_mask = opacity_dead
            if dead_mask.all():
                keep_count = max(1, int(0.1 * self._model._opacity.shape[0]))
                keep_idx = torch.topk(self._model.get_opacity.squeeze(), k=keep_count, largest=True).indices
                dead_mask[keep_idx] = False

        return dead_mask, {
            "dead_importance": int(importance_dead.sum().item()),
            "dead_opacity": int(opacity_dead.sum().item()),
            "dead_area": int(area_dead.sum().item()),
            "dead_image_size": int(image_size_dead.sum().item()),
            "dead_total": int(dead_mask.sum().item()),
        }

    def update_structure(self, step: int) -> dict[str, Any] | None:
        self._ensure_initialized()
        opt = self._opt
        before = int(self._model.get_triangles_points.shape[0])
        update_type = None
        dead_stats: dict[str, int] = {}

        with torch.no_grad():
            if (
                step < opt.densify_until_iter
                and step % opt.densification_interval == 0
                and step > opt.densify_from_iter
                and before < int(opt.max_shapes)
            ):
                dead_mask, dead_stats = self._build_dead_mask(
                    include_area=step > 1000,
                    include_image_size=not opt.outdoor,
                )

                if opt.proba_distr == 0:
                    odd_group = True
                elif opt.proba_distr == 1:
                    odd_group = False
                else:
                    odd_group = self._opacity_now
                    self._opacity_now = not self._opacity_now

                self._model.add_new_gs(
                    cap_max=opt.max_shapes,
                    oddGroup=odd_group,
                    dead_mask=dead_mask,
                )
                self._enforce_primitive_cap()
                self._removed_them = True
                self._new_round = False
                update_type = "densify"

            elif step > opt.densify_until_iter and step % opt.densification_interval == 0:
                dead_mask, dead_stats = self._build_dead_mask(
                    include_area=True,
                    include_image_size=False,
                )

                self._model.remove_final_points(dead_mask)
                self._enforce_primitive_cap()
                self._removed_them = True
                self._new_round = False
                update_type = "prune"

        if update_type is None:
            self._last_structure_update = None
            return None

        after = int(self._model.get_triangles_points.shape[0])
        self._optimizer = self._model.optimizer
        self._last_structure_update = {
            "type": update_type,
            "triangles_before": before,
            "triangles_after": after,
            "delta": after - before,
            "cap_max": int(opt.max_shapes),
            **dead_stats,
        }
        return self._last_structure_update

    def on_step_end(self, step: int, losses: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Save checkpoint at the given training step."""
        self._ensure_initialized()
        self._enforce_primitive_cap()
        ckpt_dir = self.output_dir / "point_cloud" / f"iteration_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # TriangleModel.save() expects a directory path and creates
        # point_cloud_state_dict.pt + hyperparameters.pt inside it.
        if self._scene is not None:
            self._scene.save(epoch)
        else:
            self._model.save(str(ckpt_dir))
        metadata = {
            "white_background": self.white_background,
            "background_color": self._bg_color.detach().cpu().tolist(),
        }
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
        self._model.load(str(ckpt_dir))
        self._model.training_setup(
            self._opt,
            lr_mask=self._opt.lr_mask,
            lr_features=self._opt.feature_lr,
            lr_opacity=self._opt.opacity_lr,
            lr_sigma=self._opt.lr_sigma,
            lr_triangles_points_init=self._opt.lr_triangles_points_init,
        )
        count = self._model.get_triangles_points.shape[0]
        self._model.max_radii2D = torch.zeros(count, device="cuda")
        self._model.max_density_factor = torch.zeros(count, device="cuda")
        self._model.max_scaling = torch.zeros(count, device="cuda")
        self._model.triangle_area = torch.zeros(count, dtype=torch.float, device="cuda")
        self._model.image_size = torch.zeros(count, dtype=torch.float, device="cuda")
        self._model.importance_score = torch.zeros(count, dtype=torch.float, device="cuda")
        self._enforce_primitive_cap()
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
