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
from typing import Any

import torch

from trianglebench.core.cameras import CameraBatch
from trianglebench.renderers.base import RenderOutput
from trianglebench.trainers.hooks import TrainingMethod
from trianglebench.trainers.registry import register_training_method


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

        # Lazy-initialized state
        self._model = None
        self._optimizer = None
        self._scene_cameras = None  # (train_cameras, test_cameras)
        self._bg_color: torch.Tensor | None = None
        self._opt = None
        self._viewpoint_stack: list[int] = []
        self._new_round = False
        self._removed_them = False
        self._opacity_now = True
        self._last_structure_update: dict[str, Any] | None = None
        self._initialized = False
        self._step = 0

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

        from trianglebench.vendor.triangle_splatting.scene.triangle_model import (
            TriangleModel,
        )
        from trianglebench.vendor.triangle_splatting.scene.colmap_loader import (
            read_extrinsics_binary,
            read_intrinsics_binary,
            read_points3D_binary,
            read_points3D_text,
        )
        from trianglebench.vendor.triangle_splatting.utils.graphics_utils import (
            BasicPointCloud,
            getWorld2View2,
            getProjectionMatrix,
        )
        from trianglebench.vendor.triangle_splatting.triangle_renderer import (
            render as ts_render,
        )

        self._ts_render = ts_render
        self._getWorld2View2 = getWorld2View2
        self._getProjectionMatrix = getProjectionMatrix

        # -- Load scene cameras (COLMAP) ------------------------------
        sparse_dir = self.dataset_path / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = self.dataset_path / "sparse"

        try:
            extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
            intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        except Exception:
            from trianglebench.vendor.triangle_splatting.scene.colmap_loader import (
                read_extrinsics_text,
                read_intrinsics_text,
            )
            extrinsics = read_extrinsics_text(str(sparse_dir / "images.txt"))
            intrinsics = read_intrinsics_text(str(sparse_dir / "cameras.txt"))

        from types import SimpleNamespace
        from PIL import Image
        import numpy as np

        image_names = sorted(extrinsics.keys(), key=lambda k: extrinsics[k].name)
        all_cams = []
        all_images = []
        for k in image_names:
            im = extrinsics[k]
            cam = intrinsics[im.camera_id]
            colmap_w, colmap_h = int(cam.width), int(cam.height)
            params = np.asarray(cam.params, dtype=np.float32)

            if cam.model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            fovx = 2 * np.arctan(colmap_w / (2 * fx))
            fovy = 2 * np.arctan(colmap_h / (2 * fy))

            img_path = self.dataset_path / self.images_dir / Path(im.name).name
            pil = Image.open(img_path) if img_path.exists() else None
            image_w, image_h = pil.size if pil is not None else (colmap_w, colmap_h)
            w, h, _, _ = self._resolve_resolution(image_w, image_h)

            # Native triangle-splatting stores Camera.R as qvec2rotmat().T.
            # getWorld2View2() transposes it back internally to the COLMAP W2C
            # rotation; passing qvec2rotmat() directly flips the camera frame.
            R = im.qvec2rotmat().T
            t = np.array(im.tvec, dtype=np.float32)

            wvt = torch.from_numpy(
                self._getWorld2View2(R, t, np.array([0.0, 0.0, 0.0]), 1.0).T
            ).float().cuda()
            znear, zfar = 0.01, 100.0
            proj = self._getProjectionMatrix(znear, zfar, fovx, fovy).T.cuda()
            full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

            # Build SimpleNamespace matching native Camera
            native_cam = SimpleNamespace(
                image_width=w,
                image_height=h,
                FoVx=fovx,
                FoVy=fovy,
                world_view_transform=wvt,
                full_proj_transform=full_proj,
                camera_center=wvt.inverse()[3, :3],
                uid=k,
            )
            all_cams.append(native_cam)

            # Load image
            if pil is not None:
                if pil.size != (w, h):
                    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
                arr = np.asarray(pil)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                gt = torch.from_numpy(arr[..., :3].copy())
                pil.close()
            else:
                gt = torch.zeros(h, w, 3, dtype=torch.uint8)
            all_images.append(gt)

        # Train/test split
        if self.eval_split:
            train_idx = [i for i in range(len(all_cams)) if i % self.llffhold != 0]
            test_idx = [i for i in range(len(all_cams)) if i % self.llffhold == 0]
        else:
            train_idx = list(range(len(all_cams)))
            test_idx = []

        self._train_cameras = [all_cams[i] for i in train_idx]
        self._train_images = [all_images[i] for i in train_idx]
        self._test_cameras = [all_cams[i] for i in test_idx]

        # -- Load COLMAP 3D points ----------------------------------------
        try:
            pts_xyz, pts_rgb, _ = read_points3D_binary(str(sparse_dir / "points3D.bin"))
        except Exception:
            try:
                pts_xyz, pts_rgb, _ = read_points3D_binary(str(sparse_dir / "points3D.bin").replace("points3D.bin", "points3D.BIN"))
            except Exception:
                pts_xyz, pts_rgb, _ = read_points3D_text(str(sparse_dir / "points3D.txt"))
        # Normalize colors to [0, 1]
        pts_rgb = pts_rgb.astype(np.float32) / 255.0

        pcd = BasicPointCloud(points=pts_xyz, colors=pts_rgb, normals=np.zeros_like(pts_xyz))

        # -- Compute cameras_extent (nerf normalization radius) -----------
        cam_centers = []
        extent_image_names = [image_names[i] for i in train_idx]
        for k in extent_image_names:
            R = extrinsics[k].qvec2rotmat().T
            t = np.array(extrinsics[k].tvec, dtype=np.float32)
            W2C = getWorld2View2(R, t)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        cam_centers_np = np.hstack(cam_centers)
        avg_center = np.mean(cam_centers_np, axis=1, keepdims=True)
        dist = np.linalg.norm(cam_centers_np - avg_center, axis=0, keepdims=True)
        cameras_extent = float(np.max(dist)) * 1.1

        # -- Build model via create_from_pcd ------------------------------
        sh_degree = 3
        self._model = TriangleModel(sh_degree=sh_degree)
        self._model.create_from_pcd(
            pcd,
            spatial_lr_scale=cameras_extent,
            opacity=0.28,
            init_size=2.23,
            nb_points=3,
            set_sigma=1.16,
            no_dome=False,
        )

        # -- Optimizer via training_setup ---------------------------------
        opt_defaults = dict(
            split_size=24.0,
            start_lr_sigma=0,
            max_noise_factor=1.5,
            add_shape=1.3,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=max(self.max_steps, 30_000),
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
            outdoor=False,
        )
        opt_defaults.update(self.extra_args)
        self._opt = SimpleNamespace(**opt_defaults)
        self._model.training_setup(
            self._opt,
            lr_mask=self._opt.lr_mask,
            lr_features=self._opt.feature_lr,
            lr_opacity=self._opt.opacity_lr,
            lr_sigma=self._opt.lr_sigma,
            lr_triangles_points_init=self._opt.lr_triangles_points_init,
        )
        self._optimizer = self._model.optimizer
        self._viewpoint_stack = list(range(len(self._train_cameras)))

        # -- Background color ---------------------------------------------
        bg_val = 1.0 if self.white_background else 0.0
        self._bg_color = torch.tensor([bg_val] * 3, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------------
    # TrainingMethod hooks
    # ------------------------------------------------------------------

    def sample_batch(self) -> tuple[CameraBatch, torch.Tensor]:
        self._ensure_initialized()
        if not self._viewpoint_stack:
            self._viewpoint_stack = list(range(len(self._train_cameras)))
            if not self._new_round and self._removed_them:
                self._new_round = True
                self._removed_them = False
            else:
                self._new_round = False

        stack_pos = random.randint(0, len(self._viewpoint_stack) - 1)
        idx = self._viewpoint_stack.pop(stack_pos)
        cam = self._train_cameras[idx]
        gt = self._train_images[idx].to(device="cuda", dtype=torch.float32).div_(255.0)

        # Build a CameraBatch wrapping this single camera
        import numpy as np

        w2c = np.linalg.inv(
            np.vstack([
                np.hstack([
                    cam.world_view_transform.T.cpu().numpy()[:3, :3],
                    cam.camera_center.cpu().numpy().reshape(3, 1),
                ]),
                [0, 0, 0, 1],
            ])
        )[:3, :4]
        viewmat = torch.eye(4).unsqueeze(0)
        viewmat[0, :3, :4] = torch.from_numpy(
            np.hstack([
                cam.world_view_transform.T.cpu().numpy()[:3, :3],
                np.array([[0.0], [0.0], [0.0]]),
            ])
        )
        # Simplified: just use identity-ish matrices and rely on native cam
        camera = CameraBatch(
            viewmats=torch.eye(4).unsqueeze(0),
            camtoworlds=torch.eye(4).unsqueeze(0),
            Ks=torch.eye(3).unsqueeze(0),
            width=cam.image_width,
            height=cam.image_height,
            metadata={"native_cam": cam, "idx": idx},
        )
        return camera, gt.unsqueeze(0)

    def render_train(self, cameras: CameraBatch) -> RenderOutput:
        self._ensure_initialized()
        native_cam = cameras.metadata["native_cam"]
        from types import SimpleNamespace

        iteration = self._step + 1
        if iteration % 1000 == 0:
            self._model.oneupSHdegree()

        pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=0.0)
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
        from trianglebench.vendor.triangle_splatting.utils.loss_utils import (
            l1_loss,
            l2_loss,
            ssim,
        )

        pred = output.rgb.permute(2, 0, 1).contiguous()
        gt = gt_images.squeeze(0).permute(2, 0, 1).contiguous()

        opt = self._opt
        iteration = self._step + 1
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
            from trianglebench.vendor.triangle_splatting.utils.loss_utils import (
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
        iteration = self._step + 1
        self._model.update_learning_rate(iteration)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._step += 1

    def update_structure(self, step: int) -> dict[str, Any] | None:
        self._ensure_initialized()
        opt = self._opt
        before = int(self._model.get_triangles_points.shape[0])
        update_type = None

        with torch.no_grad():
            if (
                step < opt.densify_until_iter
                and step % opt.densification_interval == 0
                and step > opt.densify_from_iter
            ):
                if len(self._train_cameras) < 250 or not self._new_round:
                    dead_mask = torch.logical_or(
                        (
                            self._model.importance_score < opt.importance_threshold
                        ).squeeze(),
                        (self._model.get_opacity <= opt.opacity_dead).squeeze(),
                    )
                else:
                    dead_mask = (self._model.get_opacity <= opt.opacity_dead).squeeze()

                if step > 1000 and not self._new_round:
                    dead_mask = torch.logical_or(
                        dead_mask, (self._model.triangle_area < 2).squeeze()
                    )
                    if not opt.outdoor:
                        dead_mask = torch.logical_or(
                            dead_mask, (self._model.image_size > 1400).squeeze()
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
                self._removed_them = True
                self._new_round = False
                update_type = "densify"

            elif step > opt.densify_until_iter and step % opt.densification_interval == 0:
                if len(self._train_cameras) < 250 or not self._new_round:
                    dead_mask = torch.logical_or(
                        (
                            self._model.importance_score < opt.importance_threshold
                        ).squeeze(),
                        (self._model.get_opacity <= opt.opacity_dead).squeeze(),
                    )
                else:
                    dead_mask = (self._model.get_opacity <= opt.opacity_dead).squeeze()

                if not self._new_round:
                    dead_mask = torch.logical_or(
                        dead_mask, (self._model.triangle_area < 2).squeeze()
                    )

                self._model.remove_final_points(dead_mask)
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
        }
        return self._last_structure_update

    def on_step_end(self, step: int, losses: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Save checkpoint at evaluation intervals."""
        self._ensure_initialized()
        ckpt_dir = self.output_dir / "point_cloud" / f"iteration_{epoch * 1000}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # TriangleModel.save() expects a directory path and creates
        # point_cloud_state_dict.pt + hyperparameters.pt inside it.
        self._model.save(str(ckpt_dir))
        metadata = {
            "white_background": self.white_background,
            "background_color": self._bg_color.detach().cpu().tolist(),
        }
        (ckpt_dir / "trianglebench_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def get_lr(self) -> dict[str, float]:
        if self._optimizer is None:
            return {}
        return {
            pg["name"]: pg["lr"]
            for pg in self._optimizer.param_groups
            if "name" in pg
        }
