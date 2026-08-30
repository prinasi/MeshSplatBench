"""Native triangle-splatting training entry point for TriBench."""

from __future__ import annotations

import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint
from typing import Sequence

import lpips
import torch
from tqdm import tqdm

from tribench.vendor.triangle_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from tribench.vendor.triangle_splatting.scene import Scene, TriangleModel
from tribench.vendor.triangle_splatting.triangle_renderer import render
from tribench.vendor.triangle_splatting.utils.general_utils import safe_state
from tribench.vendor.triangle_splatting.utils.image_utils import psnr
from tribench.vendor.triangle_splatting.utils.loss_utils import (
    equilateral_regularizer,
    l1_loss,
    l2_loss,
    ssim,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def build_parser():
    parser = ArgumentParser(description="TriBench triangle-splatting trainer")
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    return parser, model_params, opt_params, pipe_params


def _prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        # TriBench unifies tensorboard output under a sibling logs/ folder when
        # the model path is the shared ckpt/ directory.
        log_dir = args.model_path
        if os.path.basename(os.path.normpath(args.model_path)) == "ckpt":
            log_dir = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)
    print("Tensorboard not available: not logging progress")
    return None



def _training_report(
    tb_writer,
    iteration,
    pixel_loss,
    loss,
    loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    render_func,
    render_args,
    lpips_fn,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/pixel_loss", pixel_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        if not config["cameras"]:
            continue

        pixel_loss_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        total_time = 0.0

        for idx, viewpoint in enumerate(config["cameras"]):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            image = torch.clamp(
                render_func(viewpoint, scene.triangles, *render_args)["render"],
                0.0,
                1.0,
            )
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)

            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if tb_writer and idx < 5:
                tb_writer.add_images(
                    f"{config['name']}_view_{viewpoint.image_name}/render",
                    image[None],
                    global_step=iteration,
                )
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(
                        f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                        gt_image[None],
                        global_step=iteration,
                    )
            pixel_loss_test += loss_fn(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips_fn(image, gt_image).mean().double()

        count = len(config["cameras"])
        pixel_loss_test /= count
        psnr_test /= count
        ssim_test /= count
        lpips_test /= count
        total_time /= count
        print(
            f"\n[ITER {iteration}] Evaluating {config['name']}: "
            f"L1 {pixel_loss_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test} "
            f"runtime {total_time:.3f} ms"
        )

        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", pixel_loss_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
        tb_writer.add_scalar("total_points", scene.triangles.get_triangles_points.shape[0], iteration)
    torch.cuda.empty_cache()


def training(
    dataset,
    opt,
    pipe,
    *,
    no_dome,
    outdoor,
    testing_iterations,
    save_iterations,
    checkpoint,
    load_iteration,
    debug_from,
    lpips_fn,
):
    first_iter = 0
    tb_writer = _prepare_output_and_logger(dataset)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        triangles,
        opt.set_opacity,
        opt.triangle_size,
        opt.nb_points,
        opt.set_sigma,
        no_dome,
        load_iteration=load_iteration,
    )
    triangles.training_setup(
        opt,
        opt.lr_mask,
        opt.feature_lr,
        opt.opacity_lr,
        opt.lr_sigma,
        opt.lr_triangles_points_init,
    )
    if scene.loaded_iter:
        first_iter = int(scene.loaded_iter)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    total_dead = 0
    opacity_now = True
    new_round = False
    removed_them = False
    loss_fn = l2_loss if triangles.large and outdoor else l1_loss
    max_primitives_value = getattr(opt, "max_primitives", -1)
    max_primitives = (
        int(max_primitives_value)
        if max_primitives_value and int(max_primitives_value) > 0
        else int(opt.max_shapes) if getattr(opt, "max_shapes", None) else None
    )
    if max_primitives is not None:
        opt.max_shapes = min(int(opt.max_shapes), max_primitives)
        triangles.enforce_max_primitives(max_primitives)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        triangles.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        triangle_area = render_pkg["density_factor"].detach()
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        gt_image = viewpoint_cam.original_image.cuda()
        if opt.foreground_training and viewpoint_cam.gt_alpha_mask is not None:
            fg_mask = viewpoint_cam.gt_alpha_mask.cuda()
            image = image * fg_mask
            gt_image = gt_image * fg_mask
        pixel_loss = loss_fn(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss_opacity = torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        dist_loss = lambda_dist * render_pkg["rend_dist"].mean()
        normal_error = (
            1 - (render_pkg["rend_normal"] * render_pkg["surf_normal"]).sum(dim=0)
        )[None]
        normal_loss = lambda_normal * normal_error.mean()
        loss_size = opt.lambda_size / equilateral_regularizer(
            triangles.get_triangles_points
        ).mean()

        loss = loss_image + loss_opacity + normal_loss + dist_loss
        if iteration < opt.densify_until_iter:
            loss = loss + loss_size

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            _training_report(
                tb_writer,
                iteration,
                pixel_loss,
                loss,
                loss_fn,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                lpips_fn,
            )
            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving Triangles")
                scene.save(iteration)
            if iteration % 1000 == 0:
                total_dead = 0

            if (
                iteration < opt.densify_until_iter
                and iteration % opt.densification_interval == 0
                and iteration > opt.densify_from_iter
            ):
                if number_of_views < 250 or not new_round:
                    dead_mask = torch.logical_or(
                        (triangles.importance_score < opt.importance_threshold).squeeze(),
                        (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                    )
                else:
                    dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                if iteration > 1000 and not new_round:
                    dead_mask = torch.logical_or(dead_mask, (triangles.triangle_area < 2).squeeze())
                    if not outdoor:
                        dead_mask = torch.logical_or(dead_mask, (triangles.image_size > 1400).squeeze())

                if dead_mask.all():
                    dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()
                    if dead_mask.all():
                        keep_count = max(1, int(0.1 * triangles._opacity.shape[0]))
                        keep_idx = torch.topk(triangles.get_opacity.squeeze(), k=keep_count, largest=True).indices
                        dead_mask[keep_idx] = False

                total_dead += dead_mask.sum()
                if opt.proba_distr == 0:
                    odd_group = True
                elif opt.proba_distr == 1:
                    odd_group = False
                else:
                    odd_group = opacity_now
                    opacity_now = not opacity_now

                removed_them = True
                new_round = False
                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=odd_group, dead_mask=dead_mask)
                if max_primitives is not None:
                    triangles.enforce_max_primitives(max_primitives)

            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                if number_of_views < 250 or not new_round:
                    dead_mask = torch.logical_or(
                        (triangles.importance_score < opt.importance_threshold).squeeze(),
                        (triangles.get_opacity <= opt.opacity_dead).squeeze(),
                    )
                else:
                    dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                if not new_round:
                    dead_mask = torch.logical_or(dead_mask, (triangles.triangle_area < 2).squeeze())
                if dead_mask.all():
                    dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()
                    if dead_mask.all():
                        keep_count = max(1, int(0.1 * triangles._opacity.shape[0]))
                        keep_idx = torch.topk(triangles.get_opacity.squeeze(), k=keep_count, largest=True).indices
                        dead_mask[keep_idx] = False
                triangles.remove_final_points(dead_mask)
                if max_primitives is not None:
                    triangles.enforce_max_primitives(max_primitives)
                removed_them = True
                new_round = False

            if iteration < opt.iterations:
                if max_primitives is not None and triangles._triangles_points.shape[0] > max_primitives:
                    triangles.enforce_max_primitives(max_primitives)
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    if max_primitives is not None:
        with torch.no_grad():
            triangles.enforce_max_primitives(max_primitives)
    print("Training is done")


def run_training(argv: Sequence[str] | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("triangle-splatting training requires a CUDA-enabled PyTorch install.")

    parser, model_params, opt_params, pipe_params = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.source_path = str(Path(args.source_path).expanduser())
    args.model_path = str(Path(args.model_path).expanduser())

    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        model_params.extract(args),
        opt_params.extract(args),
        pipe_params.extract(args),
        no_dome=args.no_dome,
        outdoor=args.outdoor,
        testing_iterations=args.test_iterations,
        save_iterations=args.save_iterations,
        checkpoint=args.start_checkpoint,
        load_iteration=args.load_iteration,
        debug_from=args.debug_from,
        lpips_fn=lpips_fn,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    run_training()
