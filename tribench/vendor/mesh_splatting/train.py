"""Native MeshSplatting training entry point for TriBench."""

from __future__ import annotations

import os
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Sequence

import lpips
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tribench.vendor.mesh_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    update_indoor,
)
from tribench.vendor.mesh_splatting.scene import Scene, TriangleModel
from tribench.vendor.mesh_splatting.triangle_renderer import render
from tribench.vendor.mesh_splatting.utils.general_utils import (
    get_expon_lr_func,
    safe_state,
)
from tribench.vendor.mesh_splatting.utils.image_utils import psnr
from tribench.vendor.mesh_splatting.utils.loss_utils import (
    l1_loss,
    ssim,
    vertex_depth_loss_hr,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
    print("Using fused SSIM for faster training.")
except Exception:
    FUSED_SSIM_AVAILABLE = False


def build_parser():
    parser = ArgumentParser(description="TriBench MeshSplatting trainer")
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--wandb_name", default="Test", type=str)
    parser.add_argument("--scene_name", default="Garden", type=str)
    parser.add_argument("--use_sparse_adam", action="store_true", default=True)
    parser.add_argument("--indoor", action="store_true", default=False)
    return parser, model_params, opt_params, pipe_params


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def training_report(
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
            runtime = start_event.elapsed_time(end_event)
            total_time += runtime

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
        psnr_test /= count
        pixel_loss_test /= count
        ssim_test /= count
        lpips_test /= count
        total_time /= count
        fps = 1000.0 / total_time
        print(
            f"\n[ITER {iteration}] Evaluating {config['name']}: "
            f"L1 {pixel_loss_test} PSNR {psnr_test} SSIM {ssim_test} "
            f"LPIPS {lpips_test} FPS {fps}"
        )

        if tb_writer:
            tb_writer.add_scalar(
                f"{config['name']}/loss_viewpoint - l1_loss",
                pixel_loss_test,
                iteration,
            )
            tb_writer.add_scalar(
                f"{config['name']}/loss_viewpoint - psnr",
                psnr_test,
                iteration,
            )

    torch.cuda.empty_cache()


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    checkpoint,
    debug_from,
    scene_name,
    *,
    load_iteration=None,
    lpips_fn,
    use_sparse_adam=False,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        triangles,
        opt.set_weight,
        opt.set_sigma,
        load_iteration=load_iteration,
    )
    triangles.training_setup(
        opt,
        opt.feature_lr,
        opt.weight_lr,
        opt.lr_triangles_points_init,
    )
    triangles.add_percentage = opt.add_percentage

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
    number_of_training_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until
    init_opacity = 0.1
    final_opacity = 0.9999
    total_iters_opacity = opt.final_opacity_iter
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    need_delaunay = False
    run_restricted_delaunay = opt.densify_until_iter + 1000
    depth_l1_weight = get_expon_lr_func(
        opt.depth_lambda_init,
        opt.depth_lambda_final,
        max_steps=opt.iterations,
    )

    last_iteration = first_iter - 1
    for iteration in range(first_iter, opt.iterations + 1):
        last_iteration = iteration
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False

        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)

        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros(
                    (triangles._triangle_indices.shape[0]),
                    dtype=torch.float,
                    device="cuda",
                )
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, "normal_map", None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            seg_hr = gt_normal.unsqueeze(0)
            seg_ds_area = F.interpolate(
                seg_hr,
                size=(gt_image.shape[1], gt_image.shape[2]),
                mode="area",
            )
            gt_normal = seg_ds_area.squeeze(0)
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg["scaling"].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        pixel_count = render_pkg["triangle_was_rendered"].detach()
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (
            1.0 - ssim_value
        )
        loss = loss_image

        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][
                triangles._triangle_indices
            ]
            loss = loss + lambda_weight * vertex_weights.mean()

        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            vertex_depth = vertex_depth_loss_hr(
                render_pkg["vertex_depth_out"],
                render_pkg["image_2D"],
                render_pkg["vertex_rendered"],
                render_pkg["surf_depth"],
                max_diff_threshold=opt.max_diff_threshold,
            )
            loss = loss + lambda_vertex * vertex_depth

        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "invdepthmap", None) is not None:
            inv_depth = 1.0 / (render_pkg["expected_depth"] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            depth_loss = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
            loss = loss + depth_l1_weight(iteration) * depth_loss

        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]

        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            loss = loss + lambda_normal * normal_error.mean()

        if gt_normal is not None:
            lambda_normals_super = (
                opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            )
            normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
            loss = loss + lambda_normals_super * normal_error.mean()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                pixel_loss,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                lpips_fn,
            )

            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                )
                min_weights = triangle_vertex_weights.min(dim=1).values

                mask_opacity = (min_weights <= prune_triangles).squeeze()
                mask_importance = (triangles.importance_score <= prune_triangles).squeeze()
                mask_size = (triangles.image_size > prune_size).squeeze()
                delete_mask = mask_opacity | mask_size

                if number_of_training_views < 500:
                    delete_mask = delete_mask | mask_importance

                keep_mask = ~delete_mask

                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)

                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(
                    triangles.vertices.shape[0],
                    dtype=torch.bool,
                    device=device,
                )
                if triangles._triangle_indices.numel() > 0:
                    flat_indices = triangles._triangle_indices.flatten()
                    used_vertex_mask[flat_indices] = True

                weight_mask = triangles.get_vertex_weight.squeeze() >= prune_triangles
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                triangles._prune_vertices(vertex_mask)

                needs_densification = (
                    iteration < opt.densify_until_iter
                    and iteration % opt.densification_interval == 0
                    and iteration > opt.densify_from_iter
                )

                if needs_densification:
                    triangles.add_new_gs(
                        iteration,
                        cap_max=opt.max_points,
                        splitt_large_triangles=splitt_large_triangles,
                    )

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity
                    ratio = min(
                        1.0,
                        max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)),
                    )
                    current_opacity = init_opacity + (final_opacity - init_opacity) * ratio
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)
                    prune_triangles += 0.01
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:
                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity
                    ratio = min(
                        1.0,
                        max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)),
                    )
                    current_opacity = init_opacity + (final_opacity - init_opacity) * ratio
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)
                    prune_triangles += 0.01

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    if last_iteration >= opt.iterations:
        with torch.no_grad():
            viewpoint_stack = scene.getTrainCameras().copy()
            triangles.importance_score = torch.zeros(
                (triangles._triangle_indices.shape[0]),
                dtype=torch.float,
                device="cuda",
            )
            while viewpoint_stack:
                viewpoint_cam = viewpoint_stack.pop(0)
                render_pkg = render(viewpoint_cam, triangles, pipe, background)

                importance_score = render_pkg["max_blending"].detach()
                mask = importance_score > triangles.importance_score
                triangles.importance_score[mask] = importance_score[mask]
            mask_importance = (triangles.importance_score <= 0.5).squeeze()
            triangles.prune_triangles(~mask_importance)

            device = triangles.vertices.device
            used_vertex_mask = torch.zeros(
                triangles.vertices.shape[0],
                dtype=torch.bool,
                device=device,
            )
            if triangles._triangle_indices.numel() > 0:
                flat_indices = triangles._triangle_indices.flatten()
                used_vertex_mask[flat_indices] = True

            vertex_mask = used_vertex_mask
            triangles._prune_vertices(vertex_mask)

            scene.save(last_iteration)
    print("Training is done")


def run_training(argv: Sequence[str] | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("MeshSplatting training requires a CUDA-enabled PyTorch install.")

    parser, model_params, opt_params, pipe_params = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.source_path = os.path.abspath(str(args.source_path))
    args.model_path = str(args.model_path)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")
    safe_state(args.quiet)

    dataset = model_params.extract(args)
    opt = opt_params.extract(args)
    pipe = pipe_params.extract(args)

    if args.indoor:
        opt = update_indoor(opt)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    start_time = time.time()
    training(
        dataset,
        opt,
        pipe,
        args.test_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.scene_name,
        load_iteration=args.load_iteration,
        lpips_fn=lpips_fn,
        use_sparse_adam=args.use_sparse_adam,
    )
    print(f"\nTraining complete in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    run_training(sys.argv[1:])
