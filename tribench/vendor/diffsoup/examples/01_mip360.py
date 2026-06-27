# examples/01_mip360.py
# MipNeRF-360 triangle-soup radiance field optimisation with DiffSoup.
#
# Usage:
#   python examples/01_mip360.py --scene_root ./datasets/360_v2/kitchen
#
# Dependencies (beyond diffsoup):
#   pip install open3d imageio torchvision tqdm pytorch-msssim matplotlib scipy
#
# Note: LPIPS is intentionally excluded from this example.  For fair
# comparison with other methods, ensure you use a consistent LPIPS model
# and weights (e.g. VGG vs AlexNet, v0.0 vs v0.1) across all baselines.

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Optional

# import imageio.v2 as iio
# import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from pytorch_msssim import ssim
from torch.optim import Adam
from tqdm.auto import tqdm

import diffsoup as ds
from utils import (
    load_dtu_scene,
    load_mipnerf360_scene,
    mvp_from_K_Tcw,
    read_points3D,
    project_vertices,
    psnr_fn,
    exp_decay_mult,
    count_visible_triangles,
    build_keep_map,
    split_edges_from_training_views,
    NvidiaSmiMemoryMonitor,
)

# ── Reproducibility ──────────────────────────────────────────────────

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ── Main ─────────────────────────────────────────────────────────────

def _colmap_point_init(
    scene_root: str,
    target_points: int,
    robust_filter: bool = False,
):
    """Load COLMAP points and return a sampled point set plus soup spacing."""
    xyz_np = read_points3D(scene_root)
    total_points = xyz_np.shape[0]
    if total_points == 0:
        raise RuntimeError(f"No COLMAP points found in {scene_root}")

    if robust_filter and total_points >= 100:
        lo, hi = np.percentile(xyz_np, [1.0, 99.0], axis=0)
        keep = np.all((xyz_np >= lo) & (xyz_np <= hi), axis=1)
        filtered = xyz_np[keep]
        if filtered.shape[0] >= min(target_points, total_points) // 2:
            xyz_np = filtered

    extra_points = min(5_000, xyz_np.shape[0], max(0, target_points // 3))
    base_points = min(xyz_np.shape[0], target_points - extra_points)

    xyz_sel = np.empty((0, 3), dtype=np.float32)
    if extra_points > 0:
        sel = np.random.choice(xyz_np.shape[0], extra_points, replace=False)
        xyz_sel = xyz_np[sel]

    if base_points > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd_down = pcd.farthest_point_down_sample(base_points)
        xyz_np = np.asanyarray(pcd_down.points).astype(np.float32)
        if extra_points > 0:
            xyz_np = np.concatenate([xyz_np, xyz_sel], axis=0)
    else:
        xyz_np = xyz_sel

    spacing = 0.25 * ds.nn_spacing(xyz_np, reduction="mean")
    return xyz_np, total_points, spacing


def _composite_white_background(color: torch.Tensor, rast_out: torch.Tensor) -> torch.Tensor:
    mask = (rast_out.detach()[..., -1:] > 0).to(color.dtype)
    return mask * color + (1.0 - mask)


def main(
    scene_root: str = "./datasets/360_v2/kitchen",
    batch_size: int = 4,
    steps: int = 10_000,
    n_points: Optional[int] = 15_000,
    downscale: int = 4,
    flip_z: bool = True,
    dataset_type: str = "auto",
    out_dir: Optional[str] = None,
    white_background: Optional[bool] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_name = os.path.basename(os.path.normpath(scene_root))
    if out_dir is None:
        out_dir = os.path.join("./results/01_mip360", scene_name)
    os.makedirs(out_dir, exist_ok=True)

    if dataset_type == "auto":
        dataset_type = (
            "dtu"
            if os.path.isfile(os.path.join(scene_root, "cameras.npz"))
            else "colmap"
        )
    if dataset_type not in {"colmap", "dtu"}:
        raise ValueError("dataset_type must be one of: auto, colmap, dtu")
    use_white_background = dataset_type == "dtu" if white_background is None else bool(white_background)

    # ── Load data ────────────────────────────────────────────────────

    if dataset_type == "dtu":
        train_data = load_dtu_scene(
            scene_root, split="train", downscale=downscale, device=device,
        )
    else:
        train_data = load_mipnerf360_scene(
            scene_root, split="train", downscale=downscale, device=device,
        )
    K, H, W = train_data["K"], train_data["H"], train_data["W"]
    frames = train_data["frames"]
    N_train = len(frames)
    print(f"[views] train={N_train}  folder={train_data['folder']} size={H}x{W}")

    # ── Geometry: triangle soup from COLMAP points ───────────────────

    target_points = int(n_points or 15_000)
    target_points = max(1, target_points)

    if dataset_type == "dtu":
        try:
            xyz_np, N_total, spacing = _colmap_point_init(
                scene_root, target_points, robust_filter=True,
            )
            print(
                f"[points3D] DTU COLMAP init using {len(xyz_np):,} / "
                f"{N_total:,} points (subsampled)"
            )
        except FileNotFoundError:
            xyz_np = (np.random.rand(target_points, 3).astype(np.float32) * 2.0) - 1.0
            spacing = 0.01
            print(
                f"[points3D] DTU COLMAP points not found; random init with "
                f"{target_points:,} points in normalized space"
            )
    else:
        xyz_np, N_total, spacing = _colmap_point_init(scene_root, target_points)
        print(f"[points3D] loaded original {N_total:,} points")
        print(f"[points3D] using {len(xyz_np):,} / {N_total:,} points (subsampled)")
    xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)
    V_single, F = ds.triangle_soup_from_points(xyz, scale=spacing)

    # ── Feature buffers ──────────────────────────────────────────────
    #   Rmin, Rmax: multi-resolution level range.  Both colour and opacity
    #   features share the same range and are lifted together at step 5 000.

    Rmin, Rmax = 0, 0
    feat_dim = 7

    feat_src = ds.build_multires_triangle_color(
        F.shape[0], Rmin, Rmax, feat_dim,
    ).to(device="cuda")
    feat_src.requires_grad = True

    alpha_src = ds.build_multires_triangle_color(
        F.shape[0], Rmin, Rmax, feat_dim=1,
    ).to(device="cuda")
    alpha_src.requires_grad = True

    # ── Precompute GT and MVPs ───────────────────────────────────────

    gt_rgb = torch.stack([fr["image"].clamp(0, 1) for fr in frames], dim=0)

    z_near_train, z_near_test, z_far = 0.01, 0.5, 100.0

    MVPs = torch.stack([
        mvp_from_K_Tcw(K, fr["Tcw"], (H, W), z_near=z_near_train, z_far=z_far, flip_z=flip_z)
        for fr in frames
    ], dim=0)
    MVPs_inv = torch.inverse(MVPs).contiguous()

    # ── Colour MLP ───────────────────────────────────────────────────

    color_mlp = ds.ColorMLP(
        input_dim=feat_dim + 9, hidden_dim=16, n_layers=2, output_dim=3,
    ).to(device="cuda")

    V_single = V_single.requires_grad_(True)

    # ── Initial renders ──────────────────────────────────────────────

    eval_ids = torch.linspace(0, max(N_train - 1, 0), steps=min(3, N_train)).round().long().unique()
    eval_MVP = MVPs[eval_ids]
    eval_MVP_inv = MVPs_inv[eval_ids]

    with torch.no_grad():
        pred_init = []
        for view_idx in range(eval_MVP.shape[0]):
            V_clip = project_vertices(V_single, eval_MVP[view_idx : view_idx + 1])
            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), V_clip, F,
                level=Rmax,
                alpha_src=ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid(),
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax,
                feat=ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid(),
            ).view(-1, H, W, feat_dim)
            feat = torch.cat([feat, ds.encode_view_dir_sh2(rast_out, eval_MVP_inv[view_idx : view_idx + 1])], dim=-1)
            color = color_mlp.forward(feat, mask=rast_out[..., -1] > 0).view(1, H, W, 3)
            if use_white_background:
                color = _composite_white_background(color, rast_out)
            pred_init.append(color.squeeze(0))
        pred_init = torch.stack(pred_init, dim=0)
    # Diagnostic renders are intentionally not written during TriBench runs.

    # ── Optimisers ───────────────────────────────────────────────────

    optimizer_soup = Adam([
        {"params": [feat_src], "lr": 5e-2},
        {"params": [alpha_src], "lr": 5e-2},
    ])
    optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=1e-2)
    optimizer_shader = Adam([{"params": color_mlp.parameters(), "lr": 1e-2}])

    base_soup_lrs = [pg["lr"] for pg in optimizer_soup.param_groups]
    base_vert_lrs = [pg["lr"] for pg in optimizer_vert.param_groups]
    base_shader_lrs = [pg["lr"] for pg in optimizer_shader.param_groups]

    # ── Training loop ────────────────────────────────────────────────

    losses: List[float] = []
    perm = torch.randperm(N_train, device=device)
    ptr = 0

    torch.cuda.synchronize(device)
    memory_monitor = NvidiaSmiMemoryMonitor()
    memory_monitor.start()
    training_started = time.perf_counter()

    pbar = tqdm(range(1, steps + 1), desc="optimising", leave=True)
    for i_iter in pbar:
        end = min(ptr + batch_size, N_train)
        batch_idx = perm[ptr:end]
        ptr = end
        if ptr == N_train:
            perm = torch.randperm(N_train, device=device)
            ptr = 0

        # LR schedule
        mult = exp_decay_mult(i_iter, steps, final_mult=0.01)
        for opt, base_lrs in (
            (optimizer_soup, base_soup_lrs),
            (optimizer_vert, base_vert_lrs),
            (optimizer_shader, base_shader_lrs),
        ):
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = base_lr * mult

        batch_MVP = MVPs[batch_idx]
        batch_MVP_inv = MVPs_inv[batch_idx]
        batch_gt_rgb = gt_rgb[batch_idx]

        V_clip = project_vertices(V_single, batch_MVP)
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

        rast_out = ds.rasterize_multires_triangle_alpha(
            (H, W), V_clip, F, level=Rmax, alpha_src=alpha_acc,
        )
        feat = ds.multires_triangle_color(
            rast_out, level=Rmax, feat=feat_acc,
        ).view(-1, H, W, feat_dim)
        feat = torch.cat([feat, ds.encode_view_dir_sh2(rast_out, batch_MVP_inv)], dim=-1)
        color = color_mlp.forward(feat, mask=rast_out[..., -1] > 0).view(-1, H, W, 3)
        if use_white_background:
            color = _composite_white_background(color, rast_out)

        # Opacity auxiliary loss (zero-valued; hooks gradient into alpha_src)
        aux_loss = ds.opacity_aux_loss(
            color.detach(), batch_gt_rgb, rast_out, V_clip, F,
            level=Rmax, alpha_src=alpha_acc,
        )
        color = ds.edge_grad(color, rast_out, V_clip, F)
        l1_loss = (batch_gt_rgb - color).abs().mean()
        ssim_loss = 0.5 * (1 - ssim(
            batch_gt_rgb.permute(0, 3, 1, 2), color.permute(0, 3, 1, 2), data_range=1.0,
        ))
        loss = aux_loss + 0.8 * l1_loss + 0.2 * ssim_loss

        optimizer_soup.zero_grad(set_to_none=True)
        optimizer_vert.zero_grad(set_to_none=True)
        optimizer_shader.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_soup.step()
        optimizer_vert.step()
        optimizer_shader.step()

        l = float(loss.detach().item())
        losses.append(l)
        pbar.set_postfix(loss=f"{l:.6f}")

        # ── Lift multi-resolution levels at step 5 000 ───────────────
        if i_iter == 5_000 and i_iter < steps:
            with torch.no_grad():
                feat_src_lifted = ds.accumulate_to_level(0, 0, feat_src, target_level=2)
                Rmin, Rmax = 2, 5
                new_feat_src = ds.build_multires_triangle_color(
                    F.shape[0], Rmin, Rmax, feat_dim=feat_dim,
                ).to(device="cuda")
                new_feat_src[..., : feat_src_lifted.shape[1], :] = feat_src_lifted
                feat_src = new_feat_src

                alpha_src = ds.build_multires_triangle_color(
                    F.shape[0], Rmin, Rmax, feat_dim=1,
                ).to(device="cuda")

            feat_src.requires_grad = True
            alpha_src.requires_grad = True

            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=old_lr_vert)

        # ── Resample soup ────────────────────────────────────────────
        if i_iter % 100 == 0 and i_iter < 9_550:
            with torch.no_grad():
                if F.shape[0] > target_points:
                    alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
                    tri_counts = count_visible_triangles(
                        (H // 2, W // 2), MVPs, V_single, F,
                        level=Rmax, alpha_src=alpha_acc, batch_size=1,
                    )
                    keep_map = build_keep_map(tri_counts, remove=F.shape[0] - target_points)
                    F = F[keep_map]
                    V_single, F = ds.remove_unreferenced_vertices_from_soup(V_single, F)
                    feat_src = ds.expand_by_index(feat_src, keep_map)
                    alpha_src = ds.expand_by_index(alpha_src, keep_map)

                if i_iter < 9_500:
                    alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
                    V_single, F, face_map = split_edges_from_training_views(
                        (H // 2, W // 2), MVPs, V_single, F,
                        Rmax, alpha_acc,
                        tau_ratio=1 / 5, num_views_cap=20,
                    )
                    feat_src = ds.expand_by_index(feat_src, face_map)
                    alpha_src = ds.expand_by_index(alpha_src, face_map)

            # print(f"  [resample] verts={V_single.shape[0]:,}  faces={F.shape[0]:,}")

            V_single.requires_grad = True
            feat_src.requires_grad = True
            alpha_src.requires_grad = True

            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=old_lr_vert)

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    process_peak_mib = memory_monitor.stop()
    training_stats = {
        "training_time_seconds": training_seconds,
        "peak_memory_process_mib": process_peak_mib,
        "peak_memory_process_source": (
            "nvidia-smi process sampling (20 ms)"
            if process_peak_mib is not None
            else "unavailable"
        ),
    }
    with open(os.path.join(out_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    process_memory_text = (
        f"{process_peak_mib:.1f}" if process_peak_mib is not None
        else "N/A"
    )
    print(
        f"[training] time_seconds={training_seconds:.3f} "
        f"peak_process_memory_mib={process_memory_text}"
    )

    torch.cuda.empty_cache()

    # ── Final renders ────────────────────────────────────────────────

    # Final diagnostic renders are skipped; the checkpoint is the canonical output.

    # ── Loss curve ───────────────────────────────────────────────────

    # Loss curves are not written to keep run directories compact.

    # ── Checkpoint ───────────────────────────────────────────────────

    with torch.no_grad():
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

    ckpt_path = os.path.join(out_dir, "final_params.pt")
    torch.save({
        "feat_acc": feat_acc.detach().cpu(),
        "alpha_acc": alpha_acc.detach().cpu(),
        "V": V_single.detach().cpu(),
        "F": F.detach().cpu(),
        "color_mlp": color_mlp.state_dict(),
        "Rmin": Rmin,
        "Rmax": Rmax,
        "feat_dim": feat_dim,
        "K": K.detach().cpu(), "H": H, "W": W,
        "flip_z": flip_z,
        "dataset_type": dataset_type,
        "steps": steps,
        "losses": losses,
        "seed": SEED,
    }, ckpt_path)
    print(f"[save] checkpoint → {ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────

    if dataset_type == "dtu":
        test_data = load_dtu_scene(
            scene_root, split="test", downscale=downscale, device=device,
        )
    else:
        test_data = load_mipnerf360_scene(
            scene_root, split="test", downscale=downscale, device=device,
        )
    test_frames = test_data["frames"]
    print(f"[views] test={len(test_frames)}")

    out_dir_test = os.path.join(out_dir, "test_views")
    os.makedirs(out_dir_test, exist_ok=True)

    psnrs, ssims = [], []

    with torch.no_grad():
        for i, fr in enumerate(test_frames):
            MVP = mvp_from_K_Tcw(
                K, fr["Tcw"], (H, W), z_near=z_near_test, z_far=z_far, flip_z=flip_z,
            ).unsqueeze(0)
            MVP_inv = torch.inverse(MVP).contiguous()
            V_clip = project_vertices(V_single, MVP)

            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), V_clip, F,
                level=Rmax,
                alpha_src=ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid(),
                stochastic=False,
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax,
                feat=ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid(),
            ).view(-1, H, W, feat_dim)
            feat = torch.cat([feat, ds.encode_view_dir_sh2(rast_out, MVP_inv)], dim=-1)
            color = color_mlp.forward(feat, mask=rast_out[..., -1] > 0).view(1, H, W, 3)
            if use_white_background:
                color = _composite_white_background(color, rast_out)

            pred_lin = color.squeeze(0).clamp(0, 1)
            gt_lin = fr["image"].clamp(0, 1)

            psnrs.append(float(psnr_fn(pred_lin, gt_lin).item()))
            pred_nchw = pred_lin.permute(2, 0, 1).unsqueeze(0)
            gt_nchw = gt_lin.permute(2, 0, 1).unsqueeze(0)
            ssims.append(float(ssim(gt_nchw, pred_nchw, data_range=1.0).item()))

    print(f"[save] test metrics → {out_dir_test}/metrics.txt")

    if psnrs:
        print(f"[metrics] PSNR  {np.mean(psnrs):.3f} dB")
        print(f"[metrics] SSIM  {np.mean(ssims):.4f}")

        with open(os.path.join(out_dir_test, "metrics.txt"), "w") as f:
            for i, (p, s) in enumerate(zip(psnrs, ssims)):
                f.write(f"{i:04d} PSNR={p:.8f} SSIM={s:.8f}\n")
            f.write(
                f"\nmean PSNR={np.mean(psnrs):.8f} "
                f"SSIM={np.mean(ssims):.8f}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffSoup MipNeRF-360 example")
    parser.add_argument("--scene_root", type=str, default="./datasets/360_v2/kitchen")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--n_points", type=int, default=15_000)
    parser.add_argument("--downscale", type=int, default=4, choices=[0, 1, 2, 4, 8],
                        help="Image downscale factor: 0/1=images/, 2=images_2/, 4=images_4/, 8=images_8/")
    parser.add_argument("--flip_z", action="store_true", default=True)
    parser.add_argument("--dataset_type", choices=["auto", "colmap", "dtu"], default="auto")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: ./results/01_mip360/<scene>)")
    parser.add_argument("--white_background", action="store_true", default=None,
                        help="Composite unrasterized pixels over white; defaults to true for DTU only.")
    args = parser.parse_args()

    main(
        scene_root=args.scene_root,
        batch_size=args.batch_size,
        steps=args.steps,
        n_points=args.n_points,
        downscale=args.downscale,
        flip_z=args.flip_z,
        dataset_type=args.dataset_type,
        out_dir=args.out_dir,
        white_background=args.white_background,
    )
