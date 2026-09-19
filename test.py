#!/usr/bin/env python3
"""Evaluate Unity DiffSoup captures against the saved Python-render GT images."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from tools.evaluate_deployment_images import native_gaussian_ssim as ssim
from tools.evaluate_deployment_images import psnr

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FRAME_PREFIX = re.compile(r"^\d{5}_")


def image_key(path: Path) -> str:
    """Normalize ``00024_DSCF0848.png`` and ``DSCF0848.png`` to one key."""
    return FRAME_PREFIX.sub("", path.stem).casefold()


def image_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = image_key(path)
        if key in result:
            raise ValueError(f"duplicate image key {key!r}: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"no images found in {directory}")
    return result


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def build_lpips(device: str, tile_size: int) -> Callable[[np.ndarray, np.ndarray], float]:
    import lpips
    import torch

    model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    def evaluate(prediction: np.ndarray, target: np.ndarray) -> float:
        height, width = prediction.shape[:2]
        if tile_size <= 0:
            tiles = [(0, height, 0, width)]
        else:
            rows = max(1, math.ceil(height / tile_size))
            columns = max(1, math.ceil(width / tile_size))
            ys = [round(index * height / rows) for index in range(rows + 1)]
            xs = [round(index * width / columns) for index in range(columns + 1)]
            tiles = [
                (ys[row], ys[row + 1], xs[column], xs[column + 1])
                for row in range(rows)
                for column in range(columns)
            ]

        weighted_value = 0.0
        total_pixels = 0
        with torch.no_grad():
            for top, bottom, left, right in tiles:
                pred_tensor = (
                    torch.from_numpy(prediction[top:bottom, left:right])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                    * 2.0
                    - 1.0
                )
                gt_tensor = (
                    torch.from_numpy(target[top:bottom, left:right])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                    * 2.0
                    - 1.0
                )
                pixels = (bottom - top) * (right - left)
                weighted_value += float(model(pred_tensor, gt_tensor).item()) * pixels
                total_pixels += pixels
        return weighted_value / total_pixels

    return evaluate


def evaluate_scene(
    scene: str,
    prediction_dir: Path,
    ground_truth_dir: Path,
    lpips_fn: Callable[[np.ndarray, np.ndarray], float] | None,
) -> dict[str, float]:
    predictions = image_index(prediction_dir)
    ground_truth = image_index(ground_truth_dir)
    missing = sorted(set(predictions) - set(ground_truth))
    if missing:
        raise ValueError(f"{scene}: Unity images without matching GT: {missing[:5]}")

    values: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips_vgg": []}
    for key in sorted(predictions):
        pred_path = predictions[key]
        gt_path = ground_truth[key]
        with Image.open(gt_path) as gt_pil:
            target_size = gt_pil.size
        with Image.open(pred_path) as pred_pil:
            source_size = pred_pil.size
        prediction = read_rgb(pred_path, size=target_size)
        target = read_rgb(gt_path)
        if source_size != target_size:
            print(f"[{scene}] resize {pred_path.name}: {source_size} -> {target_size}")
        values["psnr"].append(psnr(prediction, target))
        values["ssim"].append(ssim(prediction, target))
        if lpips_fn is not None:
            values["lpips_vgg"].append(lpips_fn(prediction, target))

    return {
        name: float(np.mean(metric_values))
        for name, metric_values in values.items()
        if metric_values
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/diffsoup/mipnerf360"))
    parser.add_argument("--prediction-subdir", default="unity_method_aware/test")
    parser.add_argument("--ground-truth-subdir", default="renders/test/gt")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--lpips-device")
    parser.add_argument("--lpips-tile", type=int, default=512)
    args = parser.parse_args()

    lpips_fn = None
    if not args.no_lpips:
        import torch

        device = args.lpips_device or ("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = build_lpips(device, args.lpips_tile)
        print(f"LPIPS device: {device}")

    scene_metrics: list[dict[str, float]] = []
    for scene_root in sorted(path for path in args.root.iterdir() if path.is_dir()):
        prediction_dir = scene_root / args.prediction_subdir
        ground_truth_dir = scene_root / args.ground_truth_subdir
        if not prediction_dir.is_dir() or not ground_truth_dir.is_dir():
            continue
        metrics = evaluate_scene(
            scene_root.name, prediction_dir, ground_truth_dir, lpips_fn
        )
        scene_metrics.append(metrics)
        text = f"Scene {scene_root.name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}"
        if "lpips_vgg" in metrics:
            text += f", LPIPS(VGG)={metrics['lpips_vgg']:.4f}"
        print(text)

    if not scene_metrics:
        raise RuntimeError(f"no evaluable scenes found under {args.root}")
    summary = {
        key: float(np.mean([metrics[key] for metrics in scene_metrics]))
        for key in scene_metrics[0]
    }
    text = (
        f"MipNeRF 360 macro average over {len(scene_metrics)} scenes: "
        f"PSNR={summary['psnr']:.2f}, SSIM={summary['ssim']:.4f}"
    )
    if "lpips_vgg" in summary:
        text += f", LPIPS(VGG)={summary['lpips_vgg']:.4f}"
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
