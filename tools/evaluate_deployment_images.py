#!/usr/bin/env python3
"""Canonical CPU-capable evaluator for Native and Unity deployment images.

The same implementation scores every prediction against dataset ground truth
and, optionally, against the matched Native CUDA renders. Inputs remain in raw
image code-value space [0, 1]. No prediction-derived normalization is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
EVALUATOR_REVISION = 2
NATIVE_SHAPE_POLICIES = ("strict", "crop", "skip")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def native_gaussian_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Match the 11x11 Gaussian SSIM used by the native triangle methods."""

    import torch
    import torch.nn.functional as functional

    x = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
    coordinates = torch.arange(11, dtype=x.dtype) - 5
    gaussian = torch.exp(-(coordinates**2) / (2.0 * 1.5**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian).reshape(1, 1, 11, 11).expand(3, 1, 11, 11)

    def smooth(value):
        return functional.conv2d(value, window, padding=5, groups=3)

    mean_x, mean_y = smooth(x), smooth(y)
    mean_x2, mean_y2, mean_xy = mean_x.square(), mean_y.square(), mean_x * mean_y
    variance_x = smooth(x * x) - mean_x2
    variance_y = smooth(y * y) - mean_y2
    covariance = smooth(x * y) - mean_xy
    value = ((2.0 * mean_xy + 0.01**2) * (2.0 * covariance + 0.03**2)) / (
        (mean_x2 + mean_y2 + 0.01**2) * (variance_x + variance_y + 0.03**2)
    )
    return float(value.mean().item())


def _image_key(path: Path) -> str:
    # MeshSplatBench Native renders use ``00023_DSC08140.png`` while Unity captures
    # use ``DSC08140.png``. Strip only the framework's numeric prefix.
    return re.sub(r"^\d{5}_", "", path.stem).casefold()


def image_index(root: Path) -> dict[str, Path]:
    if (root / "test").is_dir():
        root = root / "test"
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    result: dict[str, Path] = {}
    for path in files:
        key = _image_key(path)
        if key in result:
            raise ValueError(f"duplicate normalized image key {key!r}: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"no RGB images found in {root}")
    return result


def build_lpips(net: str, device: str, tile_size: int) -> Callable[[np.ndarray, np.ndarray], float]:
    import lpips
    import torch

    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    def evaluate(prediction: np.ndarray, target: np.ndarray) -> float:
        if tile_size <= 0:
            with torch.no_grad():
                x = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                y = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                return float(model(x, y).item())
        height, width = prediction.shape[:2]

        def edges(length: int) -> list[int]:
            count = max(1, math.ceil(length / tile_size))
            return [round(i * length / count) for i in range(count + 1)]

        ys, xs = edges(height), edges(width)
        weighted, pixels_total = 0.0, 0
        with torch.no_grad():
            for row in range(len(ys) - 1):
                for column in range(len(xs) - 1):
                    top, bottom = ys[row], ys[row + 1]
                    left, right = xs[column], xs[column + 1]
                    x = torch.from_numpy(prediction[top:bottom, left:right]).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                    y = torch.from_numpy(target[top:bottom, left:right]).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                    pixels = (bottom - top) * (right - left)
                    weighted += float(model(x, y).item()) * pixels
                    pixels_total += pixels
        return weighted / pixels_total

    return evaluate


def score_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    lpips_fn: Callable[[np.ndarray, np.ndarray], float] | None,
    lpips_net: str,
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"image shape mismatch: {prediction.shape} versus {target.shape}")
    result = {"psnr": psnr(prediction, target), "ssim": native_gaussian_ssim(prediction, target)}
    if lpips_fn is not None:
        result[f"lpips_{lpips_net}"] = lpips_fn(prediction, target)
    return result


def center_crop_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(prediction.shape[0], target.shape[0])
    width = min(prediction.shape[1], target.shape[1])

    def crop(image: np.ndarray) -> np.ndarray:
        top = (image.shape[0] - height) // 2
        left = (image.shape[1] - width) // 2
        return image[top:top + height, left:left + width]

    return crop(prediction), crop(target)


def align_native_pair(
    prediction: np.ndarray,
    native: np.ndarray,
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    if prediction.shape == native.shape:
        return prediction, native, None
    if policy == "strict":
        raise ValueError(f"image shape mismatch: {prediction.shape} versus {native.shape}")
    adjustment = {
        "policy": policy,
        "prediction_shape": list(prediction.shape),
        "native_shape": list(native.shape),
    }
    if policy == "skip":
        return prediction, native, adjustment
    if policy == "crop":
        cropped_prediction, cropped_native = center_crop_pair(prediction, native)
        adjustment["scored_shape"] = list(cropped_prediction.shape)
        return cropped_prediction, cropped_native, adjustment
    raise ValueError(f"unknown native shape policy: {policy}")


def summarize(rows: list[dict[str, object]], metric_field: str) -> dict[str, float | int]:
    metrics = [row[metric_field] for row in rows]
    assert all(isinstance(value, dict) for value in metrics)
    keys = list(metrics[0]) if metrics else []
    return {"count": len(rows), **{key: float(np.mean([float(value[key]) for value in metrics])) for key in keys}}


def evaluate_directories(
    prediction_dir: Path,
    ground_truth_dir: Path,
    *,
    native_dir: Path | None = None,
    lpips_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    lpips_net: str = "vgg",
    allow_partial: bool = False,
    native_shape_policy: str = "strict",
) -> dict[str, object]:
    if native_shape_policy not in NATIVE_SHAPE_POLICIES:
        raise ValueError(f"unknown native shape policy: {native_shape_policy}")
    predictions = image_index(prediction_dir)
    ground_truth = image_index(ground_truth_dir)
    native = image_index(native_dir) if native_dir is not None else None
    expected = set(predictions)
    missing_gt = expected - set(ground_truth)
    missing_native = expected - set(native or {}) if native is not None else set()
    extra_gt = set(ground_truth) - expected
    if not allow_partial and (missing_gt or missing_native):
        raise ValueError(
            "unmatched image sets: "
            f"missing_gt={sorted(missing_gt)}, missing_native={sorted(missing_native)}, extra_gt={sorted(extra_gt)}"
        )
    keys = sorted(expected - missing_gt - missing_native)
    rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    native_shape_adjustments: list[dict[str, object]] = []
    for key in keys:
        prediction = read_rgb(predictions[key])
        target = read_rgb(ground_truth[key])
        row: dict[str, object] = {
            "name": predictions[key].name,
            "key": key,
            "vs_gt": score_pair(prediction, target, lpips_fn, lpips_net),
        }
        if native is not None:
            native_prediction, native_target, adjustment = align_native_pair(
                prediction, read_rgb(native[key]), policy=native_shape_policy
            )
            if adjustment is not None:
                adjustment["key"] = key
                native_shape_adjustments.append(adjustment)
            if adjustment is not None and adjustment["policy"] == "skip":
                row["native_fidelity_skipped"] = adjustment
            else:
                row["vs_native"] = score_pair(native_prediction, native_target, lpips_fn, lpips_net)
                native_rows.append(row)
        rows.append(row)
    if not rows:
        raise ValueError("no matched images remain after pairing")
    report: dict[str, object] = {
        "protocol": {
            "evaluator_revision": EVALUATOR_REVISION,
            "color_space": "raw RGB code values [0,1]; no transfer or prediction normalization",
            "ssim": "native triangle-family 11x11 sigma-1.5 Gaussian SSIM with zero padding",
            "lpips": lpips_net if lpips_fn is not None else "unavailable",
            "strict_image_pairing": not allow_partial,
            "native_shape_policy": native_shape_policy if native is not None else None,
            "native_shape_adjustments": native_shape_adjustments,
        },
        "test": summarize(rows, "vs_gt"),
        "per_view": rows,
    }
    if native is not None and native_rows:
        report["native_fidelity"] = summarize(native_rows, "vs_native")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--native", type=Path, help="Matched Native CUDA render directory.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="vgg")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument(
        "--lpips-tile", type=int, default=0,
        help="Tile edge for an explicitly non-canonical low-memory LPIPS approximation; 0 evaluates full frames.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--native-shape-policy",
        choices=NATIVE_SHAPE_POLICIES,
        default="strict",
        help="How to handle Unity/native shape mismatches; GT scoring remains strict.",
    )
    args = parser.parse_args()
    lpips_fn = build_lpips(args.lpips_net, args.lpips_device, args.lpips_tile) if args.with_lpips else None
    report = evaluate_directories(
        args.prediction.resolve(),
        args.ground_truth.resolve(),
        native_dir=args.native.resolve() if args.native else None,
        lpips_fn=lpips_fn,
        lpips_net=args.lpips_net,
        allow_partial=args.allow_partial,
        native_shape_policy=args.native_shape_policy,
    )
    report["protocol"].update({
        "prediction": str(args.prediction.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "native": str(args.native.resolve()) if args.native else None,
        "lpips_tile": args.lpips_tile if args.with_lpips else None,
        "lpips_full_frame": bool(args.with_lpips and args.lpips_tile <= 0),
    })
    output = args.output or (args.prediction.resolve() / "metrics_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
