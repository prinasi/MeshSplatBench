#!/usr/bin/env python3
"""Evaluate Unity COLMAP-camera captures against MipNeRF360 images_4.

PSNR and SSIM are evaluated in the native raw RGB code-value domain [0, 1]. SSIM uses the
standard local statistics formula with an 11x11 reflected-window mean.  LPIPS
is included when torch + lpips are installed; it is deliberately reported as
unavailable rather than replaced by a different perceptual metric.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def box_mean(image: np.ndarray, size: int = 11) -> np.ndarray:
    """Reflected local mean, implemented using an integral image."""
    radius = size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    return (integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]) / float(size * size)


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mean_x, mean_y = box_mean(prediction), box_mean(target)
    variance_x = np.maximum(box_mean(prediction * prediction) - mean_x * mean_x, 0.0)
    variance_y = np.maximum(box_mean(target * target) - mean_y * mean_y, 0.0)
    covariance = box_mean(prediction * target) - mean_x * mean_y
    value = ((2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)) / (
        (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
    )
    return float(value.mean())


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def lpips_evaluator(net: str, tile_size: int = 512):
    try:
        import torch
        import lpips
    except ImportError:
        return None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = lpips.LPIPS(net=net).to(device).eval()

    def evaluate(prediction: np.ndarray, target: np.ndarray) -> float:
        # VGG16's full-resolution activation footprint can exceed unified
        # memory on 1.5K MipNeRF360 frames.  Tiling preserves the input
        # resolution and the VGG-LPIPS definition; only convolutional context
        # at tile boundaries differs.  The area-weighted reduction matches the
        # standard spatial mean away from those boundaries.
        height, width = prediction.shape[:2]
        def edges(length: int):
            # Partition rather than leave a thin final stripe: VGG has five
            # pooling stages and requires every tile edge to be at least 32.
            count = max(1, math.ceil(length / tile_size))
            return [round(index * length / count) for index in range(count + 1)]
        ys, xs = edges(height), edges(width)
        tiles = [(ys[row], ys[row + 1], xs[column], xs[column + 1])
                 for row in range(len(ys) - 1) for column in range(len(xs) - 1)]
        total, area = 0.0, 0
        with torch.no_grad():
            for top, bottom, left, right in tiles:
                a = torch.from_numpy(prediction[top:bottom, left:right].transpose(2, 0, 1)).unsqueeze(0).to(device) * 2.0 - 1.0
                b = torch.from_numpy(target[top:bottom, left:right].transpose(2, 0, 1)).unsqueeze(0).to(device) * 2.0 - 1.0
                pixels = (bottom - top) * (right - left)
                total += float(model(a, b).item()) * pixels
                area += pixels
        return total / area
    return evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="vgg")
    parser.add_argument("--lpips-tile", type=int, default=512, help="Maximum full-resolution VGG-LPIPS tile edge; no image resampling.")
    args = parser.parse_args()

    manifest_path = args.renders / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open()))
    lpips_fn = lpips_evaluator(args.lpips_net, args.lpips_tile) if args.with_lpips else None
    if args.with_lpips and lpips_fn is None:
        print("LPIPS unavailable: install torch, torchvision, and lpips, then re-run --with-lpips.", file=sys.stderr)

    results = []
    for number, row in enumerate(rows, start=1):
        prediction_path = args.renders / row["render"]
        reference_path = args.references / row["image"]
        prediction, target = read_rgb(prediction_path), read_rgb(reference_path)
        if prediction.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: {prediction_path} {prediction.shape}, {reference_path} {target.shape}")
        item = {"split": row["split"], "index": int(row["index"]), "image": row["image"],
                "psnr": psnr(prediction, target), "ssim": ssim(prediction, target)}
        if lpips_fn is not None:
            item[f"lpips_{args.lpips_net}"] = lpips_fn(prediction, target)
        results.append(item)
        print(f"{number}/{len(rows)} {row['image']}: PSNR={item['psnr']:.4f} SSIM={item['ssim']:.5f}")

    lpips_column = f"lpips_{args.lpips_net}"
    columns = ["split", "index", "image", "psnr", "ssim"] + ([lpips_column] if lpips_fn else [])
    with (args.renders / "metrics_per_view.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(results)

    summary = {"protocol": {"reference": str(args.references), "color_space": "native raw RGB code values [0,1]; no sRGB transfer",
                             "ssim": "11x11 reflected local-window SSIM", "lpips": args.lpips_net if lpips_fn else "unavailable",
                             "lpips_tile": args.lpips_tile if lpips_fn else None}}
    for split in ("train", "test", "all"):
        items = results if split == "all" else [item for item in results if item["split"] == split]
        values = {key: float(np.mean([item[key] for item in items])) for key in columns[3:]} if items else {key: None for key in columns[3:]}
        summary[split] = {"count": len(items), **values}
    (args.renders / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
