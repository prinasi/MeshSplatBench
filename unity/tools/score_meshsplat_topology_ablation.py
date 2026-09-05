#!/usr/bin/env python3
"""Score the four MeshSplatting indexed/soup × Unity-renderer conditions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import lpips
import math
import numpy as np
from evaluate_unity_renders_vgg import image, local_ssim, lpips_value

SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR = {"bonsai", "counter", "kitchen", "room"}
CONDITIONS = {
    "indexed_default": "unity_standard_mesh_opacity_floor_fixed_test_eval_unity6_metal_srgb",
    "indexed_dedicated": "unity_method_specific_terminal_solid_test_eval_unity6_metal_srgb",
    "soup_default": "unity_default_triangle_soup_test_eval_unity6_metal_srgb",
    "soup_dedicated": "unity_dedicated_triangle_soup_test_eval_unity6_metal_srgb",
}


def mean(values: list[float]) -> float: return sum(values) / len(values)


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "outputs/mesh-splatting/mipnerf360"
    datasets = Path("/Volumes/GLOWAY/Datasets/MipNeRF360")
    model = lpips.LPIPS(net="vgg").to("mps").eval()
    result: dict = {"protocol": {"split": "MipNeRF360 LLFF holdout; 246 views", "topology": "soup duplicates every indexed corner and all learned per-vertex attributes", "quality": "raw RGB PSNR, reflected 11x11 local SSIM, VGG LPIPS", "fps": "per-view engine frame interval, no ReadPixels/encoding/file I/O"}, "conditions": {}}
    for name, output_name in CONDITIONS.items():
        rows = []
        for scene in SCENES:
            render_dir = root / scene / output_name
            reference_dir = datasets / scene / ("images_2" if scene in INDOOR else "images_4")
            print(f"[score] {name}/{scene}", flush=True)
            psnrs, ssims, perceptual = [], [], []
            for prediction in sorted((render_dir / "test").glob("*.png")):
                candidates = [reference_dir / (prediction.stem + suffix) for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")]
                reference = next(path for path in candidates if path.exists())
                x, y = image(prediction), image(reference)
                psnrs.append(-10 * math.log10(float(np.mean((x - y) ** 2))))
                ssims.append(local_ssim(x, y)); perceptual.append(lpips_value(model, x, y, "mps"))
            quality = {"count": len(psnrs), "psnr": mean(psnrs), "ssim": mean(ssims), "lpips_vgg": mean(perceptual)}
            (render_dir / "metrics_topology_vgg.json").write_text(json.dumps({"test": quality}, indent=2) + "\n")
            fps = list(csv.DictReader((render_dir / "fps_per_test_view.csv").open()))
            rows.append({"scene": scene, "views": quality["count"], "psnr": quality["psnr"], "ssim": quality["ssim"], "lpips_vgg": quality["lpips_vgg"], "fps": mean([float(row["fps"]) for row in fps])})
        result["conditions"][name] = {"scenes": rows, "test": {"scene_count": len(rows), "views_total": sum(row["views"] for row in rows), "psnr_macro": mean([row["psnr"] for row in rows]), "ssim_macro": mean([row["ssim"] for row in rows]), "lpips_vgg": mean([row["lpips_vgg"] for row in rows]), "fps_mean_over_scene_means": mean([row["fps"] for row in rows])}}
    output = Path(__file__).resolve().parents[1] / "Results" / "meshsplat_topology_renderer_ablation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
