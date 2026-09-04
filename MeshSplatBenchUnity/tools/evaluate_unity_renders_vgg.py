#!/usr/bin/env python3
"""Matched Unity PNG evaluation: raw RGB PSNR, reflected 11x11 SSIM, VGG LPIPS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def local_ssim(x: np.ndarray, y: np.ndarray) -> float:
    # Matches the established 11x11 reflected local-window protocol.
    ux = uniform_filter(x, size=(11, 11, 1), mode="reflect")
    uy = uniform_filter(y, size=(11, 11, 1), mode="reflect")
    uxx = uniform_filter(x * x, size=(11, 11, 1), mode="reflect")
    uyy = uniform_filter(y * y, size=(11, 11, 1), mode="reflect")
    uxy = uniform_filter(x * y, size=(11, 11, 1), mode="reflect")
    vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
    c1, c2 = 0.01**2, 0.03**2
    return float(np.mean(((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))))


def lpips_value(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> float:
    # LPIPS uses tiled full-resolution evaluation to avoid changing image scale.
    vals = []
    for top in range(0, x.shape[0], 512):
        for left in range(0, x.shape[1], 512):
            xa, ya = x[top:top + 512, left:left + 512], y[top:top + 512, left:left + 512]
            weight = xa.shape[0] * xa.shape[1]
            # VGG's five pooling stages require each spatial dimension to be
            # at least 32.  Pad only narrow boundary tiles and retain their
            # original-pixel weight in the final average.
            pad_h, pad_w = max(0, 32 - xa.shape[0]), max(0, 32 - xa.shape[1])
            if pad_h or pad_w:
                xa = np.pad(xa, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                ya = np.pad(ya, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            tx = torch.from_numpy(xa).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            ty = torch.from_numpy(ya).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad(): vals.append(float(model(tx, ty).item()) * weight)
    return sum(vals) / (x.shape[0] * x.shape[1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--renders", type=Path, required=True); p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None); p.add_argument("--device", default="cpu")
    args = p.parse_args(); renders = args.renders / "test" if (args.renders / "test").is_dir() else args.renders
    output = args.output or (args.renders / "metrics_summary.json")
    model = lpips.LPIPS(net="vgg").to(args.device).eval()
    psnrs=[]; ssims=[]; lp=[]
    for prediction in sorted(renders.glob("*.png")):
        reference = args.references / prediction.name
        if not reference.exists():
            candidates = [args.references / (prediction.stem + suffix) for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")]
            reference = next((path for path in candidates if path.exists()), reference)
        if not reference.exists(): raise FileNotFoundError(reference)
        x,y=image(prediction),image(reference)
        if x.shape != y.shape: raise ValueError(f"shape mismatch {prediction}: {x.shape} vs {y.shape}")
        psnrs.append(-10 * math.log10(float(np.mean((x-y)**2))))
        ssims.append(local_ssim(x,y)); lp.append(lpips_value(model,x,y,args.device))
    report={"protocol":{"color_space":"native raw RGB code values [0,1]; no sRGB transfer","ssim":"11x11 reflected local-window SSIM","lpips":"vgg","lpips_tile":512},"test":{"count":len(psnrs),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims)),"lpips_vgg":float(np.mean(lp))},"all":{"count":len(psnrs),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims)),"lpips_vgg":float(np.mean(lp))}}
    output.write_text(json.dumps(report, indent=2)+"\n"); print(output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
