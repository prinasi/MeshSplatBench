#!/usr/bin/env python3
"""Render MeshSplatting as indexed mesh or de-indexed triangle soup in Unity.

The only intervention is the vertex layout.  ``soup`` duplicates every indexed
corner and every corresponding learned per-vertex attribute, then replaces the
index buffer with sequential indices.  Camera split, resolution, shaders,
background, and FPS protocol are otherwise identical to the established Unity
MeshSplatting evaluation.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from PIL import Image

SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR = {"bonsai", "counter", "kitchen", "room"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--unity", type=Path, required=True)
    p.add_argument("--unity-project", type=Path, required=True)
    p.add_argument("--tribench-root", type=Path, default=Path("/Volumes/GLOWAY/Workspace/tribench"))
    p.add_argument("--datasets-root", type=Path, default=Path("/Volumes/GLOWAY/Datasets/MipNeRF360"))
    p.add_argument("--output-name", required=True)
    p.add_argument("--topology", choices=("indexed", "soup"), required=True)
    p.add_argument("--renderer", choices=("default", "dedicated"), required=True)
    p.add_argument("--fps-warmup", type=int, default=3)
    p.add_argument("--fps-frames", type=int, default=12)
    p.add_argument("--scenes", nargs="*", default=SCENES)
    args = p.parse_args()
    unity = args.unity.resolve(); project = args.unity_project.resolve()
    root = args.tribench_root.resolve() / "outputs" / "mesh-splatting" / "mipnerf360"
    data = args.datasets_root.resolve()
    for scene in args.scenes:
        triasset = root / scene / "unity_native" / "mesh-splatting.triasset"
        image_dir = data / scene / ("images_2" if scene in INDOOR else "images_4")
        reference = next(p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        with Image.open(reference) as image: width, height = image.size
        output = root / scene / args.output_name
        output.mkdir(parents=True, exist_ok=True)
        marker = output / ".unity_capture_complete"; marker.unlink(missing_ok=True)
        command = [str(unity), "-batchmode", "-force-metal", "-projectPath", str(project),
                   "-executeMethod", "TriBench.UnityNative.Editor.TriAssetBatchRunner.Run",
                   "-method", "mesh-splatting", "-triasset", str(triasset), "-dataset", str(data / scene),
                   "-output", str(output), "-triasset-width", str(width), "-triasset-height", str(height),
                   "-fps-warmup", str(args.fps_warmup), "-fps-frames", str(args.fps_frames),
                   "-topology", args.topology, "-test-only", "1", "-logFile", str(output / "unity_editor.log")]
        command.extend(("-standard-mesh", "1") if args.renderer == "default" else ("-method-specific", "1"))
        print(f"[topology-eval] {scene}: {args.renderer}/{args.topology} {width}x{height}", flush=True)
        process = subprocess.Popen(command)
        while process.poll() is None and not marker.is_file(): time.sleep(2)
        if marker.is_file() and process.poll() is None: process.terminate()
        try: code = process.wait(timeout=30)
        except subprocess.TimeoutExpired: process.kill(); code = process.wait(timeout=10)
        if not marker.is_file(): raise RuntimeError(f"Unity failed scene={scene}, exit={code}; see {output / 'unity_editor.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
