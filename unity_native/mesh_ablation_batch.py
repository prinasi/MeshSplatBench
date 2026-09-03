#!/usr/bin/env python3
"""Run MeshSplatting Unity renderer ablations over MipNeRF360 test views."""
from __future__ import annotations

import subprocess
import time
import os
from pathlib import Path

from PIL import Image


UNITY = Path("/Applications/Unity/Unity-6000.4.7f1/Unity.app/Contents/MacOS/Unity")
PROJECT = Path("/Volumes/GLOWAY/Unity/unity_native/TriBenchUnity")
OUTPUTS = Path("/Volumes/GLOWAY/Workspace/tribench/outputs/mesh-splatting/mipnerf360")
DATASETS = Path("/Volumes/GLOWAY/Datasets/MipNeRF360")
SCENES = tuple(os.environ.get("MESH_ABLATION_SCENES", "bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill").split(","))
INDOORS = frozenset(("bonsai", "counter", "kitchen", "room"))
ABLATIONS = tuple(os.environ.get("MESH_ABLATIONS", "full,alpha-test-depth,opaque-depth").split(","))


def main() -> int:
    for ablation in ABLATIONS:
        for scene in SCENES:
            image_dir = DATASETS / scene / ("images_2" if scene in INDOORS else "images_4")
            image_path = next(path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            with Image.open(image_path) as image:
                width, height = image.size
            triasset = OUTPUTS / scene / "unity_native" / "mesh-splatting.triasset"
            output = OUTPUTS / scene / f"unity_mesh_ablation_{ablation.replace('-', '_')}_test_eval_unity6_metal_srgb"
            output.mkdir(parents=True, exist_ok=True)
            command = [
                str(UNITY), "-batchmode", "-force-metal", "-projectPath", str(PROJECT),
                "-executeMethod", "TriBench.UnityNative.Editor.TriAssetBatchRunner.Run",
                "-method", "mesh-splatting", "-method-specific", "1", "-mesh-ablation", ablation,
                "-triasset", str(triasset), "-dataset", str(DATASETS / scene), "-output", str(output),
                "-triasset-width", str(width), "-triasset-height", str(height),
                "-fps-warmup", "3", "-fps-frames", "12", "-test-only",
                "-logFile", str(output / "unity_editor.log"),
            ]
            print(f"[mesh-ablation] {ablation} {scene} {width}x{height}", flush=True)
            marker = output / ".unity_capture_complete"
            marker.unlink(missing_ok=True)
            process = subprocess.Popen(command)
            while process.poll() is None and not marker.is_file():
                time.sleep(2)
            if marker.is_file() and process.poll() is None:
                process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10)
            if exit_code not in (0, -15) or not marker.is_file():
                raise RuntimeError(f"Unity failed for {ablation}/{scene}: exit={exit_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
