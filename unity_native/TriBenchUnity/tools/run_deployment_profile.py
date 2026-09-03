#!/usr/bin/env python3
"""Repeated no-I/O Unity deployment profiling for TriBench assets.

The Unity-side profiler measures three evenly spaced LLFF held-out cameras per
scene by default.  Each independent run is a fresh Editor process so that load
time and post-load memory are meaningful.  Quality PNGs are intentionally not
captured here; the completed matched-camera quality evaluations are reused.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from PIL import Image


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
METHODS = ("2dts", "diffsoup", "mesh-splatting", "triangle-splatting")


def image_directory(scene: str) -> str:
    return "images_2" if scene in {"bonsai", "counter", "kitchen", "room"} else "images_4"


def run_one(args: argparse.Namespace, method: str, mode: str, scene: str, run_id: int) -> None:
    scene_root = args.tribench_root / "outputs" / method / "mipnerf360" / scene
    triasset = scene_root / "unity_native" / f"{method}.triasset"
    dataset = args.datasets_root / scene
    output = args.output_root / method / mode / scene / f"run_{run_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    profile = output / f"runtime_profile_run_{run_id:02d}.json"
    marker = output / ".unity_capture_complete"
    if profile.is_file() and not args.force:
        print(f"[TriBench] reuse {profile}", flush=True)
        return
    if args.force:
        profile.unlink(missing_ok=True)
    if not (triasset / "manifest.json").is_file():
        raise FileNotFoundError(triasset)
    if not (dataset / "sparse/0/cameras.bin").is_file() or not (dataset / image_directory(scene)).is_dir():
        raise FileNotFoundError(f"COLMAP/data missing for {scene}")
    reference = next((p for p in sorted((dataset / image_directory(scene)).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
    if reference is None:
        raise FileNotFoundError(f"No reference image for {scene}")
    with Image.open(reference) as image:
        width, height = image.size
    marker.unlink(missing_ok=True)
    # Metal FrameTiming is not populated by a macOS Player launched with
    # -batchmode.  Standalone profiling therefore uses a normal windowed
    # Player; it still renders only to the fixed off-screen target.
    command = [str(args.player if args.player else args.unity), "-force-metal"]
    if not args.player:
        command.insert(1, "-batchmode")
        command.extend(("-projectPath", str(args.unity_project), "-executeMethod", "TriBench.UnityNative.Editor.TriAssetBatchRunner.Run"))
    command.extend((
        "-method", method, "-triasset", str(triasset), "-dataset", str(dataset), "-output", str(output),
        "-triasset-width", str(width), "-triasset-height", str(height),
        "-profile-only", "1", "-profile-run", str(run_id),
        "-profile-views", str(args.profile_views), "-profile-warmup", str(args.warmup_frames),
        "-profile-frames", str(args.timed_frames), "-logFile", str(output / ("unity_player.log" if args.player else "unity_editor.log")),
    ))
    if mode == "common":
        command.extend(("-standard-mesh", "1"))
    elif method != "triangle-splatting":
        command.extend(("-method-specific", "1"))
    print(f"[TriBench] {method}/{mode}/{scene} run={run_id}", flush=True)
    process = subprocess.Popen(command)
    deadline = time.monotonic() + args.timeout_seconds
    while process.poll() is None and not marker.is_file():
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise TimeoutError(
                f"Unity profile exceeded {args.timeout_seconds}s without a completion marker: {output}"
            )
        time.sleep(2)
    try:
        # The Player calls Application.Quit after writing the marker.  Let it
        # unregister cleanly from macOS before the next independent launch.
        code = process.wait(timeout=30 if marker.is_file() else 5)
    except subprocess.TimeoutExpired:
        process.kill()
        code = process.wait(timeout=10)
    if not profile.is_file():
        raise RuntimeError(f"profile missing after Unity exit={code}: {output}")
    payload = json.loads(profile.read_text())
    if not payload.get("views"):
        raise RuntimeError(f"profile contains no view samples: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity", type=Path, default=Path("/Applications/Unity/Unity-6000.4.7f1/Unity.app/Contents/MacOS/Unity"))
    parser.add_argument("--player", type=Path, help="Optional standalone Player executable; required for valid GPU timing on this setup.")
    parser.add_argument("--unity-project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tribench-root", type=Path, default=Path("/Volumes/GLOWAY/Workspace/tribench"))
    parser.add_argument("--datasets-root", type=Path, default=Path("/Volumes/GLOWAY/Datasets/MipNeRF360"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "Results" / "deployment_profiles")
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=METHODS)
    parser.add_argument("--modes", nargs="*", choices=("common", "faithful"), default=("common", "faithful"))
    parser.add_argument("--scenes", nargs="*", choices=SCENES, default=SCENES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile-views", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--timed-frames", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=int, default=180,
                        help="Per-process watchdog; timed-out runs are excluded rather than entering the aggregate.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.unity = args.unity.resolve(); args.unity_project = args.unity_project.resolve()
    if args.player is not None: args.player = args.player.resolve()
    args.tribench_root = args.tribench_root.resolve(); args.datasets_root = args.datasets_root.resolve(); args.output_root = args.output_root.resolve()
    if not args.unity.is_file(): raise FileNotFoundError(args.unity)
    if args.player is not None and not args.player.is_file(): raise FileNotFoundError(args.player)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = vars(args).copy()
    for key, value in list(protocol.items()):
        if isinstance(value, Path): protocol[key] = str(value)
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for method in args.methods:
        for mode in args.modes:
            for scene in args.scenes:
                for run_id in range(args.runs): run_one(args, method, mode, scene, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
