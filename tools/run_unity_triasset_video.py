#!/usr/bin/env python3
"""Render camera trajectory videos using the Unity-native TriAsset renderer.

Renders videos for method-aware (procedural splatting) or general-purpose
(standard Unity Mesh) conditions along the exact same PCA ellipse camera
trajectory as ``msbench render video``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def find_unity_executable() -> Path | None:
    """Locate the Unity Editor executable on the local system."""
    env_unity = os.environ.get("UNITY")
    if env_unity and Path(env_unity).is_file():
        return Path(env_unity)

    # Search standard macOS locations
    if sys.platform == "darwin":
        candidates = sorted(
            Path("/Applications/Unity/Hub/Editor").glob("*/Unity.app/Contents/MacOS/Unity"),
            reverse=True,
        )
        if candidates:
            return candidates[0]

    # Search standard Linux locations
    if sys.platform.startswith("linux"):
        hub_dir = Path.home() / "Unity/Hub/Editor"
        if hub_dir.is_dir():
            candidates = sorted(hub_dir.glob("*/Editor/Unity"), reverse=True)
            if candidates:
                return candidates[0]

    unity_in_path = shutil.which("Unity") or shutil.which("unity")
    if unity_in_path:
        path = Path(unity_in_path).resolve()
        # Verify it's an executable file
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def find_unity_project() -> Path | None:
    """Locate the MeshSplatBench Unity project root."""
    env_proj = os.environ.get("PROJECT") or os.environ.get("UNITY_PROJECT")
    if env_proj and Path(env_proj).is_dir():
        return Path(env_proj).resolve()

    # Search relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "unity"
    if (candidate / "Assets").is_dir() and (candidate / "ProjectSettings").is_dir():
        return candidate.resolve()
    return None


def reference_image_directory(scene: str) -> str:
    """Match the MipNeRF360 resolution convention used by native evaluation."""
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def unity_graphics_arguments() -> list[str]:
    """Select the native Unity graphics API for the current host platform."""
    if sys.platform == "darwin":
        return ["-force-metal"]
    if sys.platform.startswith("linux"):
        return ["-force-vulkan"]
    return []


def unity_failure_summary(log_path: Path, *, max_lines: int = 16) -> str:
    """Return the useful tail of a Unity failure without dumping its crash trace."""
    if not log_path.is_file():
        return f"Unity log was not created: {log_path}"
    needles = (
        "[msbench]", "[meshsplatbench]", "error", "exception", "fatal", "crash", "sigsegv",
        "missing vulkan framebuffer", "shader error", "compilation failed",
    )
    matches = []
    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped and any(needle in stripped.lower() for needle in needles):
            matches.append(stripped)
    if not matches:
        return f"No explicit error found; inspect {log_path}"
    return "\n".join(matches[-max_lines:])


def validate_captured_frames(frames_dir: Path, expected_count: int, *, newer_than: float | None = None) -> list[Path]:
    """Validate that rendered PNG frames exist and are non-empty."""
    from PIL import Image

    images = sorted(frames_dir.glob("*.png"))
    if newer_than is not None:
        images = [p for p in images if p.stat().st_mtime >= newer_than - 1.0]
    if len(images) < expected_count:
        raise RuntimeError(
            f"Expected {expected_count} rendered frames, but found {len(images)} under {frames_dir}"
        )

    # Check a sample for all-solid false-successes
    sampled = [images[0], images[len(images) // 2], images[-1]]
    solid = []
    for path in sampled:
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
        solid.append(all(low == high for low, high in extrema))
    if all(solid):
        raise RuntimeError(
            "Unity rendered only solid-color frames; no geometry reached the camera. "
            f"Inspect {frames_dir.parent / 'unity_editor.log'}"
        )
    return images


def encode_video_from_frames(
    frames_dir: Path,
    output_video: Path,
    fps: int = 30,
) -> Path:
    """Encode PNG frames in a directory into an MP4 video."""
    import numpy as np
    from PIL import Image

    images = sorted(frames_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG frames to encode in {frames_dir}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio
        writer_factory = iio.get_writer
        imwrite = None
    except ImportError:
        try:
            import imageio.v3 as iio_v3
            writer_factory = None
            imwrite = iio_v3.imwrite
        except ImportError as exc:
            raise ImportError("encode_video_from_frames requires imageio. Install imageio[ffmpeg].") from exc

    writer = writer_factory(str(output_video), fps=fps) if writer_factory is not None else None
    frame_list = [] if writer is None else None
    try:
        for p in images:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB"))
            if writer is not None:
                writer.append_data(arr)
            else:
                frame_list.append(arr)
    finally:
        if writer is not None:
            writer.close()
    if imwrite is not None and frame_list is not None:
        imwrite(str(output_video), np.asarray(frame_list), fps=fps)
    return output_video


def generate_trajectory_from_dataset(
    dataset_path: str | Path,
    *,
    dataset_type: str = "auto",
    split: str = "train",
    image_dir: str = "images",
    resolution: int = 1,
    eval_every: int = 8,
    frames: int = 240,
    fps: int = 30,
    zoom: float = 1.0,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> tuple[dict[str, Any], list[Any]]:
    """Build the exact PCA ellipse camera trajectory as msbench render video."""
    from msbench.core.datasets import load_dataset
    from msbench.core.rendering import generate_ellipse_cameras
    from msbench.core.trajectory import save_trajectory

    ds = load_dataset(
        dataset_path,
        dataset_type=dataset_type,
        split=split,
        eval_every=eval_every,
        image_dir=image_dir,
        resolution=resolution,
    )
    base_cameras = ds.get_all_cameras()
    cam_batches = generate_ellipse_cameras(
        base_cameras,
        n_frames=frames,
        zoom=zoom,
        z_variation=z_variation,
        z_phase=z_phase,
    )

    w = int(cam_batches[0].width)
    h = int(cam_batches[0].height)
    traj_frames = []
    for cam in cam_batches:
        c2w = cam.camtoworlds[0].detach().cpu().numpy()
        K = cam.Ks[0].detach().cpu().numpy()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        traj_frames.append({
            "pose": c2w.flatten().tolist(),
            "intrinsics": [fx, fy, cx, cy],
            "image_size": [w, h],
        })

    trajectory: dict[str, Any] = {
        "format": "nerfbaselines-v1",
        "camera_model": "pinhole",
        "image_size": [w, h],
        "fps": fps,
        "frames": traj_frames,
    }
    return trajectory, cam_batches


def run_unity_video_for_scene(
    *,
    unity_bin: Path,
    unity_project: Path,
    method: str,
    triasset_path: Path,
    dataset_path: Path | None,
    output_dir: Path,
    condition: str = "method-aware",
    topology: str = "indexed",
    indexed_mesh_method_aware: bool = False,
    trajectory_path: Path | None = None,
    dataset_type: str = "auto",
    image_dir: str = "images",
    resolution: int = 1,
    eval_every: int = 8,
    frames: int = 240,
    fps: int = 30,
    zoom: float = 1.0,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    split: str = "train",
    write_frames: bool = False,
    background_color: str | None = None,
    log_name: str = "unity_editor.log",
) -> Path:
    """Run Unity in headless batchmode to render a video trajectory."""
    from tools.validate_unity_triasset_cpu import validate_triasset

    unity_bin = Path(unity_bin).resolve()
    unity_project = Path(unity_project).resolve()
    triasset_path = Path(triasset_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    log_path = output_dir / log_name

    if not (triasset_path / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing triasset: {triasset_path}")
    validate_triasset(triasset_path, max_faces=10000)

    manifest_path = triasset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rendering = manifest.get("rendering", {})
    general = rendering.get("general_purpose", {})
    if condition == "general-purpose" and not general.get("supported", False):
        raise RuntimeError(
            f"{method} cannot enter general-purpose video rendering: "
            f"{general.get('reason', 'manifest does not declare a comparable appearance field')}"
        )

    if background_color is None:
        background_color = rendering.get("background_color", "black")

    # 1. Resolve / generate trajectory JSON
    target_traj_json = output_dir / "trajectory.json"
    if trajectory_path is not None and Path(trajectory_path).is_file():
        trajectory_data = json.loads(Path(trajectory_path).read_text())
        if target_traj_json != Path(trajectory_path).resolve():
            shutil.copyfile(trajectory_path, target_traj_json)
        total_frames = len(trajectory_data.get("frames", []))
        fps = int(trajectory_data.get("fps", fps))
    else:
        if dataset_path is None or not Path(dataset_path).is_dir():
            raise ValueError("Either --trajectory or a valid --dataset path must be supplied.")
        trajectory_data, _ = generate_trajectory_from_dataset(
            dataset_path,
            dataset_type=dataset_type,
            split=split,
            image_dir=image_dir,
            resolution=resolution,
            eval_every=eval_every,
            frames=frames,
            fps=fps,
            zoom=zoom,
            z_variation=z_variation,
            z_phase=z_phase,
        )
        target_traj_json.write_text(json.dumps(trajectory_data, indent=2) + "\n")
        total_frames = len(trajectory_data["frames"])

    # 2. Prepare markers
    marker = output_dir / ".unity_video_complete"
    marker.unlink(missing_ok=True)
    alt_marker = output_dir / ".unity_capture_complete"
    alt_marker.unlink(missing_ok=True)

    # 3. Assemble Unity invocation
    command = [
        str(unity_bin),
        "-batchmode",
        *unity_graphics_arguments(),
        "-projectPath", str(unity_project),
        "-executeMethod", "MeshSplatBench.UnityNative.Editor.TriAssetBatchRunner.RunVideo",
        "-method", method,
        "-triasset", str(triasset_path),
        "-video-trajectory", str(target_traj_json),
        "-output", str(output_dir),
        "-topology", topology,
        "-background-color", str(background_color),
        "-logFile", str(log_path),
    ]
    if condition == "general-purpose":
        command.extend(("-standard-mesh", "1"))
    else:
        command.extend(("-method-specific", "1"))
    if indexed_mesh_method_aware:
        command.extend(("-indexed-mesh-method-aware", "1"))

    print(
        f"[MeshSplatBench] Launching Unity video render: method={method}, "
        f"condition={condition}, frames={total_frames}, fps={fps}, output={output_dir}",
        flush=True,
    )
    started = time.time()
    process = subprocess.Popen(command)
    while process.poll() is None and not marker.is_file() and not alt_marker.is_file():
        time.sleep(1)

    if (marker.is_file() or alt_marker.is_file()) and process.poll() is None:
        process.terminate()

    try:
        exit_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait(timeout=10)

    if not marker.is_file() and not alt_marker.is_file():
        raise RuntimeError(
            f"Unity video rendering failed without completion marker; exit={exit_code}\n"
            f"Relevant log lines ({log_path}):\n{unity_failure_summary(log_path)}"
        )

    # 4. Validate captured PNG frames
    validate_captured_frames(frames_dir, total_frames, newer_than=started)

    # 5. Encode MP4 video
    video_path = output_dir / "render_traj.mp4"
    encode_video_from_frames(frames_dir, video_path, fps=fps)
    print(f"[MeshSplatBench] Unity video saved: {video_path}", flush=True)

    # 6. Cleanup frames if not requested
    if not write_frames and frames_dir.is_dir():
        shutil.rmtree(frames_dir, ignore_errors=True)

    return video_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render video trajectories in Unity for MeshSplatBench assets.",
    )
    parser.add_argument("--unity", type=Path, help="Path to Unity Editor binary.")
    parser.add_argument("--unity-project", type=Path, help="Path to MeshSplatBench Unity project.")
    parser.add_argument("--config", type=Path, help="MeshSplatBench render/training config YAML.")
    parser.add_argument("--method", "-m", help="Method name (e.g. triangle-splatting, mesh-splatting, 2dts, diffsoup).")
    parser.add_argument("--triasset", type=Path, help="Path to source .triasset directory.")
    parser.add_argument("--dataset", "-d", type=Path, help="Path to dataset directory.")
    parser.add_argument("--datasets-root", type=Path, help="Root directory containing scenes.")
    parser.add_argument("--outputs-root", type=Path, help="Outputs root directory.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names to process.")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for the video.")
    parser.add_argument("--output-name", help="Output subfolder name under run directory.")
    parser.add_argument("--trajectory", "-t", type=Path, help="Existing trajectory JSON file.")
    parser.add_argument("--frames", type=int, default=240, help="Number of frames in ellipse trajectory.")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Focal length multiplier.")
    parser.add_argument("--z-variation", type=float, default=0.0, help="Vertical oscillation amplitude.")
    parser.add_argument("--z-phase", type=float, default=0.0, help="Phase offset for vertical oscillation.")
    parser.add_argument("--split", default="train", help="Dataset split for trajectory generation.")
    parser.add_argument("--image-dir", default="images", help="COLMAP image directory.")
    parser.add_argument("--resolution", type=int, default=1, help="Downscale factor or width.")
    parser.add_argument("--eval-every", type=int, default=8, help="Holdout stride.")
    parser.add_argument("--topology", choices=("indexed", "mesh", "soup"), default="indexed")
    parser.add_argument("--indexed-mesh-method-aware", action="store_true")
    parser.add_argument("--write-frames", action="store_true", help="Keep rendered PNG frames alongside MP4.")
    parser.add_argument("--background-color", choices=("black", "white"), default=None)

    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument("--method-aware", "--method-specific", dest="method_aware", action="store_true", default=True, help="Use method-aware procedural Unity renderer (default).")
    cond_group.add_argument("--general-purpose", "--standard-mesh", dest="general_purpose", action="store_true", help="Use standard Unity Mesh baseline.")

    args = parser.parse_args()
    condition = "general-purpose" if args.general_purpose else "method-aware"

    unity_bin = args.unity or find_unity_executable()
    if unity_bin is None or not Path(unity_bin).is_file():
        raise FileNotFoundError(
            "Unity Editor executable not found. Pass --unity /path/to/Unity or set $UNITY."
        )

    unity_project = args.unity_project or find_unity_project()
    if unity_project is None or not Path(unity_project).is_dir():
        raise FileNotFoundError(
            "Unity project directory not found. Pass --unity-project /path/to/unity or set $PROJECT."
        )

    # Config-driven path
    if args.config is not None:
        from msbench.cli.config import (
            adapter_config,
            dataset_config,
            load_cli_config,
            merged_section,
            output_dir as config_output_dir,
        )

        cfg = load_cli_config(args.config)
        assert cfg is not None
        video_cfg = merged_section(cfg, "render", nested="video")
        adapter_cfg = adapter_config(cfg, method=args.method)
        method = str(args.method or adapter_cfg.get("type") or "triangle-splatting")
        dataset_cfg = dataset_config(cfg, dataset=str(args.dataset) if args.dataset else None, stage="render")

        dataset_path = Path(dataset_cfg["root"])
        if "scene" in dataset_cfg and (dataset_path / dataset_cfg["scene"]).is_dir():
            dataset_path = dataset_path / dataset_cfg["scene"]

        run_dir = config_output_dir(cfg)
        cond_folder = "unity_method_aware" if condition == "method-aware" else "unity_general_purpose"
        default_out = Path(run_dir) / cond_folder / "video" if run_dir else Path(f"video_{cond_folder}")
        output_dir = Path(args.output_dir or default_out)

        triasset_path = args.triasset or (Path(run_dir) / "unity_native" / f"{method}.triasset")
        if not triasset_path.is_dir():
            # Auto-export if missing
            from msbench.unity_assets import export_triasset
            ckpt = adapter_cfg.get("checkpoint")
            if ckpt:
                print(f"[MeshSplatBench] Exporting triasset for {method} to {triasset_path}...")
                export_triasset(method=method, checkpoint=ckpt, output_dir=triasset_path)

        run_unity_video_for_scene(
            unity_bin=unity_bin,
            unity_project=unity_project,
            method=method,
            triasset_path=triasset_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            condition=condition,
            topology="indexed" if args.topology == "mesh" else args.topology,
            indexed_mesh_method_aware=args.indexed_mesh_method_aware,
            trajectory_path=args.trajectory,
            dataset_type=str(dataset_cfg.get("type", "auto")),
            image_dir=str(dataset_cfg.get("image_dir", args.image_dir)),
            resolution=int(dataset_cfg.get("resolution", args.resolution)),
            eval_every=int(dataset_cfg.get("eval_every", args.eval_every)),
            frames=int(video_cfg.get("frames", args.frames)),
            fps=int(video_cfg.get("fps", args.fps)),
            zoom=float(video_cfg.get("zoom", args.zoom)),
            z_variation=float(video_cfg.get("z_variation", args.z_variation)),
            z_phase=float(video_cfg.get("z_phase", args.z_phase)),
            split=str(video_cfg.get("split", args.split)),
            write_frames=bool(args.write_frames),
            background_color=args.background_color,
        )
        return 0

    # Direct arguments path
    if args.method is None:
        raise ValueError("Must provide --config or --method.")
    method = args.method
    scenes = args.scenes or ([args.dataset.name] if args.dataset else list(SCENES))

    for scene in scenes:
        if args.dataset and Path(args.dataset).is_dir():
            dataset_path = Path(args.dataset)
        elif args.datasets_root:
            dataset_path = Path(args.datasets_root) / scene
        else:
            dataset_path = None

        cond_folder = "unity_method_aware" if condition == "method-aware" else "unity_general_purpose"
        if args.outputs_root:
            scene_out = Path(args.outputs_root) / scene
            triasset = args.triasset or (scene_out / "unity_native" / f"{method}.triasset")
            out_dir = Path(args.output_dir or (scene_out / cond_folder / "video"))
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            triasset = args.triasset or Path(f"outputs/{method}/{scene}/unity_native/{method}.triasset")
        else:
            scene_out = Path(f"outputs/{method}/{scene}")
            triasset = args.triasset or (scene_out / "unity_native" / f"{method}.triasset")
            out_dir = scene_out / cond_folder / "video"

        ref_img_dir = args.image_dir
        if ref_img_dir == "images" and dataset_path and not (dataset_path / "images").is_dir():
            ref_img_dir = reference_image_directory(scene)

        run_unity_video_for_scene(
            unity_bin=unity_bin,
            unity_project=unity_project,
            method=method,
            triasset_path=triasset,
            dataset_path=dataset_path,
            output_dir=out_dir,
            condition=condition,
            topology="indexed" if args.topology == "mesh" else args.topology,
            indexed_mesh_method_aware=args.indexed_mesh_method_aware,
            trajectory_path=args.trajectory,
            image_dir=ref_img_dir,
            resolution=args.resolution,
            eval_every=args.eval_every,
            frames=args.frames,
            fps=args.fps,
            zoom=args.zoom,
            z_variation=args.z_variation,
            z_phase=args.z_phase,
            split=args.split,
            write_frames=args.write_frames,
            background_color=args.background_color,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
