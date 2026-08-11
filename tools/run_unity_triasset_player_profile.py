#!/usr/bin/env python3
"""Run standalone Unity Player profiles for TriBench TriAssets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def reference_image_directory(scene: str) -> str:
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def unity_graphics_arguments() -> list[str]:
    if sys.platform.startswith("linux"):
        return ["-force-vulkan"]
    if sys.platform == "darwin":
        return ["-force-metal"]
    return []


def valid_gpu_ms(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and 0.01 < float(value) < 1000.0 else None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def player_failure_summary(log_paths: list[Path], *, max_lines: int = 16) -> str:
    needles = (
        "[tribench]", "error", "exception", "fatal", "crash", "gpu frametiming",
        "vulkan", "failed", "display", "x11", "wayland",
    )
    matches = []
    existing = [path for path in log_paths if path.is_file()]
    if not existing:
        return "Unity Player logs were not created: " + ", ".join(str(path) for path in log_paths)
    for path in existing:
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped and any(needle in stripped.lower() for needle in needles):
                matches.append(stripped)
    return "\n".join(matches[-max_lines:]) if matches else "No explicit error found; inspect " + ", ".join(str(path) for path in existing)


def profile_summary(path: Path) -> dict[str, float | int]:
    payload = json.loads(path.read_text())
    gpu = []
    for view in payload.get("views", []):
        for raw in view.get("gpu_frame_samples_ms", []):
            value = valid_gpu_ms(raw)
            if value is not None:
                gpu.append(value)
    p50 = quantile(gpu, 0.50)
    p95 = quantile(gpu, 0.95)
    return {
        "gpu_samples": len(gpu),
        "gpu_p50_ms": p50,
        "gpu_p95_ms": p95,
        "gpu_fps": 1000.0 / p50 if p50 > 0.0 else 0.0,
        "rendering_memory_mib": float(payload.get("graphics_driver_allocated_bytes", 0)) / (1024.0 * 1024.0),
    }


def validate_profile(
    path: Path,
    *,
    expected_views: int,
    expected_frames: int,
    newer_than: float,
    require_gpu_timing: bool,
    gpu_timing_min_fraction: float,
) -> None:
    if not path.is_file() or path.stat().st_mtime < newer_than - 1.0:
        raise RuntimeError(f"Unity Player profile is missing or stale: {path}")
    payload = json.loads(path.read_text())
    views = payload.get("views", [])
    if len(views) != expected_views:
        raise RuntimeError(f"Unity Player profile has {len(views)} views, expected {expected_views}: {path}")
    cpu_samples = sum(len(view.get("cpu_frame_samples_ms", [])) for view in views)
    expected_samples = expected_views * expected_frames
    if cpu_samples < expected_samples:
        raise RuntimeError(
            f"Unity Player profile has incomplete CPU/engine timing: {cpu_samples}/{expected_samples} samples"
        )
    per_view_gpu = [
        sum(
            1
            for value in view.get("gpu_frame_samples_ms", [])
            if valid_gpu_ms(value) is not None
        )
        for view in views
    ]
    gpu_samples = sum(per_view_gpu)
    min_per_view = max(1, int(expected_frames * min(gpu_timing_min_fraction, 0.25)))
    if require_gpu_timing and (
        gpu_samples < expected_samples * gpu_timing_min_fraction
        or any(count < min_per_view for count in per_view_gpu)
    ):
        raise RuntimeError(
            "Unity Player did not produce enough GPU frame timings: "
            f"{gpu_samples}/{expected_samples} valid samples, per_view={per_view_gpu}. "
            "This usually means the Player was launched without a real graphics device/display, "
            "or the graphics API/backend does not expose FrameTimingManager.gpuFrameTime for this run."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/triangle-splatting/mipnerf360"))
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-name", default="unity_method_aware")
    parser.add_argument("--log-name", default="unity_player_profile.log")
    parser.add_argument("--method", default="triangle-splatting")
    parser.add_argument("--asset-subdir", default="unity_native")
    parser.add_argument(
        "--topology",
        choices=("indexed", "mesh", "soup"),
        default="indexed",
        help="Unity mesh layout intervention. 'mesh' is an alias for indexed; 'soup' de-indexes shared vertices at load time.",
    )
    parser.add_argument(
        "--indexed-mesh-method-aware",
        action="store_true",
        help="For mesh-splatting + --general-purpose, use a true Unity indexed MeshRenderer with method-aware SH appearance.",
    )
    parser.add_argument("--reference-image-dir")
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--profile-run", type=int, default=1)
    parser.add_argument("--profile-views", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=60)
    parser.add_argument("--profile-frames", type=int, default=180)
    parser.add_argument("--batchmode", action="store_true", help="Launch the Player with -batchmode; do not use -nographics.")
    parser.add_argument("--display", help="Linux X display used by the Player; defaults to DISPLAY or :0.")
    parser.add_argument("--show-console-output", action="store_true", help="Do not redirect Player stdout/stderr into a per-scene console log.")
    parser.add_argument("--require-gpu-timing", action="store_true")
    parser.add_argument("--gpu-timing-min-fraction", type=float, default=0.8)
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument("--general-purpose", "--standard-mesh", dest="general_purpose", action="store_true")
    condition.add_argument("--method-aware", "--method-specific", dest="method_aware", action="store_true")
    args = parser.parse_args()

    if args.profile_run <= 0 or args.profile_views <= 0 or args.profile_warmup <= 0 or args.profile_frames <= 0:
        parser.error("profile run/views/warmup/frames must be positive")
    if not 0.0 < args.gpu_timing_min_fraction <= 1.0:
        parser.error("--gpu-timing-min-fraction must be in (0, 1]")

    player = args.player.expanduser().resolve()
    if not player.is_file():
        raise FileNotFoundError(f"missing Unity Player executable: {player}")
    args.outputs_root = args.outputs_root.resolve()
    args.datasets_root = args.datasets_root.resolve()
    if args.topology == "mesh":
        args.topology = "indexed"

    try:
        from tools.validate_unity_triasset_cpu import validate_triasset
    except ModuleNotFoundError:
        from validate_unity_triasset_cpu import validate_triasset

    for scene in args.scenes:
        scene_root = args.outputs_root / scene
        triasset = (scene_root / args.asset_subdir / f"{args.method}.triasset").resolve()
        dataset = (args.datasets_root / scene).resolve()
        output = (scene_root / args.output_name).resolve()
        image_dir = dataset / (args.reference_image_dir or reference_image_directory(scene))
        if not (triasset / "manifest.json").is_file():
            raise FileNotFoundError(f"missing triasset: {triasset}")
        validate_triasset(triasset, max_faces=10000)
        manifest_path = triasset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rendering = manifest["rendering"]
        topology_label = str(rendering.get("export_topology") or args.topology)
        general = rendering.get("general_purpose", {})
        if args.general_purpose and not general.get("supported", False):
            raise RuntimeError(
                f"{args.method}/{scene} cannot enter the general-purpose benchmark: "
                f"{general.get('reason', 'manifest does not declare a comparable appearance field')}"
            )
        if not (dataset / "sparse/0/cameras.bin").is_file() or not (dataset / "sparse/0/images.bin").is_file():
            raise FileNotFoundError(f"missing COLMAP sparse binaries: {dataset}")
        reference_image = next((p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if reference_image is None:
            raise FileNotFoundError(f"{image_dir.name} is empty: {dataset}")
        with Image.open(reference_image) as image:
            width, height = image.size

        output.mkdir(parents=True, exist_ok=True)
        log_path = output / args.log_name
        console_log_path = output / f"{args.log_name}.console.log"
        profile_path = output / f"runtime_profile_run_{args.profile_run:02d}.json"
        marker = output / ".unity_capture_complete"
        profile_path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        run_protocol = {
            "asset": str(triasset),
            "asset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": manifest.get("source", {}).get("checkpoint_sha256"),
            "schema_version": manifest.get("schema_version"),
            "export_contract_revision": manifest.get("export_contract_revision"),
            "method": args.method,
            "renderer_condition": "general-purpose" if args.general_purpose else "method-aware",
            "indexed_mesh_method_aware": args.indexed_mesh_method_aware,
            "mesh_topology_layout": topology_label,
            "runtime_topology_layout": args.topology,
            "method_aware_status": rendering.get("unity_method_aware_status"),
            "cuda_equivalent": False,
            "requires_per_camera_depth_sort": rendering.get("unity_method_aware_requires_per_camera_depth_sort", False),
            "background_color": rendering.get("background_color"),
            "dataset": str(dataset),
            "resolution": [width, height],
            "execution_mode": "profile",
            "profile": {
                "run": args.profile_run,
                "views": args.profile_views,
                "warmup_frames": args.profile_warmup,
                "timed_frames": args.profile_frames,
                "runtime": "standalone-player",
                "gpu_timing_required": args.require_gpu_timing,
                "gpu_timing_min_fraction": args.gpu_timing_min_fraction,
            },
        }
        (output / "tribench_run_protocol.json").write_text(json.dumps(run_protocol, indent=2) + "\n")
        command = [
            str(player),
            *unity_graphics_arguments(),
            "-screen-fullscreen", "0",
            "-screen-width", str(width),
            "-screen-height", str(height),
            "-method", args.method,
            "-triasset", str(triasset),
            "-dataset", str(dataset),
            "-output", str(output),
            "-triasset-width", str(width),
            "-triasset-height", str(height),
            "-profile-only", "1",
            "-profile-run", str(args.profile_run),
            "-profile-views", str(args.profile_views),
            "-profile-warmup", str(args.profile_warmup),
            "-profile-frames", str(args.profile_frames),
            "-topology", args.topology,
            "-background-color", str(rendering.get("background_color", "black")),
            "-logFile", str(log_path),
        ]
        if args.batchmode:
            command.insert(1, "-batchmode")
        if args.general_purpose:
            command.extend(("-standard-mesh", "1"))
        if args.indexed_mesh_method_aware:
            command.extend(("-indexed-mesh-method-aware", "1"))
        if args.method_aware:
            command.extend(("-method-specific", "1"))

        print(f"[TriBench] Player profile scene={scene}, reference={image_dir.name}, resolution={width}x{height}", flush=True)
        started = time.time()
        env = os.environ.copy()
        if sys.platform.startswith("linux") and not env.get("DISPLAY"):
            env["DISPLAY"] = args.display or ":0"
        elif args.display:
            env["DISPLAY"] = args.display
        console_handle = None
        if args.show_console_output:
            process = subprocess.Popen(command, env=env)
        else:
            console_handle = console_log_path.open("w", encoding="utf-8", errors="replace")
            process = subprocess.Popen(command, env=env, stdout=console_handle, stderr=subprocess.STDOUT)
        while process.poll() is None and not marker.is_file():
            time.sleep(1)
        try:
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    exit_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=10)
        finally:
            if console_handle is not None:
                console_handle.close()
        if not marker.is_file() or exit_code != 0:
            raise RuntimeError(
                f"Unity Player profile failed: marker={marker.is_file()} exit={exit_code}\n"
                f"Relevant Player log lines ({log_path}, {console_log_path}):\n"
                f"{player_failure_summary([log_path, console_log_path])}"
            )
        validate_profile(
            profile_path,
            expected_views=args.profile_views,
            expected_frames=args.profile_frames,
            newer_than=started,
            require_gpu_timing=args.require_gpu_timing,
            gpu_timing_min_fraction=args.gpu_timing_min_fraction,
        )
        summary = profile_summary(profile_path)
        print(
            f"[TriBench] Player profile complete scene={scene} run={args.profile_run} "
            f"gpu_fps={summary['gpu_fps']:.2f} gpu_p50_ms={summary['gpu_p50_ms']:.3f} "
            f"gpu_p95_ms={summary['gpu_p95_ms']:.3f} gpu_samples={summary['gpu_samples']} "
            f"render_mib={summary['rendering_memory_mib']:.1f} log={log_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
