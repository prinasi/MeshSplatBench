#!/usr/bin/env python3
"""Run the Unity TriAsset evaluator for config-resolved COLMAP scenes.

Each Unity process loads one asset, which avoids retaining multiple 0.5--1.2
GiB triassets in unified memory.  PNGs and the no-I/O FPS CSV are written under
the corresponding MeshSplatBench output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def reference_image_directory(scene: str) -> str:
    """Match the MipNeRF360 resolution convention used by native evaluation."""
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def unity_graphics_arguments() -> list[str]:
    """Select the native Unity graphics API for the current host platform."""
    if sys.platform == "darwin":
        return ["-force-metal"]
    if sys.platform.startswith("linux"):
        return ["-force-vulkan"]
    # On Windows, retain the project's pinned graphics API instead of making
    # an unverified D3D11/D3D12 choice here.
    return []


def validate_captured_images(output: Path, *, newer_than: float | None = None) -> None:
    """Reject false-success runs that wrote no image or only solid clears."""
    images = sorted(output.rglob("*.png"))
    if newer_than is not None:
        # Allow one second for coarse filesystem timestamp resolution.
        images = [path for path in images if path.stat().st_mtime >= newer_than - 1.0]
    if not images:
        raise RuntimeError(f"Unity completion marker exists but no PNG was captured under {output}")

    sampled = images[: min(3, len(images))]
    solid = []
    for path in sampled:
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
        solid.append(all(low == high for low, high in extrema))
    if all(solid):
        names = ", ".join(path.relative_to(output).as_posix() for path in sampled)
        raise RuntimeError(
            "Unity wrote only solid-color captures; the renderer did not produce "
            f"validated scene pixels (sampled: {names}). Inspect {output / 'unity_editor.log'}"
        )


def unity_failure_summary(log_path: Path, *, max_lines: int = 12) -> str:
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


def validate_runtime_profile(path: Path, *, expected_views: int, newer_than: float) -> None:
    if not path.is_file() or path.stat().st_mtime < newer_than - 1.0:
        raise RuntimeError(f"Unity completion marker exists but runtime profile is missing or stale: {path}")
    payload = json.loads(path.read_text())
    views = payload.get("views", [])
    if len(views) != expected_views:
        raise RuntimeError(
            f"Unity runtime profile has {len(views)} views, expected {expected_views}: {path}"
        )
    if not all(view.get("cpu_frame_samples_ms") for view in views):
        raise RuntimeError(f"Unity runtime profile has incomplete CPU timing samples: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity", type=Path, required=True, help="Unity executable inside Unity.app")
    parser.add_argument("--unity-project", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/triangle-splatting/mipnerf360"))
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-name", default="unity_eval_unity6_metal_srgb")
    parser.add_argument("--log-name", default="unity_editor.log")
    parser.add_argument("--method", default="triangle-splatting", help="Triasset method name and Unity batch renderer selector.")
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
    parser.add_argument(
        "--reference-image-dir",
        help="Dataset image directory used for capture resolution and ground truth. Defaults to MipNeRF360 convention.",
    )
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--fps-warmup", type=int, default=3)
    parser.add_argument("--fps-frames", type=int, default=12)
    parser.add_argument("--test-only", action="store_true", help="Capture only LLFF held-out views.")
    parser.add_argument("--profile-only", action="store_true", help="Write a no-I/O runtime profile instead of PNG captures.")
    parser.add_argument("--profile-run", type=int, default=1)
    parser.add_argument("--profile-views", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=60)
    parser.add_argument("--profile-frames", type=int, default=180)
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument(
        "--general-purpose", "--standard-mesh", dest="general_purpose", action="store_true",
        help="Use the ordinary opaque Unity Mesh baseline. The legacy --standard-mesh spelling is accepted.",
    )
    condition.add_argument(
        "--method-aware", "--method-specific", dest="method_aware", action="store_true",
        help="Use the Unity method-aware renderer. The legacy --method-specific spelling is accepted.",
    )
    args = parser.parse_args()
    args.unity = args.unity.resolve()
    args.unity_project = args.unity_project.resolve()
    args.outputs_root = args.outputs_root.resolve()
    args.datasets_root = args.datasets_root.resolve()
    if args.topology == "mesh":
        args.topology = "indexed"
    for name in ("fps_warmup", "fps_frames", "profile_run", "profile_views", "profile_warmup", "profile_frames"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        from tools.validate_unity_triasset_cpu import validate_triasset
    except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
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
        if not (dataset / "sparse/0/cameras.bin").is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f"missing COLMAP/{image_dir.name} dataset: {dataset}")
        reference_image = next((p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if reference_image is None:
            raise FileNotFoundError(f"{image_dir.name} is empty: {dataset}")
        with Image.open(reference_image) as image:
            width, height = image.size
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / args.log_name
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
            "execution_mode": "profile" if args.profile_only else "capture",
            "profile": {
                "run": args.profile_run,
                "views": args.profile_views,
                "warmup_frames": args.profile_warmup,
                "timed_frames": args.profile_frames,
                "runtime": "editor-development-only",
            } if args.profile_only else None,
        }
        (output / "msbench_run_protocol.json").write_text(json.dumps(run_protocol, indent=2) + "\n")
        # A prior successful invocation leaves this marker behind.  It must be
        # removed before starting Unity; otherwise the supervisor can mistake
        # it for this process's completion and terminate Unity before Play
        # mode has begun.
        marker = output / ".unity_capture_complete"
        marker.unlink(missing_ok=True)
        profile_path = output / f"runtime_profile_run_{args.profile_run:02d}.json"
        if args.profile_only:
            profile_path.unlink(missing_ok=True)
        command = [
            str(args.unity), "-batchmode", *unity_graphics_arguments(), "-projectPath", str(args.unity_project),
            "-executeMethod", "MeshSplatBench.UnityNative.Editor.TriAssetBatchRunner.Run",
            "-method", args.method, "-triasset", str(triasset), "-dataset", str(dataset), "-output", str(output),
            "-triasset-width", str(width), "-triasset-height", str(height),
            "-fps-warmup", str(args.fps_warmup), "-fps-frames", str(args.fps_frames),
            "-topology", args.topology,
            "-logFile", str(log_path),
        ]
        if args.test_only:
            command.append("-test-only")
        if args.profile_only:
            command.extend((
                "-profile-only", "1",
                "-profile-run", str(args.profile_run),
                "-profile-views", str(args.profile_views),
                "-profile-warmup", str(args.profile_warmup),
                "-profile-frames", str(args.profile_frames),
            ))
        command.extend(("-background-color", str(rendering.get("background_color", "black"))))
        if args.general_purpose:
            command.extend(("-standard-mesh", "1"))
        if args.indexed_mesh_method_aware:
            command.extend(("-indexed-mesh-method-aware", "1"))
        if args.method_aware:
            command.extend(("-method-specific", "1"))
        print(f"[MeshSplatBench] Unity scene={scene}, reference={image_dir.name}, resolution={width}x{height}", flush=True)
        capture_started = time.time()
        process = subprocess.Popen(command)
        while process.poll() is None and not marker.is_file():
            time.sleep(2)
        if marker.is_file() and process.poll() is None:
            # The Editor remains alive after Play mode finishes in batch mode.
            # All requested artifacts have already been flushed before this
            # marker is written, so release the per-project Unity lock now.
            process.terminate()
        try:
            exit_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=10)
        if not marker.is_file():
            raise RuntimeError(
                f"Unity exited without completion marker: {marker}; exit={exit_code}\n"
                f"Relevant Unity log lines ({log_path}):\n{unity_failure_summary(log_path)}"
            )
        if args.profile_only:
            validate_runtime_profile(
                profile_path, expected_views=args.profile_views, newer_than=capture_started
            )
        else:
            validate_captured_images(output, newer_than=capture_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
