#!/usr/bin/env python3
"""Strict aggregation for repeated Unity standalone deployment profiles."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DEFAULT_METHODS = ("2dts", "diffsoup", "mesh-splatting", "triangle-splatting")
DEFAULT_SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
DEFAULT_CONDITIONS = ("general-purpose", "method-aware")


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _run_summary(payload: dict, *, expected_views: int, expected_frames: int, gpu_fraction: float) -> dict:
    views = payload.get("views", [])
    if len(views) != expected_views:
        raise ValueError(f"expected {expected_views} profiled views, found {len(views)}")
    cpu = [float(value) for view in views for value in view.get("cpu_frame_samples_ms", [])]
    gpu = [float(value) for view in views for value in view.get("gpu_frame_samples_ms", []) if float(value) > 0.01]
    expected_samples = expected_views * expected_frames
    if len(cpu) < expected_samples:
        raise ValueError(f"incomplete CPU timing: expected at least {expected_samples} samples, found {len(cpu)}")
    gpu_complete = len(gpu) >= gpu_fraction * expected_samples
    result = {
        "cpu_p50_ms": quantile(cpu, 0.50),
        "cpu_p95_ms": quantile(cpu, 0.95),
        "gpu_complete": gpu_complete,
        "gpu_sample_fraction": len(gpu) / expected_samples,
        "load_to_ready_ms": float(payload.get("load_to_ready_ms", 0.0)),
        "graphics_driver_allocated_bytes": int(payload.get("graphics_driver_allocated_bytes", 0)),
        "asset_bytes": int(payload.get("asset_bytes", 0)),
    }
    if gpu_complete:
        result.update({"gpu_p50_ms": quantile(gpu, 0.50), "gpu_p95_ms": quantile(gpu, 0.95)})
    return result


def aggregate_condition(
    root: Path,
    method: str,
    condition: str,
    scene: str,
    *,
    expected_runs: int,
    expected_views: int,
    expected_frames: int,
    gpu_fraction: float,
) -> dict:
    files = sorted((root / method / condition / scene).glob("run_*/runtime_profile_run_*.json"))
    if len(files) != expected_runs:
        raise ValueError(
            f"{method}/{condition}/{scene}: expected {expected_runs} independent runs, found {len(files)}"
        )
    runs = [
        _run_summary(json.loads(path.read_text()), expected_views=expected_views, expected_frames=expected_frames, gpu_fraction=gpu_fraction)
        for path in files
    ]

    def stats(key: str, subset: list[dict] = runs) -> dict[str, float] | None:
        values = [float(run[key]) for run in subset if key in run]
        if not values:
            return None
        return {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "run_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    gpu_runs = [run for run in runs if run["gpu_complete"]]
    return {
        "runs": len(runs),
        "gpu_complete_runs": len(gpu_runs),
        "cpu_p50_ms": stats("cpu_p50_ms"),
        "cpu_p95_ms": stats("cpu_p95_ms"),
        "gpu_p50_ms": stats("gpu_p50_ms", gpu_runs),
        "gpu_p95_ms": stats("gpu_p95_ms", gpu_runs),
        "load_to_ready_ms": stats("load_to_ready_ms"),
        "graphics_driver_allocated_bytes": stats("graphics_driver_allocated_bytes"),
        "asset_bytes": stats("asset_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--conditions", nargs="*", choices=DEFAULT_CONDITIONS, default=DEFAULT_CONDITIONS)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile-views", type=int, default=3)
    parser.add_argument("--timed-frames", type=int, default=180)
    parser.add_argument("--gpu-completeness", type=float, default=0.8)
    parser.add_argument("--require-player", action="store_true")
    args = parser.parse_args()
    protocol_path = args.profile_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing deployment protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    if args.require_player and not protocol.get("player"):
        raise ValueError("standalone Player is required; Editor profiles cannot satisfy this aggregate")
    if not 0.0 < args.gpu_completeness <= 1.0:
        raise ValueError("--gpu-completeness must be in (0,1]")
    rows = []
    for method in args.methods:
        for condition in args.conditions:
            for scene in args.scenes:
                rows.append({
                    "method": method,
                    "condition": condition,
                    "scene": scene,
                    **aggregate_condition(
                        args.profile_root, method, condition, scene,
                        expected_runs=args.runs,
                        expected_views=args.profile_views,
                        expected_frames=args.timed_frames,
                        gpu_fraction=args.gpu_completeness,
                    ),
                })
    report = {
        "protocol": {
            "source": str(protocol_path),
            "runtime": "standalone-player" if protocol.get("player") else "editor-development-only",
            "expected_runs": args.runs,
            "expected_profile_views": args.profile_views,
            "expected_timed_frames_per_view": args.timed_frames,
            "gpu_completeness_threshold": args.gpu_completeness,
        },
        "conditions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
