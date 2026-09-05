#!/usr/bin/env python3
"""Create paper-ready Unity TriAsset metric reports from scene output folders."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPORT_REVISION = 1


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mib(value: Any) -> float | None:
    number = finite(value)
    return number / (1024.0 * 1024.0) if number is not None else None


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def valid_gpu_ms(value: Any) -> float | None:
    number = finite(value)
    return number if number is not None and 0.01 < number < 1000.0 else None


def read_quality(root: Path) -> dict[str, Any]:
    path = root / "metrics_summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    test = payload.get("test", {})
    native = payload.get("native_fidelity") or {}
    return {
        "views": test.get("count"),
        "psnr": test.get("psnr"),
        "ssim": test.get("ssim"),
        "lpips_vgg": test.get("lpips_vgg"),
        "native_psnr": native.get("psnr"),
        "native_ssim": native.get("ssim"),
        "native_lpips_vgg": native.get("lpips_vgg"),
    }


def read_capture_fps(root: Path) -> dict[str, Any]:
    path = root / "fps_per_test_view.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    frame_ms = [value for row in rows if (value := finite(row.get("mean_frame_ms"))) is not None]
    per_view_fps = [value for row in rows if (value := finite(row.get("fps"))) is not None]
    mean_ms = statistics.fmean(frame_ms) if frame_ms else None
    return {
        "capture_timed_views": len(frame_ms),
        "capture_frame_mean_ms": mean_ms,
        "capture_fps": 1000.0 / mean_ms if mean_ms and mean_ms > 0 else None,
        "capture_fps_view_p50": quantile(per_view_fps, 0.50),
        "capture_fps_view_p95": quantile(per_view_fps, 0.95),
    }


def read_profiles(root: Path, expected_runs: int | None = None) -> dict[str, Any]:
    paths = (
        [root / f"runtime_profile_run_{run:02d}.json" for run in range(1, expected_runs + 1)]
        if expected_runs is not None
        else sorted(root.glob("runtime_profile_run_*.json"))
    )
    paths = [path for path in paths if path.is_file()]
    if not paths:
        return {}
    runs = []
    for path in paths:
        payload = json.loads(path.read_text())
        views = payload.get("views", [])
        cpu = [
            float(value)
            for view in views
            for value in view.get("cpu_frame_samples_ms", [])
            if finite(value) is not None
        ]
        gpu = []
        for view in views:
            for raw in view.get("gpu_frame_samples_ms", []):
                value = valid_gpu_ms(raw)
                if value is not None:
                    gpu.append(value)
        runs.append({
            "cpu_p50_ms": quantile(cpu, 0.50),
            "cpu_p95_ms": quantile(cpu, 0.95),
            "gpu_p50_ms": quantile(gpu, 0.50),
            "gpu_p95_ms": quantile(gpu, 0.95),
            "load_to_ready_ms": finite(payload.get("load_to_ready_ms")),
            "rendering_memory_mib": mib(payload.get("graphics_driver_allocated_bytes")),
            "unity_allocated_mib": mib(payload.get("unity_total_allocated_bytes")),
            "unity_reserved_mib": mib(payload.get("unity_total_reserved_bytes")),
            "asset_memory_mib": mib(payload.get("asset_bytes")),
            "profile_views": len(views),
            "cpu_samples": len(cpu),
            "gpu_samples": len(gpu),
        })

    def run_median(key: str) -> float | None:
        values = [value for run in runs if (value := finite(run.get(key))) is not None]
        return median(values)

    cpu_p50 = run_median("cpu_p50_ms")
    gpu_p50 = run_median("gpu_p50_ms")
    return {
        "profile_runs": len(runs),
        "profile_views_per_run": min(run["profile_views"] for run in runs),
        "profile_cpu_samples": sum(run["cpu_samples"] for run in runs),
        "profile_gpu_samples": sum(run["gpu_samples"] for run in runs),
        "cpu_frame_p50_ms": cpu_p50,
        "cpu_frame_p95_ms": run_median("cpu_p95_ms"),
        "gpu_frame_p50_ms": gpu_p50,
        "gpu_frame_p95_ms": run_median("gpu_p95_ms"),
        "fps_profile_cpu_p50": 1000.0 / cpu_p50 if cpu_p50 and cpu_p50 > 0 else None,
        "fps_profile_gpu_p50": 1000.0 / gpu_p50 if gpu_p50 and gpu_p50 > 0 else None,
        "load_to_ready_ms": run_median("load_to_ready_ms"),
        "rendering_memory_mib": run_median("rendering_memory_mib"),
        "unity_allocated_mib": run_median("unity_allocated_mib"),
        "unity_reserved_mib": run_median("unity_reserved_mib"),
        "asset_memory_mib": run_median("asset_memory_mib"),
    }


def read_protocol(root: Path) -> dict[str, Any]:
    path = root / "msbench_run_protocol.json"
    return json.loads(path.read_text()) if path.is_file() else {}



def scene_row(
    label: str,
    root: Path,
    method: str,
    condition: str,
    runtime: str,
    profile_runs: int | None,
) -> dict[str, Any]:
    dataset, scene = label.split("/", 1)
    protocol = read_protocol(root)
    row = {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "condition": condition,
        "topology": protocol.get("mesh_topology_layout", "indexed"),
        "runtime": runtime,
        **read_quality(root),
        **read_capture_fps(root),
        **read_profiles(root, profile_runs),
    }
    asset = protocol.get("asset")
    if row.get("asset_memory_mib") is None and asset:
        asset_path = Path(asset)
        if asset_path.is_dir():
            row["asset_memory_mib"] = directory_bytes(asset_path) / (1024.0 * 1024.0)
    return row


def aggregate_row(rows: list[dict[str, Any]], method: str, condition: str, runtime: str) -> dict[str, Any]:
    excluded = {"dataset", "scene", "method", "condition", "topology", "runtime", "views"}
    topologies = sorted({str(row.get("topology") or "indexed") for row in rows})
    numeric_keys = sorted({key for row in rows for key in row if key not in excluded})
    result: dict[str, Any] = {
        "dataset": "ALL",
        "scene": "MEAN",
        "method": method,
        "condition": condition,
        "topology": topologies[0] if len(topologies) == 1 else "mixed",
        "runtime": runtime,
        "views": sum(int(row.get("views") or 0) for row in rows),
    }
    for key in numeric_keys:
        values = [value for row in rows if (value := finite(row.get(key))) is not None]
        result[key] = statistics.fmean(values) if values else None
    return result


REPORT_COLUMNS = (
    "dataset", "scene", "method", "condition", "topology", "runtime", "views",
    "psnr", "ssim", "lpips_vgg",
    "native_psnr", "native_ssim", "native_lpips_vgg",
    "fps_profile_gpu_p50", "gpu_frame_p50_ms", "gpu_frame_p95_ms", "profile_gpu_samples",
    "rendering_memory_mib", "unity_allocated_mib", "unity_reserved_mib",
    "asset_memory_mib", "load_to_ready_ms", "profile_runs",
)


def display(value: Any, digits: int = 4) -> str:
    number = finite(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        ("Dataset/Scene", lambda row: f"{row['dataset']}/{row['scene']}"),
        ("PSNR", lambda row: display(row.get("psnr"))),
        ("SSIM", lambda row: display(row.get("ssim"))),
        ("LPIPS-VGG", lambda row: display(row.get("lpips_vgg"))),
        ("FPS (GPU P50)", lambda row: display(row.get("fps_profile_gpu_p50"), 2)),
        ("GPU P50 ms", lambda row: display(row.get("gpu_frame_p50_ms"), 3)),
        ("GPU P95 ms", lambda row: display(row.get("gpu_frame_p95_ms"), 3)),
        ("Render Mem MiB", lambda row: display(row.get("rendering_memory_mib"), 1)),
        ("Asset MiB", lambda row: display(row.get("asset_memory_mib"), 1)),
        ("Load ms", lambda row: display(row.get("load_to_ready_ms"), 1)),
    )
    lines = [
        "| " + " | ".join(name for name, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(render(row) for _, render in columns) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n")


def report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in REPORT_COLUMNS}


def print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'Dataset/Scene':<26} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} "
        f"{'GPU FPS':>10} {'GPU P50 ms':>11} {'GPU P95 ms':>11} "
        f"{'Render MiB':>12} {'Asset MiB':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = f"{row['dataset']}/{row['scene']}"
        print(
            f"{label:<26} {display(row.get('psnr')):>9} {display(row.get('ssim')):>9} "
            f"{display(row.get('lpips_vgg')):>9} {display(row.get('fps_profile_gpu_p50'), 2):>10} "
            f"{display(row.get('gpu_frame_p50_ms'), 3):>11} "
            f"{display(row.get('gpu_frame_p95_ms'), 3):>11} "
            f"{display(row.get('rendering_memory_mib'), 1):>12} {display(row.get('asset_memory_mib'), 1):>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="DATASET/SCENE=PATH")
    parser.add_argument("--method", required=True)
    parser.add_argument("--condition", choices=("method-aware", "general-purpose"), required=True)
    parser.add_argument("--runtime", default="editor-development-only")
    parser.add_argument("--profile-runs", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="unity_metrics_report")
    args = parser.parse_args()
    if args.profile_runs is not None and args.profile_runs <= 0:
        parser.error("--profile-runs must be positive")

    entries = []
    for value in args.result:
        if "=" not in value:
            parser.error(f"invalid --result value: {value!r}")
        label, raw_path = value.split("=", 1)
        if label.count("/") != 1:
            parser.error(f"result label must be DATASET/SCENE: {label!r}")
        entries.append((label, Path(raw_path).expanduser().resolve()))

    rows = [
        scene_row(label, root, args.method, args.condition, args.runtime, args.profile_runs)
        for label, root in entries
    ]
    aggregate = aggregate_row(rows, args.method, args.condition, args.runtime)
    report_rows = [*rows, aggregate]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.name

    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)
    payload = {
        "protocol": {
            "report_revision": REPORT_REVISION,
            "method": args.method,
            "condition": args.condition,
            "runtime": args.runtime,
            "quality": "raw RGB code values; Gaussian SSIM; full-frame LPIPS-VGG",
            "fps_profile": "1000 / median per-run pooled GPU P50 frame time; no image readback or file I/O; N/A when Unity reports no GPU frame timings",
            "capture_fps": "computed internally from engine frame time during capture but excluded from paper report columns",
            "rendering_memory": "median Unity Profiler graphics-driver allocated bytes across profile runs",
            "asset_memory": "serialized TriAsset directory bytes",
            "aggregate": "macro mean of scene metrics; views are summed",
        },
        "scenes": [report_row(row) for row in rows],
        "aggregate": report_row(aggregate),
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(prefix.with_suffix(".md"), report_rows)
    print_summary(report_rows)
    print(f"CSV: {prefix.with_suffix('.csv')}")
    print(f"JSON: {prefix.with_suffix('.json')}")
    print(f"Markdown: {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
