#!/usr/bin/env python3
"""Create per-method deployment tables from repeated Unity runtime profiles."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
QUALITY = {
    "2dts": {
        "common": "unity_standard_mesh_test_eval_unity6_metal_srgb_summary.json",
        "faithful": "unity_method_specific_test_eval_unity6_metal_srgb_summary.json",
    },
    "diffsoup": {
        "common": "unity_standard_mesh_test_eval_unity6_metal_srgb_summary.json",
        "faithful": "unity_method_specific_test_eval_unity6_metal_srgb_summary.json",
    },
    "mesh-splatting": {
        "common": "unity_standard_mesh_opacity_floor_fixed_test_eval_unity6_metal_srgb_summary.json",
        "faithful": "unity_method_specific_terminal_solid_test_eval_unity6_metal_srgb_summary.json",
    },
    "triangle-splatting": {
        "common": "unity_standard_mesh_test_eval_unity6_metal_srgb_summary.json",
        "faithful": "unity_method_specific_test_eval_unity6_metal_srgb_summary.json",
    },
}


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def fmt_unc(values: list[float], digits: int = 2) -> str:
    if not values: return "--"
    return f"{statistics.median(values):.{digits}f} $\\pm$ {std(values):.{digits}f}"


def quality_by_scene(root: Path, method: str, mode: str) -> dict[str, dict]:
    path = root / "outputs" / method / "mipnerf360" / QUALITY[method][mode]
    data = json.loads(path.read_text())
    return {item["scene"]: item for item in data["scenes"] if item["split"] == "test"}


def profile_scene(root: Path, method: str, mode: str, scene: str) -> dict:
    files = sorted((root / method / mode / scene).glob("run_*/runtime_profile_run_*.json"))
    runs = [json.loads(path.read_text()) for path in files]
    per_run: list[dict] = []
    for run in runs:
        views = run.get("views", [])
        cpu = [x for v in views for x in v.get("cpu_frame_samples_ms", [])]
        gpu = [x for v in views for x in v.get("gpu_frame_samples_ms", []) if x > 0.01]
        draws = [x for v in views for x in v.get("draw_call_samples_values", [])]
        per_run.append({
            "cpu_p50": quantile(cpu, .50), "cpu_p95": quantile(cpu, .95),
            "gpu_p50": quantile(gpu, .50), "gpu_p95": quantile(gpu, .95),
            "load_s": run.get("load_to_ready_ms", 0.0) / 1000.0,
            "gfx_mib": run.get("graphics_driver_allocated_bytes", 0) / 1024**2,
            "asset_mib": run.get("asset_bytes", 0) / 1024**2,
            "draw_p50": quantile(draws, .50), "draw_p95": quantile(draws, .95),
            "draw_source": next((v.get("draw_call_source") for v in views if v.get("draw_call_source")), "--"),
            "gpu_complete": len(gpu) >= 0.8 * sum(len(v.get("engine_frame_samples_ms", [])) for v in views),
        })
    valid_gpu = [r for r in per_run if r["gpu_complete"]]
    def values(key: str, subset: list[dict] = per_run) -> list[float]: return [r[key] for r in subset if r[key] is not None]
    return {
        "runs": len(per_run),
        "cpu_p50": values("cpu_p50"), "cpu_p95": values("cpu_p95"),
        "gpu_p50": values("gpu_p50", valid_gpu), "gpu_p95": values("gpu_p95", valid_gpu),
        "load_s": values("load_s"), "gfx_mib": values("gfx_mib"), "asset_mib": values("asset_mib"),
        "draw_p50": values("draw_p50"), "draw_p95": values("draw_p95"),
        "draw_source": per_run[0]["draw_source"] if per_run else "--",
        "gpu_status": "available" if valid_gpu else "unavailable on macOS Metal via Unity FrameTimingManager",
    }


def summary(values: list[dict], key: str) -> list[float]:
    return [statistics.median(v[key]) for v in values if v[key]]


def table_for_method(profile_root: Path, msbench_root: Path, method: str) -> str:
    qualities = {mode: quality_by_scene(msbench_root, method, mode) for mode in ("common", "faithful")}
    rows: list[tuple[str, str, dict, dict]] = []
    for scene in SCENES:
        for mode in ("common", "faithful"):
            rows.append((scene, mode, qualities[mode][scene], profile_scene(profile_root, method, mode, scene)))
    out = [f"## {method}", "", "| Scene | Mode | PSNR | SSIM | VGG-LPIPS | CPU P50 / P95 (ms) | GPU P50 / P95 (ms) | Graphics-driver allocation (MiB) | Load-to-ready (s) | Asset (MiB) | Draw submissions/frame | Runs |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scene, mode, q, p in rows:
        gpu = "--" if not p["gpu_p50"] else f"{fmt_unc(p['gpu_p50'])} / {fmt_unc(p['gpu_p95'])}"
        draws = "--" if not p["draw_p50"] else f"{fmt_unc(p['draw_p50'], 1)} / {fmt_unc(p['draw_p95'], 1)}"
        out.append(
            f"| {scene} | {mode} | {q['psnr']:.3f} | {q['ssim']:.4f} | {q['lpips_vgg']:.4f} | "
            f"{fmt_unc(p['cpu_p50'])} / {fmt_unc(p['cpu_p95'])} | {gpu} | {fmt_unc(p['gfx_mib'], 1)} | "
            f"{fmt_unc(p['load_s'], 2)} | {fmt_unc(p['asset_mib'], 1)} | {draws} | {p['runs']} |"
        )
    for mode in ("common", "faithful"):
        selected = [(q, p) for _, m, q, p in rows if m == mode]
        group = [p for _, p in selected]
        psnr = mean([q["psnr"] for q, _ in selected])
        ssim = mean([q["ssim"] for q, _ in selected])
        lpips = mean([q["lpips_vgg"] for q, _ in selected])
        out.append(
            f"| **Mean over scenes** | {mode} | {fmt(psnr, 3)} | {fmt(ssim, 4)} | {fmt(lpips, 4)} | {fmt(mean(summary(group, 'cpu_p50')))} / {fmt(mean(summary(group, 'cpu_p95')))} | "
            f"{'--' if not any(p['gpu_p50'] for p in group) else 'see per-scene'} | {fmt(mean(summary(group, 'gfx_mib')), 1)} | "
            f"{fmt(mean(summary(group, 'load_s')), 2)} | {fmt(mean(summary(group, 'asset_mib')), 1)} | {fmt(mean(summary(group, 'draw_p50')), 1)} | 3 |"
        )
    sources = sorted({p["draw_source"] for _, _, _, p in rows})
    status = sorted({p["gpu_status"] for _, _, _, p in rows})
    out += ["", f"CPU/load/allocation uncertainty is run-to-run sample standard deviation across three fresh headless Unity Editor launches. The Graphics-driver counter is a Unity allocation counter on Apple-silicon unified memory, not a separate physical-VRAM measurement. Draw-call source: {', '.join(sources)}. GPU status: {', '.join(status)}.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--msbench-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sections = ["# MeshSplatBench Unity deployment profile", "", "Quality values reuse the completed matched-camera Unity evaluations. CPU timing, load time, Unity graphics-driver allocation, and submissions are collected from repeated headless Unity Editor launches. GPU columns remain unavailable unless a valid GPU counter backend supplies at least 80% of timed frames.", ""]
    for method in QUALITY: sections.append(table_for_method(args.profile_root, args.msbench_root, method))
    args.output.write_text("\n".join(sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
