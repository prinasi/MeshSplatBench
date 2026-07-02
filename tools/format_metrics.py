#!/usr/bin/env python3
"""Format TriBench metrics.json files as a console table or CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tribench.core.config import Config  # noqa: E402


MIPNERF360_OUTDOOR_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill"]
MIPNERF360_INDOOR_SCENES = ["room", "counter", "kitchen", "bonsai"]
MIPNERF360_SCENES = MIPNERF360_OUTDOOR_SCENES + MIPNERF360_INDOOR_SCENES
TANKS_AND_TEMPLES_SCENES = ["truck", "train"]
NERF_SYNTHETIC_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
DTU_SCENES = [
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
    "scan105",
    "scan106",
    "scan110",
    "scan114",
    "scan118",
    "scan122",
]

DATASET_SCENES = {
    "mipnerf360": MIPNERF360_SCENES,
    "tandt": TANKS_AND_TEMPLES_SCENES,
    "nerf_synthetic": NERF_SYNTHETIC_SCENES,
    "dtu": DTU_SCENES,
}

HEADERS = [
    "Dataset",
    "Scene",
    "PSNR",
    "SSIM",
    "LPIPS",
    "FPS",
    "Training Memory (MiB)",
    "Training Time (s)",
    "CD",
    "Metrics File",
]


@dataclass
class MetricsTarget:
    method: str
    dataset: str
    scene: str
    metrics_file: Path
    mesh_metrics_file: Path | None = None


@dataclass
class MetricsRow:
    dataset: str
    scene: str
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    fps: float | None = None
    training_memory_mib: float | None = None
    training_time_s: float | None = None
    cd: float | None = None
    metrics_file: str = ""

    def as_cells(self) -> list[str]:
        return [
            self.dataset,
            self.scene,
            _fmt_float(self.psnr, 4),
            _fmt_float(self.ssim, 4),
            _fmt_float(self.lpips, 4),
            _fmt_float(self.fps, 2),
            _fmt_float(self.training_memory_mib, 0),
            _fmt_float(self.training_time_s, 1),
            _fmt_float(self.cd, 4),
            self.metrics_file,
        ]


def canonical_method(name: str) -> str:
    method = name.lower().replace("_", "-")
    aliases = {
        "trianglesplatting": "triangle-splatting",
        "meshsplatting": "mesh-splatting",
        "2dts": "2dts",
        "d2ts": "2dts",
    }
    return aliases.get(method, method)


def canonical_dataset(name: str) -> str:
    dataset = name.lower().replace("_", "-")
    aliases = {
        "mipnerf-360": "mipnerf360",
        "m360": "mipnerf360",
        "360": "mipnerf360",
        "nerf360": "mipnerf360",
        "tanksandtemples": "tandt",
        "tanks-and-temples": "tandt",
        "tat": "tandt",
        "tanks": "tandt",
        "nerfsynthetic": "nerf_synthetic",
        "nerf-synthetic": "nerf_synthetic",
        "blender": "nerf_synthetic",
        "synthetic": "nerf_synthetic",
    }
    return aliases.get(dataset, dataset)


def parse_scene_list(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    scenes: list[str] = []
    for item in raw:
        scenes.extend(part.strip() for part in item.split(",") if part.strip())
    return scenes or None


def expand_targets_from_dataset(
    method: str,
    dataset_arg: str,
    *,
    scenes: list[str] | None,
    outputs_root: Path,
) -> list[MetricsTarget]:
    dataset_text = dataset_arg.strip("/")
    if "/" in dataset_text:
        dataset_name, scene_part = dataset_text.split("/", 1)
    else:
        dataset_name, scene_part = dataset_text, None

    dataset = canonical_dataset(dataset_name)
    if scenes is None:
        if scene_part in {None, "", "all"}:
            scenes = DATASET_SCENES.get(dataset)
            if scenes is None:
                raise SystemExit(f"Unknown dataset '{dataset_arg}'. Provide --scenes for custom datasets.")
        else:
            scenes = [scene_part]

    return [
        target_from_layout(
            method=canonical_method(method),
            dataset=dataset,
            scene=scene,
            outputs_root=outputs_root,
        )
        for scene in scenes
    ]


def target_from_layout(method: str, dataset: str, scene: str, outputs_root: Path) -> MetricsTarget:
    run_dir = outputs_root / method / dataset / scene
    return MetricsTarget(
        method=method,
        dataset=dataset,
        scene=scene,
        metrics_file=run_dir / "metrics.json",
        mesh_metrics_file=run_dir / "mesh_metrics.json" if dataset == "dtu" else None,
    )


def target_from_config(path: Path) -> MetricsTarget:
    cfg = Config.fromfile(path)
    dataset_cfg = cfg.get("dataset", {}) or {}
    trainer_cfg = cfg.get("trainer", {}) or {}
    adapter_cfg = cfg.get("adapter", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    method = canonical_method(
        str(adapter_cfg.get("type") or trainer_cfg.get("type") or trainer_cfg.get("name") or path.parent.name)
    )
    dataset = canonical_dataset(
        str(dataset_cfg.get("name") or _infer_dataset_from_root(dataset_cfg.get("root")) or "unknown")
    )
    scene = str(dataset_cfg.get("scene") or Path(str(dataset_cfg.get("root", path.stem))).name)

    metrics_value = eval_cfg.get("output") or eval_cfg.get("metrics_file") or output_cfg.get("metrics_file")
    if metrics_value is None:
        out_dir = output_cfg.get("dir")
        metrics_file = Path(out_dir) / "metrics.json" if out_dir else Path("outputs") / method / dataset / scene / "metrics.json"
    else:
        metrics_file = Path(str(metrics_value))

    mesh_value = output_cfg.get("mesh_metrics_file")
    mesh_metrics_file = Path(str(mesh_value)) if mesh_value else Path("outputs") / method / dataset / scene / "mesh_metrics.json"
    return MetricsTarget(
        method=method,
        dataset=dataset,
        scene=scene,
        metrics_file=metrics_file,
        mesh_metrics_file=mesh_metrics_file if dataset == "dtu" else None,
    )


def _infer_dataset_from_root(root: Any) -> str | None:
    if root is None:
        return None
    parts = {part.lower() for part in Path(str(root)).parts}
    if {"dtu", "dtu_official"} & parts:
        return "dtu"
    if {"tandt", "tanksandtemples", "tanks-and-temples"} & parts:
        return "tandt"
    if {"mipnerf360", "mipnerf-360", "360_v2"} & parts:
        return "mipnerf360"
    if {"nerf_synthetic", "nerf-synthetic", "blender"} & parts:
        return "nerf_synthetic"
    return None


def read_row(target: MetricsTarget) -> MetricsRow:
    metrics_path = _resolve_path(target.metrics_file)
    row = MetricsRow(
        dataset=target.dataset,
        scene=target.scene,
        metrics_file=str(metrics_path),
    )
    if metrics_path.exists():
        data = _read_json(metrics_path)
        aggregate = data.get("aggregate", {}) if isinstance(data, dict) else {}
        inference = data.get("inference", {}) if isinstance(data, dict) else {}
        training = data.get("training", {}) if isinstance(data, dict) else {}
        row.psnr = _number(aggregate.get("psnr_mean"))
        row.ssim = _number(aggregate.get("ssim_mean"))
        row.lpips = _number(aggregate.get("lpips_mean"))
        row.fps = (
            _number(inference.get("gpu_fps"))
            or _number(data.get("inference_fps"))
            or _number(inference.get("fps"))
        )
        row.training_memory_mib = _number(data.get("training_peak_gpu_memory_mib")) or _number(
            training.get("peak_gpu_memory_mib")
        )
        row.training_time_s = _number(data.get("training_time_s")) or _number(training.get("total_time_s"))
    else:
        row.metrics_file = f"{metrics_path} (missing)"

    if target.mesh_metrics_file is not None:
        mesh_path = _resolve_path(target.mesh_metrics_file)
        if mesh_path.exists():
            mesh = _read_json(mesh_path)
            row.cd = _first_number(mesh, ["overall", "chamfer_distance", "chamfer", "cd", "mean"])
    return row


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(data: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _number(data.get(key))
        if value is not None:
            return value
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            value = _number(metrics.get(key))
            if value is not None:
                return value
    return None


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return "N/A"
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def mean_row(rows: list[MetricsRow]) -> MetricsRow:
    dataset = rows[0].dataset if rows else "N/A"
    return MetricsRow(
        dataset=dataset,
        scene="MEAN",
        psnr=_mean(row.psnr for row in rows),
        ssim=_mean(row.ssim for row in rows),
        lpips=_mean(row.lpips for row in rows),
        fps=_mean(row.fps for row in rows),
        training_memory_mib=_mean(row.training_memory_mib for row in rows),
        training_time_s=_mean(row.training_time_s for row in rows),
        cd=_mean(row.cd for row in rows),
        metrics_file="",
    )


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def print_table(rows: list[MetricsRow]) -> None:
    cells = [HEADERS] + [row.as_cells() for row in rows]
    widths = [max(len(record[i]) for record in cells) for i in range(len(HEADERS))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*HEADERS))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row.as_cells()))


def write_csv(rows: list[MetricsRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for row in rows:
            writer.writerow(row.as_cells())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Format TriBench metrics.json files from configs or output layout."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", "-c", nargs="+", type=Path, help="One or more TriBench config YAML files.")
    source.add_argument("--dataset", "-d", help="Dataset target, e.g. dtu, dtu/scan24, mipnerf360/all.")
    parser.add_argument("--method", "-m", help="Method name when using --dataset, e.g. triangle-splatting.")
    parser.add_argument("--scenes", nargs="+", help="Scene list for partial datasets; comma-separated is accepted.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root output directory.")
    parser.add_argument("--output", choices=["console", "csv"], default="console", help="Output destination type.")
    parser.add_argument("--csv", type=Path, help="CSV output path. Implies --output csv.")
    parser.add_argument("--mean", dest="mean", action="store_true", default=True, help="Append a mean row for multi-scene output.")
    parser.add_argument("--no-mean", dest="mean", action="store_false", help="Do not append a mean row.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_kind = "csv" if args.csv is not None else args.output
    targets: list[MetricsTarget]
    if args.config:
        targets = [target_from_config(path) for path in args.config]
    else:
        if not args.method:
            raise SystemExit("--method is required when using --dataset.")
        targets = expand_targets_from_dataset(
            args.method,
            args.dataset,
            scenes=parse_scene_list(args.scenes),
            outputs_root=args.outputs_root,
        )

    rows = [read_row(target) for target in targets]
    if args.mean and len(rows) > 1:
        rows.append(mean_row(rows))

    if output_kind == "csv":
        csv_path = args.csv or Path("metrics_summary.csv")
        write_csv(rows, csv_path)
        print(f"Wrote {csv_path}")
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
