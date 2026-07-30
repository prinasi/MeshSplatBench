#!/usr/bin/env python3
"""Aggregate strict per-scene Unity quality and Native-fidelity reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")


def mean(values):
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/triangle-splatting/mipnerf360"))
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()

    rows = []
    for scene in args.scenes:
        root = args.outputs_root / scene / args.output_name
        metrics_path = root / "metrics_summary.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing scene report: {metrics_path}")
        summary = json.loads(metrics_path.read_text())
        if summary.get("protocol", {}).get("evaluator_revision") != 1:
            raise ValueError(f"stale or non-canonical evaluator report: {metrics_path}")
        values = summary["test"]
        if values["count"] <= 0:
            raise ValueError(f"empty test result: {metrics_path}")
        row = {"scene": scene, "views": values["count"], **{key: value for key, value in values.items() if key != "count"}}
        native = summary.get("native_fidelity")
        if native is None or native.get("count") != values["count"]:
            raise ValueError(f"missing or incomplete Native-fidelity result: {metrics_path}")
        row.update({f"native_{key}": value for key, value in native.items() if key != "count"})
        rows.append(row)

    if len(rows) != len(args.scenes):
        raise ValueError(f"expected {len(args.scenes)} scenes, found {len(rows)}")

    fieldnames = sorted({key for row in rows for key in row})
    output_csv = args.outputs_root / f"{args.output_name}_summary.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    metric_keys = [key for key in rows[0] if key not in {"scene", "views"}]
    aggregate = {
        "protocol": {
            "quality_mean": "macro mean of per-scene means",
            "evaluator_revision": 1,
            "timing": "not mixed into quality aggregation; use standalone Player deployment profile",
        },
        "scenes": rows,
        "test": {
            "scene_count": len(rows),
            "views_total": sum(row["views"] for row in rows),
            **{f"{key}_macro": mean([row[key] for row in rows]) for key in metric_keys},
        },
    }
    output_json = args.outputs_root / f"{args.output_name}_summary.json"
    output_json.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(output_csv)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
