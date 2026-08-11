#!/usr/bin/env python3
"""Dependency-light CPU entry point for revision-2 Unity TriAsset export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tribench.unity_assets import export_triasset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mesh-opacity-floor", type=float)
    parser.add_argument(
        "--export-topology",
        default="indexed",
        help="MeshSplatting export layout: indexed/mesh or soup/materialized-soup.",
    )
    parser.add_argument("--background-color", choices=("black", "white"))
    gamma = parser.add_mutually_exclusive_group()
    gamma.add_argument(
        "--d2ts-gamma-rescale", dest="d2ts_gamma_rescale", action="store_true"
    )
    gamma.add_argument(
        "--no-d2ts-gamma-rescale", dest="d2ts_gamma_rescale", action="store_false"
    )
    parser.set_defaults(d2ts_gamma_rescale=None)
    args = parser.parse_args()

    package = export_triasset(
        args.method,
        args.checkpoint,
        args.output,
        overwrite=args.force,
        mesh_opacity_floor=args.mesh_opacity_floor,
        export_topology=args.export_topology,
        d2ts_gamma_rescale=args.d2ts_gamma_rescale,
        background_color=args.background_color,
    )
    print(f"Unity-native asset: {package.path}")
    print(f"Manifest: {package.manifest_path}")
    print(f"Method: {package.method}; buffers: {len(package.buffers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
