#!/usr/bin/env python3
"""Create CG-compatible geometry from any TriBench method checkpoint.

The filename is intentionally kept as ``create_ply.py`` for compatibility with
older workflows, but the default export is OBJ + MTL because Unity imports OBJ
without a custom PLY importer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _filename_with_format(filename: str | None, fmt: str) -> str:
    if not filename:
        return f"geometry_original.{fmt}"
    path = Path(filename)
    if path.suffix:
        return str(path)
    return str(path.with_suffix(f".{fmt}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export original topology as OBJ/PLY/GLB from triangle-splatting, "
            "mesh-splatting, 2dts, or diffsoup checkpoints."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="TriBench config YAML path")
    group.add_argument(
        "--method",
        type=str,
        help="Method name: triangle-splatting, mesh-splatting, 2dts, or diffsoup",
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        "--checkpoint_path",
        "-c",
        dest="checkpoint",
        type=str,
        help="Checkpoint path; required with --method and overrides config",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default with --config: <output.dir>/viewer)",
    )
    parser.add_argument(
        "--output-name",
        "--output_name",
        "--filename",
        dest="filename",
        type=str,
        help="Output filename. Extension overrides --format.",
    )
    parser.add_argument(
        "--format",
        choices=("obj", "ply", "glb"),
        default="obj",
        help="Default output format when --output-name has no extension",
    )
    parser.add_argument("--export-glb", action="store_true", help="Also export GLB")
    parser.add_argument("--export-obj", action="store_true", help="Also export OBJ/MTL")
    args = parser.parse_args()

    from tribench.cli.config import (
        adapter_config,
        load_cli_config,
        output_dir as config_output_dir,
    )
    from tribench.core.builder import build_adapter
    from tribench.core.viewer_geometry import export_original_geometry

    if args.config:
        cfg = load_cli_config(args.config)
        adapter_cfg = adapter_config(cfg, checkpoint=args.checkpoint)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            run_dir = config_output_dir(cfg)
            if run_dir is None:
                parser.error("--output-dir is required when config output.dir is not set")
            output_dir = Path(run_dir) / "viewer"
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required when using --method")
        adapter_cfg = adapter_config(None, method=args.method, checkpoint=args.checkpoint)
        if not args.output_dir:
            parser.error("--output-dir is required when using --method")
        output_dir = Path(args.output_dir)

    method = adapter_cfg.get("type")
    checkpoint = adapter_cfg.get("checkpoint")
    if method is None or checkpoint is None:
        parser.error("Could not resolve adapter type and checkpoint")

    print(f"[create_ply] method={method}, checkpoint={checkpoint}")
    print(f"[create_ply] output_dir={output_dir}")
    adapter = build_adapter(adapter_cfg)

    try:
        primitive = adapter.to_primitive()
    except Exception as exc:
        print(f"[create_ply] ERROR: to_primitive() failed: {exc}", file=sys.stderr)
        sys.exit(1)

    filename = _filename_with_format(args.filename, args.format)
    result = export_original_geometry(
        primitive,
        output_dir,
        method_name=str(method),
        checkpoint_path=str(checkpoint),
        filename=filename,
        export_obj=args.export_obj,
        export_glb=args.export_glb,
    )

    print(f"[create_ply] geometry: {result.geometry_path}")
    if result.mesh_obj_path:
        print(f"[create_ply] OBJ/MTL:  {result.mesh_obj_path}")
    if result.geometry_ply:
        print(f"[create_ply] PLY:      {result.geometry_ply}")
    if result.mesh_glb_path:
        print(f"[create_ply] GLB:      {result.mesh_glb_path}")
    print(f"[create_ply] vertices: {result.num_vertices}")
    print(f"[create_ply] faces:    {result.num_faces}")


if __name__ == "__main__":
    main()
