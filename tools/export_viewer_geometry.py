#!/usr/bin/env python3
"""Export original CG-readable geometry from a trained TriBench checkpoint.

Usage:
  python tools/export_viewer_geometry.py \
      --method 2dts \
      --checkpoint outputs/2dts/mipnerf360/garden/ckpt/30000.ckpt \
      --output-dir /tmp/tribench-geometry

  python tools/export_viewer_geometry.py \
      --config configs/triangle-splatting/dtu/scan24.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure tribench is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(
        description="Export original CG-readable geometry from a TriBench checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="TriBench config YAML path")
    group.add_argument(
        "--method",
        type=str,
        help="Method name (triangle-splatting, mesh-splatting, 2dts, diffsoup)",
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="Checkpoint path (required with --method; overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default with --config: <output.dir>/viewer)",
    )
    parser.add_argument("--export-glb", action="store_true", help="Also export mesh GLB")
    parser.add_argument(
        "--filename",
        type=str,
        default="geometry_original.ply",
        help="Output PLY filename (default: geometry_original.ply)",
    )
    args = parser.parse_args()

    from tribench.core.builder import build_adapter
    from tribench.core.viewer_geometry import export_original_geometry
    from tribench.cli.config import (
        adapter_config,
        load_cli_config,
        output_dir as config_output_dir,
    )

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

    print(f"[export] method={method}, checkpoint={checkpoint}")
    print(f"[export] output_dir={output_dir}")
    adapter = build_adapter(adapter_cfg)

    try:
        primitive = adapter.to_primitive()
    except Exception as e:
        print(f"[export] ERROR: to_primitive() failed: {e}", file=sys.stderr)
        sys.exit(1)

    num_primitives = getattr(primitive, "num_primitives", None)
    if num_primitives is None:
        vertices = getattr(primitive, "vertices", None)
        num_primitives = len(vertices) if vertices is not None else "unknown"
    print(f"[export] primitive: {num_primitives} triangles")

    result = export_original_geometry(
        primitive,
        output_dir,
        method_name=str(method),
        checkpoint_path=str(checkpoint),
        filename=args.filename,
        export_glb=args.export_glb,
    )

    print(f"[export] geometry PLY: {result.geometry_ply}")
    print(f"[export] vertices:     {result.num_vertices}")
    print(f"[export] faces:        {result.num_faces}")
    if result.mesh_glb_path:
        print(f"[export] mesh GLB:     {result.mesh_glb_path}")
    if result.metadata_path:
        print(f"[export] metadata:   {result.metadata_path}")
    if result.attributes_path:
        print(f"[export] attributes: {result.attributes_path}")
    print("[export] done.")


if __name__ == "__main__":
    main()
