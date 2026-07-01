#!/usr/bin/env python3
"""Export dual point cloud geometry from a trained TriBench checkpoint.

Usage:
  python tools/export_viewer_geometry.py \
      --method 2dts \
      --checkpoint outputs/2dts/mipnerf360/garden/ckpt/30000.ckpt \
      --output-dir /tmp/tribench-geometry

  python tools/export_viewer_geometry.py \
      --config configs/triangle-splatting/dtu/scan24.yaml \
      --output-dir /tmp/tribench-geometry
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
        description="Export viewer + CG point clouds from a TriBench checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="TriBench config YAML path")
    group.add_argument("--method", type=str, help="Method name (triangle-splatting, mesh-splatting, 2dts, diffsoup)")

    parser.add_argument("--checkpoint", "-c", type=str, help="Checkpoint path (required with --method)")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--viewer-num-points", type=int, default=500_000, help="Viewer point cloud size (default: 500K)")
    parser.add_argument("--cg-num-points", type=int, default=3_000_000, help="CG point cloud size (default: 3M)")
    parser.add_argument("--color-mode", choices=["dc", "opacity", "white", "rendered"], default="dc")
    parser.add_argument("--normal-mode", choices=["none", "primitive", "estimated"], default="none")
    parser.add_argument("--export-mesh", action="store_true", help="Also export mesh PLY")
    parser.add_argument("--export-glb", action="store_true", help="Export GLB (future)")
    parser.add_argument("--no-voxel-downsample", action="store_true", help="Disable voxel downsampling of viewer cloud")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size for downsampling")
    args = parser.parse_args()

    # Build adapter
    from tribench.core.builder import build_adapter
    from tribench.core.viewer_geometry import GeometryExportConfig, export_viewer_geometry

    if args.config:
        from tribench.core.config import load_config
        cfg = load_config(args.config)
        method = cfg.get("method", "triangle-splatting")
        checkpoint = cfg.get("checkpoint", None)
        dataset_path = cfg.get("dataset", {}).get("path", None)
    else:
        method = args.method
        checkpoint = args.checkpoint
        dataset_path = None

    if not checkpoint:
        parser.error("--checkpoint is required when using --method")

    print(f"[export] method={method}, checkpoint={checkpoint}")
    adapter = build_adapter(method)
    adapter.load_checkpoint(checkpoint)

    # Extract primitive
    try:
        primitive = adapter.to_primitive()
        print(f"[export] primitive: {primitive.num_primitives} triangles")
    except Exception as e:
        print(f"[export] ERROR: to_primitive() failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Export
    geo_cfg = GeometryExportConfig(
        viewer_num_points=args.viewer_num_points,
        cg_num_points=args.cg_num_points,
        color_mode=args.color_mode,
        normal_mode=args.normal_mode,
        export_mesh=args.export_mesh,
        export_glb=args.export_glb,
        voxel_downsample=not args.no_voxel_downsample,
        voxel_size=args.voxel_size,
    )

    result = export_viewer_geometry(
        primitive,
        args.output_dir,
        method_name=method,
        checkpoint_path=checkpoint,
        config=geo_cfg,
    )

    print(f"[export] viewer PLY: {result.viewer_ply} ({result.num_viewer_points} pts)")
    print(f"[export] CG PLY:     {result.cg_ply} ({result.num_cg_points} pts)")
    if result.mesh_ply:
        print(f"[export] mesh PLY:   {result.mesh_ply}")
    if result.metadata_path:
        print(f"[export] metadata:   {result.metadata_path}")
    if result.attributes_path:
        print(f"[export] attributes: {result.attributes_path}")
    print("[export] done.")


if __name__ == "__main__":
    main()
