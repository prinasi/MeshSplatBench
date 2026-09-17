#!/usr/bin/env python3
"""Render camera trajectory videos using the Unity-native TriAsset renderer.

Renders videos for method-aware (procedural splatting) or general-purpose
(standard Unity Mesh) conditions along the exact same PCA ellipse camera
trajectory as ``msbench render video``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path when executed directly
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from msbench.unity_video import (
    INDOOR_SCENES,
    SCENES,
    encode_video_from_frames,
    find_unity_executable,
    find_unity_project,
    generate_trajectory_from_dataset,
    reference_image_directory,
    run_unity_video_for_scene,
    unity_failure_summary,
    unity_graphics_arguments,
    validate_captured_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render video trajectories in Unity for MeshSplatBench assets.",
    )
    parser.add_argument("--unity", type=Path, help="Path to Unity Editor binary.")
    parser.add_argument("--unity-project", type=Path, help="Path to MeshSplatBench Unity project.")
    parser.add_argument("--config", type=Path, help="MeshSplatBench render/training config YAML.")
    parser.add_argument("--method", "-m", help="Method name (e.g. triangle-splatting, mesh-splatting, 2dts, diffsoup).")
    parser.add_argument("--triasset", type=Path, help="Path to source .triasset directory.")
    parser.add_argument("--dataset", "-d", type=Path, help="Path to dataset directory.")
    parser.add_argument("--datasets-root", type=Path, help="Root directory containing scenes.")
    parser.add_argument("--outputs-root", type=Path, help="Outputs root directory.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names to process.")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for the video.")
    parser.add_argument("--output-name", help="Output subfolder name under run directory.")
    parser.add_argument("--trajectory", "-t", type=Path, help="Existing trajectory JSON file.")
    parser.add_argument("--frames", type=int, default=240, help="Number of frames in ellipse trajectory.")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Focal length multiplier.")
    parser.add_argument("--z-variation", type=float, default=0.0, help="Vertical oscillation amplitude.")
    parser.add_argument("--z-phase", type=float, default=0.0, help="Phase offset for vertical oscillation.")
    parser.add_argument("--split", default="train", help="Dataset split for trajectory generation.")
    parser.add_argument("--image-dir", default="images", help="COLMAP image directory.")
    parser.add_argument("--resolution", type=int, default=1, help="Downscale factor or width.")
    parser.add_argument("--eval-every", type=int, default=8, help="Holdout stride.")
    parser.add_argument("--topology", choices=("indexed", "mesh", "soup"), default="indexed")
    parser.add_argument("--indexed-mesh-method-aware", action="store_true")
    parser.add_argument("--write-frames", action="store_true", help="Keep rendered PNG frames alongside MP4.")
    parser.add_argument("--background-color", choices=("black", "white"), default=None)

    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument("--method-aware", "--method-specific", dest="method_aware", action="store_true", default=True, help="Use method-aware procedural Unity renderer (default).")
    cond_group.add_argument("--general-purpose", "--standard-mesh", dest="general_purpose", action="store_true", help="Use standard Unity Mesh baseline.")

    args = parser.parse_args()
    condition = "general-purpose" if args.general_purpose else "method-aware"

    unity_bin = args.unity or find_unity_executable()
    if unity_bin is None or not Path(unity_bin).is_file():
        raise FileNotFoundError(
            "Unity Editor executable not found. Pass --unity /path/to/Unity or set $UNITY."
        )

    unity_project = args.unity_project or find_unity_project()
    if unity_project is None or not Path(unity_project).is_dir():
        raise FileNotFoundError(
            "Unity project directory not found. Pass --unity-project /path/to/unity or set $PROJECT."
        )

    # Config-driven path
    if args.config is not None:
        from msbench.cli.config import (
            adapter_config,
            dataset_config,
            load_cli_config,
            merged_section,
            output_dir as config_output_dir,
        )

        cfg = load_cli_config(args.config)
        assert cfg is not None
        video_cfg = merged_section(cfg, "render", nested="video")
        adapter_cfg = adapter_config(cfg, method=args.method)
        method = str(args.method or adapter_cfg.get("type") or "triangle-splatting")
        dataset_cfg = dataset_config(cfg, dataset=str(args.dataset) if args.dataset else None, stage="render")

        dataset_path = Path(dataset_cfg["root"])
        if "scene" in dataset_cfg and (dataset_path / dataset_cfg["scene"]).is_dir():
            dataset_path = dataset_path / dataset_cfg["scene"]

        run_dir = config_output_dir(cfg)
        cond_folder = "unity_method_aware" if condition == "method-aware" else "unity_general_purpose"
        default_out = Path(run_dir) / cond_folder / "video" if run_dir else Path(f"video_{cond_folder}")
        output_dir = Path(args.output_dir or default_out)

        triasset_path = args.triasset or (Path(run_dir) / "unity_native" / f"{method}.triasset")
        if not triasset_path.is_dir():
            from msbench.unity_assets import export_triasset
            ckpt = adapter_cfg.get("checkpoint")
            if ckpt:
                print(f"[MeshSplatBench] Exporting triasset for {method} to {triasset_path}...")
                export_triasset(method=method, checkpoint=ckpt, output_dir=triasset_path)

        run_unity_video_for_scene(
            unity_bin=unity_bin,
            unity_project=unity_project,
            method=method,
            triasset_path=triasset_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            condition=condition,
            topology="indexed" if args.topology == "mesh" else args.topology,
            indexed_mesh_method_aware=args.indexed_mesh_method_aware,
            trajectory_path=args.trajectory,
            dataset_type=str(dataset_cfg.get("type", "auto")),
            image_dir=str(dataset_cfg.get("image_dir", args.image_dir)),
            resolution=int(dataset_cfg.get("resolution", args.resolution)),
            eval_every=int(dataset_cfg.get("eval_every", args.eval_every)),
            frames=int(video_cfg.get("frames", args.frames)),
            fps=int(video_cfg.get("fps", args.fps)),
            zoom=float(video_cfg.get("zoom", args.zoom)),
            z_variation=float(video_cfg.get("z_variation", args.z_variation)),
            z_phase=float(video_cfg.get("z_phase", args.z_phase)),
            split=str(video_cfg.get("split", args.split)),
            write_frames=bool(args.write_frames),
            background_color=args.background_color,
        )
        return 0

    # Direct arguments path
    if args.method is None:
        raise ValueError("Must provide --config or --method.")
    method = args.method
    scenes = args.scenes or ([args.dataset.name] if args.dataset else list(SCENES))

    for scene in scenes:
        if args.dataset and Path(args.dataset).is_dir():
            dataset_path = Path(args.dataset)
        elif args.datasets_root:
            dataset_path = Path(args.datasets_root) / scene
        else:
            dataset_path = None

        cond_folder = "unity_method_aware" if condition == "method-aware" else "unity_general_purpose"
        if args.outputs_root:
            scene_out = Path(args.outputs_root) / scene
            triasset = args.triasset or (scene_out / "unity_native" / f"{method}.triasset")
            out_dir = Path(args.output_dir or (scene_out / cond_folder / "video"))
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            triasset = args.triasset or Path(f"outputs/{method}/{scene}/unity_native/{method}.triasset")
        else:
            scene_out = Path(f"outputs/{method}/{scene}")
            triasset = args.triasset or (scene_out / "unity_native" / f"{method}.triasset")
            out_dir = scene_out / cond_folder / "video"

        ref_img_dir = args.image_dir
        if ref_img_dir == "images" and dataset_path and not (dataset_path / "images").is_dir():
            ref_img_dir = reference_image_directory(scene)

        run_unity_video_for_scene(
            unity_bin=unity_bin,
            unity_project=unity_project,
            method=method,
            triasset_path=triasset,
            dataset_path=dataset_path,
            output_dir=out_dir,
            condition=condition,
            topology="indexed" if args.topology == "mesh" else args.topology,
            indexed_mesh_method_aware=args.indexed_mesh_method_aware,
            trajectory_path=args.trajectory,
            image_dir=ref_img_dir,
            resolution=args.resolution,
            eval_every=args.eval_every,
            frames=args.frames,
            fps=args.fps,
            zoom=args.zoom,
            z_variation=args.z_variation,
            z_phase=args.z_phase,
            split=args.split,
            write_frames=args.write_frames,
            background_color=args.background_color,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
