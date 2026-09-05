#!/usr/bin/env python3
"""Score Unity outputs with the same evaluator used for Native CUDA renders."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def reference_image_directory(scene: str) -> str:
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("evaluate_deployment_images.py"))
    parser.add_argument("--pythonpath", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/triangle-splatting/mipnerf360"))
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--method")
    parser.add_argument("--condition", choices=("method-aware", "general-purpose"), required=True)
    parser.add_argument("--native-render-subdir", default="renders/test/renders")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.pythonpath)
    if args.torch_home is not None:
        env["TORCH_HOME"] = str(args.torch_home)
    try:
        from tools.validate_unity_triasset_cpu import validate_triasset
    except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
        from validate_unity_triasset_cpu import validate_triasset

    method = args.method or args.outputs_root.resolve().parent.name
    for scene in args.scenes:
        scene_root = args.outputs_root.resolve() / scene
        root = scene_root / args.output_name
        asset = scene_root / "unity_native" / f"{method}.triasset"
        validate_triasset(asset, max_faces=10000)
        import json
        manifest = json.loads((asset / "manifest.json").read_text())
        general = manifest["rendering"].get("general_purpose", {})
        if args.condition == "general-purpose" and not general.get("supported", False):
            raise RuntimeError(
                f"{method}/{scene} has no fair general-purpose appearance contract: "
                f"{general.get('reason', 'unsupported')}"
            )
        image_dir = reference_image_directory(scene)
        command = [
            str(args.python), str(args.evaluator),
            "--prediction", str(root),
            "--ground-truth", str(args.datasets_root.resolve() / scene / image_dir),
            "--native", str(scene_root / args.native_render_subdir),
            "--output", str(root / "metrics_summary.json"),
            "--with-lpips", "--lpips-net", "vgg", "--lpips-device", args.lpips_device,
        ]
        print(f"[MeshSplatBench] metrics scene={scene}, reference={image_dir}", flush=True)
        with (root / "metrics_vgg.log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            print((root / "metrics_vgg.log").read_text()[-4000:], file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
