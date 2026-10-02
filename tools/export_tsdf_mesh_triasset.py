#!/usr/bin/env python3
"""Export a colored TSDF mesh as a Unity general-purpose TriAsset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


SH_C0 = 0.28209479177387814


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_buffer(root: Path, name: str, values: np.ndarray) -> dict:
    values = np.ascontiguousarray(values)
    relative = Path("buffers") / f"{name}.bin"
    values.tofile(root / relative)
    if values.dtype == np.float32:
        dtype = "float32"
    elif values.dtype == np.int32:
        dtype = "int32"
    else:
        raise ValueError(f"unsupported buffer dtype: {values.dtype}")
    return {
        "file": relative.as_posix(),
        "dtype": dtype,
        "shape": [int(value) for value in values.shape],
        "semantic": {
            "positions": "indexed mesh positions",
            "indices": "triangle indices",
            "sh_dc": "per-vertex SH DC reconstructed from TSDF vertex colors",
        }[name],
    }


def export(mesh_path: Path, output: Path, method: str, background: str, force: bool) -> dict:
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "buffers").mkdir()

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"mesh has no usable vertices: {mesh_path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError(f"mesh has no usable triangles: {mesh_path}")
    if colors.shape != vertices.shape:
        raise ValueError(
            f"TSDF mesh must contain per-vertex colors; got {colors.shape}, expected {vertices.shape}"
        )

    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    # Unity's general-purpose renderer evaluates color = 0.28209479 * dc + 0.5.
    sh_dc = ((colors - 0.5) / SH_C0).astype(np.float32)

    buffers = {
        "positions": write_buffer(output, "positions", vertices),
        "indices": write_buffer(output, "indices", triangles),
        "sh_dc": write_buffer(output, "sh_dc", sh_dc),
    }
    buffer_list = [{"name": name, **spec} for name, spec in buffers.items()]
    manifest = {
        "schema_version": "1.0",
        "export_contract_revision": 2,
        "asset_kind": "msbench-unity-native",
        "method": method,
        "source": {
            "mesh": str(mesh_path.resolve()),
            "mesh_sha256": sha256(mesh_path),
        },
        "coordinate_system": {
            "name": "msbench_world",
            "up_axis": "+y",
            "handedness": "source-defined",
            "unit_scale": 1.0,
        },
        "rendering": {
            "color_space": "native_image_code_values",
            "requires_method_renderer": False,
            "generic_mesh_equivalent": True,
            "unity_profile_name": "method-aware",
            "cuda_equivalent": False,
            "background_color": background,
            "renderer": "general-purpose-mesh",
            "primitive_topology": "indexed-triangles",
            "primitive_count": int(len(triangles)),
            "general_purpose": {
                "supported": True,
                "appearance": "vertex_color",
                "view_dependent_appearance": False,
                "opaque_depth_tested": True,
            },
            "unity_method_aware_requires_per_camera_depth_sort": False,
            "unity_method_aware_status": "opaque-tsdf-mesh",
        },
        "buffers": buffers,
        "buffer_list": buffer_list,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "ok",
        "asset": str(output),
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "bytes": sum((output / spec["file"]).stat().st_size for spec in buffers.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="2dgs-tsdf-mesh")
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.mesh, args.output, args.method, args.background, args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
