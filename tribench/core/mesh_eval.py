"""Mesh export and Chamfer-style evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tribench.primitives.mesh_triangle import IndexedMeshTriangle
from tribench.primitives.triangle import IndependentTriangle


def primitive_to_mesh_arrays(primitive: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a TriBench primitive to numpy vertices/faces."""
    if isinstance(primitive, IndexedMeshTriangle):
        vertices = primitive.vertices.detach().cpu().numpy().astype(np.float32)
        faces = primitive.faces.detach().cpu().numpy().astype(np.int64)
        return vertices, faces
    if isinstance(primitive, IndependentTriangle):
        vertices = primitive.vertices.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
        return vertices, faces
    if hasattr(primitive, "to_tensor"):
        tri = primitive.to_tensor().detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        faces = np.arange(tri.shape[0], dtype=np.int64).reshape(-1, 3)
        return tri, faces
    raise TypeError(f"Cannot convert primitive {type(primitive)!r} to mesh arrays")


def export_ply(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> Path:
    """Export a triangle mesh as ASCII PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return path


def export_adapter_mesh(adapter: Any, path: str | Path) -> Path:
    """Export an adapter's primitive representation to PLY."""
    vertices, faces = primitive_to_mesh_arrays(adapter.to_primitive())
    return export_ply(vertices, faces, path)


def _load_points(path: str | Path, *, num_samples: int = 500_000) -> np.ndarray:
    path = Path(path)
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("Mesh/point-cloud evaluation requires trimesh.") from exc

    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if hasattr(obj, "faces") and getattr(obj, "faces", None) is not None and len(obj.faces) > 0:
        count = min(num_samples, max(num_samples // 2, len(obj.faces) * 8))
        pts = obj.sample(count)
    else:
        pts = np.asarray(obj.vertices)
    return np.asarray(pts, dtype=np.float32)


def chamfer_distance(
    pred_path: str | Path,
    gt_path: str | Path,
    *,
    num_samples: int = 500_000,
) -> dict[str, float]:
    """Compute symmetric Chamfer distance between mesh/point cloud files."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Chamfer evaluation requires scipy.") from exc

    pred = _load_points(pred_path, num_samples=num_samples)
    gt = _load_points(gt_path, num_samples=num_samples)
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    gt_to_pred, _ = pred_tree.query(gt, k=1)
    pred_to_gt, _ = gt_tree.query(pred, k=1)
    accuracy = float(np.mean(pred_to_gt))
    completeness = float(np.mean(gt_to_pred))
    overall = float((accuracy + completeness) * 0.5)
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "overall": overall,
        "pred_points": int(pred.shape[0]),
        "gt_points": int(gt.shape[0]),
    }


def write_mesh_metrics(metrics: dict[str, float], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    return output


def find_dtu_ground_truth(dtu_root: str | Path, scan_id: str | int) -> Path:
    """Find a DTU official point cloud for a scan id."""
    root = Path(dtu_root)
    scan = str(scan_id)
    if scan.startswith("scan"):
        scan_num = scan[4:]
    else:
        scan_num = scan
    candidates = [
        root / f"Points/stl/stl{scan_num:0>3}_total.ply",
        root / f"Points/stl/stl{scan_num}_total.ply",
        root / f"scan{scan_num}" / "stl" / f"stl{scan_num}_total.ply",
        root / f"scan{scan_num}.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find DTU ground-truth point cloud for scan {scan_id} under {root}")

