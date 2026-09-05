"""Mesh export and Chamfer/DTU-style evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msbench.primitives.mesh_triangle import IndexedMeshTriangle
from msbench.primitives.triangle import IndependentTriangle


def primitive_to_mesh_arrays(primitive: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a MeshSplatBench primitive to numpy vertices/faces."""
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


def export_adapter_mesh(adapter: Any, path: str | Path, **kwargs: Any) -> Path:
    """Export an adapter's primitive representation to PLY."""
    if kwargs and hasattr(adapter, "export_mesh"):
        return adapter.export_mesh(path, **kwargs)
    vertices, faces = primitive_to_mesh_arrays(adapter.to_primitive())
    return export_ply(vertices, faces, path)


def _load_mesh_object(path: str | Path):
    path = Path(path)
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("Mesh/point-cloud evaluation requires trimesh.") from exc

    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    return obj


def _sample_points(obj: Any, *, num_samples: int = 500_000) -> np.ndarray:
    if hasattr(obj, "faces") and getattr(obj, "faces", None) is not None and len(obj.faces) > 0:
        count = min(num_samples, max(num_samples // 2, len(obj.faces) * 8))
        pts = obj.sample(count)
    else:
        pts = np.asarray(obj.vertices)
    return np.asarray(pts, dtype=np.float32)


def _resolve_dtu_geometry_mode(pred_path: str | Path, geometry_mode: str = "auto") -> str:
    mode = str(geometry_mode or "auto").lower()
    if mode not in {"auto", "mesh", "pcd"}:
        raise ValueError(f"DTU geometry_mode must be 'auto', 'mesh', or 'pcd', got {geometry_mode!r}.")
    if mode != "auto":
        return mode
    stem = Path(pred_path).stem.lower()
    return "pcd" if stem.endswith("_pcd") or "_pcd_" in stem else "mesh"


def _load_points(path: str | Path, *, num_samples: int = 500_000) -> np.ndarray:
    return _sample_points(_load_mesh_object(path), num_samples=num_samples)


def _chamfer_points(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Chamfer evaluation requires scipy.") from exc

    if len(pred) == 0:
        raise ValueError("Predicted mesh/point cloud produced no evaluation points.")
    if len(gt) == 0:
        raise ValueError("Ground-truth point cloud produced no evaluation points.")
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


def chamfer_distance(
    pred_path: str | Path,
    gt_path: str | Path,
    *,
    num_samples: int = 500_000,
) -> dict[str, float]:
    """Compute symmetric Chamfer distance between mesh/point cloud files."""
    pred = _load_points(pred_path, num_samples=num_samples)
    gt = _load_points(gt_path, num_samples=num_samples)
    return _chamfer_points(pred, gt)


def dtu_mesh_metrics(
    pred_path: str | Path,
    dtu_root: str | Path,
    scan_id: str | int,
    *,
    scene_root: str | Path | None = None,
    num_samples: int = 500_000,
    downsample_density: float = 0.2,
    patch_size: float = 60.0,
    max_dist: float = 20.0,
    cull_masks: bool = True,
    geometry_mode: str = "auto",
) -> dict[str, Any]:
    """Evaluate a DTU mesh in the official DTU coordinate frame.

    The triangle-splatting DTU scripts first cull the normalized mesh with scene
    masks, then apply the first DTU ``scale_mat`` before comparing with official
    STL points. This function mirrors that flow and uses official ``ObsMask`` /
    ``Plane`` files when they are present under ``dtu_root``.
    """
    dtu_root = Path(dtu_root)
    resolved_geometry_mode = _resolve_dtu_geometry_mode(pred_path, geometry_mode)
    gt_path = find_dtu_ground_truth(dtu_root, scan_id)
    scene_dir = _find_dtu_scene_root(dtu_root, scan_id, scene_root)
    warnings: list[str] = []

    obj = _load_mesh_object(pred_path)
    if cull_masks:
        obj, cull_info = _cull_dtu_mesh_with_scene_masks(obj, scene_dir)
        warnings.extend(cull_info.pop("warnings", []))
    else:
        cull_info = {"mask_culled": False}

    scale_mat, scale_path = _load_dtu_scale_matrix(dtu_root, scan_id, scene_dir)
    if scale_mat is not None:
        _apply_dtu_scale(obj, scale_mat)
    else:
        warnings.append("DTU cameras.npz not found; evaluated without DTU scale transform.")

    gt = _load_points(gt_path, num_samples=num_samples)

    obs_mask_path = _dtu_obs_mask_path(dtu_root, scan_id)
    plane_path = _dtu_plane_path(dtu_root, scan_id)
    if obs_mask_path.exists():
        metrics = _official_dtu_metrics(
            obj,
            gt,
            obs_mask_path=obs_mask_path,
            plane_path=plane_path if plane_path.exists() else None,
            downsample_density=downsample_density,
            patch_size=patch_size,
            max_dist=max_dist,
            geometry_mode=resolved_geometry_mode,
        )
        protocol = "dtu_official"
        if not plane_path.exists():
            warnings.append("DTU Plane*.mat not found; completeness used all STL points.")
    else:
        pred = (
            np.asarray(obj.vertices, dtype=np.float32)
            if resolved_geometry_mode == "pcd"
            else _sample_points(obj, num_samples=num_samples)
        )
        metrics = _chamfer_points(pred, gt)
        protocol = "dtu_scaled_chamfer_fallback" if scale_mat is not None else "raw_chamfer_fallback"
        warnings.append(
            "DTU ObsMask file not found; used scaled Chamfer fallback instead of official DTU masking."
        )

    metrics.update(
        {
            "protocol": protocol,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "scene_root": str(scene_dir) if scene_dir is not None else None,
            "scale_mat_path": str(scale_path) if scale_path is not None else None,
            "scale_transform_applied": scale_mat is not None,
            "geometry_mode": resolved_geometry_mode,
            "samples": int(num_samples),
            **cull_info,
        }
    )
    if warnings:
        metrics["warnings"] = warnings
    return metrics


def _scan_number(scan_id: str | int) -> str:
    scan = str(scan_id)
    return scan[4:] if scan.startswith("scan") else scan


def _find_dtu_scene_root(
    dtu_root: Path,
    scan_id: str | int,
    scene_root: str | Path | None,
) -> Path | None:
    if scene_root is not None:
        path = Path(scene_root)
        if path.exists():
            return path
    scan_num = _scan_number(scan_id)
    for candidate in (dtu_root / f"scan{scan_num}", dtu_root / str(scan_id)):
        if candidate.exists():
            return candidate
    return None


def _load_dtu_scale_matrix(
    dtu_root: Path,
    scan_id: str | int,
    scene_root: Path | None,
) -> tuple[np.ndarray | None, Path | None]:
    scan_num = _scan_number(scan_id)
    candidates = []
    if scene_root is not None:
        candidates.append(scene_root / "cameras.npz")
    candidates.extend(
        [dtu_root / f"scan{scan_num}" / "cameras.npz", dtu_root / str(scan_id) / "cameras.npz"]
    )
    for path in candidates:
        if path.exists():
            data = np.load(path)
            key = "scale_mat_0"
            if key in data.files:
                return data[key].astype(np.float32), path
    return None, None


def _apply_dtu_scale(obj: Any, scale_mat: np.ndarray) -> None:
    scale = float(scale_mat[0, 0])
    translate = scale_mat[:3, 3].astype(np.float64)
    obj.vertices = np.asarray(obj.vertices, dtype=np.float64) * scale + translate[None]


def _cull_dtu_mesh_with_scene_masks(obj: Any, scene_root: Path | None) -> tuple[Any, dict[str, Any]]:
    info: dict[str, Any] = {
        "mask_culled": False,
        "vertices_before_cull": int(len(getattr(obj, "vertices", []))),
        "faces_before_cull": int(len(getattr(obj, "faces", []))) if hasattr(obj, "faces") else 0,
        "warnings": [],
    }
    if scene_root is None:
        info["warnings"].append("DTU scene root not found; skipped scene-mask culling.")
        return obj, info
    camera_file = scene_root / "cameras.npz"
    mask_dir = scene_root / "mask"
    image_dir = scene_root / "images"
    if not image_dir.exists():
        image_dir = scene_root / "image"
    if not camera_file.exists() or not mask_dir.exists() or not image_dir.exists():
        info["warnings"].append("DTU cameras/images/mask directory incomplete; skipped scene-mask culling.")
        return obj, info
    has_faces = hasattr(obj, "faces") and getattr(obj, "faces", None) is not None and len(obj.faces) > 0

    try:
        import cv2
        from scipy.ndimage import binary_dilation
    except ImportError as exc:
        raise ImportError("DTU mask culling requires opencv-python and scipy.") from exc

    image_paths = sorted(path for path in image_dir.glob("*.png") if not path.name.startswith("._"))
    if not image_paths:
        info["warnings"].append("No DTU images found; skipped scene-mask culling.")
        return obj, info
    camera_data = np.load(camera_file)
    vertices = np.asarray(obj.vertices, dtype=np.float64)
    vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    keep = np.ones(vertices.shape[0], dtype=bool)
    footprint = _disk_footprint(24)

    for image_idx, image_path in enumerate(image_paths):
        world_key = f"world_mat_{image_idx}"
        if world_key not in camera_data.files:
            continue
        scale_key = f"scale_mat_{image_idx}"
        scale_mat = (
            camera_data[scale_key].astype(np.float64)
            if scale_key in camera_data.files
            else np.eye(4, dtype=np.float64)
        )
        world_mat = camera_data[world_key].astype(np.float64)
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            mask_candidates = sorted(mask_dir.glob(f"{image_idx:03d}.*"))
            mask_path = mask_candidates[0] if mask_candidates else mask_path
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = binary_dilation(mask > 0, structure=footprint)
        projection = (world_mat @ scale_mat)[:3, :4]
        decomposed = cv2.decomposeProjectionMatrix(projection.astype(np.float64))
        K, R, t = decomposed[0], decomposed[1], decomposed[2]
        K = K / K[2, 2]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R.T
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        w2c = np.linalg.inv(pose)
        cam = K @ (w2c[:3, :] @ vertices_h.T)
        z = cam[2]
        u = cam[0] / (z + 1e-6)
        v = cam[1] / (z + 1e-6)
        height, width = mask.shape[:2]
        valid = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        visible_or_outside = ~valid
        ui = np.clip(np.rint(u[valid]).astype(np.int64), 0, width - 1)
        vi = np.clip(np.rint(v[valid]).astype(np.int64), 0, height - 1)
        visible_or_outside[valid] = mask[vi, ui]
        keep &= visible_or_outside

    if keep.all():
        info["vertices_after_cull"] = info["vertices_before_cull"]
        info["faces_after_cull"] = info["faces_before_cull"]
        return obj, info
    if has_faces:
        face_keep = keep[np.asarray(obj.faces)].all(axis=1)
        obj.update_faces(face_keep)
        obj.update_vertices(keep)
        if len(obj.faces) == 0:
            raise ValueError("DTU scene-mask culling removed all predicted mesh faces.")
    else:
        obj.vertices = np.asarray(obj.vertices)[keep]
        if len(obj.vertices) == 0:
            raise ValueError("DTU scene-mask culling removed all predicted point-cloud vertices.")
    info.update(
        {
            "mask_culled": True,
            "vertices_after_cull": int(len(obj.vertices)),
            "faces_after_cull": int(len(obj.faces)) if has_faces else 0,
        }
    )
    return obj, info


def _disk_footprint(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _dtu_obs_mask_path(dtu_root: Path, scan_id: str | int) -> Path:
    scan_num = int(_scan_number(scan_id))
    return dtu_root / "ObsMask" / f"ObsMask{scan_num}_10.mat"


def _dtu_plane_path(dtu_root: Path, scan_id: str | int) -> Path:
    scan_num = int(_scan_number(scan_id))
    return dtu_root / "ObsMask" / f"Plane{scan_num}.mat"


def _official_dtu_metrics(
    pred_obj: Any,
    gt: np.ndarray,
    *,
    obs_mask_path: Path,
    plane_path: Path | None,
    downsample_density: float,
    patch_size: float,
    max_dist: float,
    geometry_mode: str,
) -> dict[str, float]:
    try:
        from scipy.io import loadmat
        import sklearn.neighbors as skln
    except ImportError as exc:
        raise ImportError("Official DTU mesh evaluation requires scipy and scikit-learn.") from exc

    pred = _official_dtu_sample_geometry(pred_obj, downsample_density, geometry_mode=geometry_mode)
    rng = np.random.default_rng()
    rng.shuffle(pred, axis=0)
    pred_down = _official_radius_downsample(pred, downsample_density)
    obs = loadmat(obs_mask_path)
    obs_mask, bb, resolution = [obs[key] for key in ("ObsMask", "BB", "Res")]
    bb = bb.astype(np.float32)
    inbound = ((pred_down >= bb[:1] - patch_size) & (pred_down < bb[1:] + patch_size * 2)).all(axis=1)
    pred_in = pred_down[inbound]
    grid = np.rint((pred_in - bb[:1]) / resolution).astype(np.int32)
    grid_inbound = ((grid >= 0) & (grid < np.array(obs_mask.shape)[None])).all(axis=1)
    grid_valid = grid[grid_inbound]
    in_obs = obs_mask[grid_valid[:, 0], grid_valid[:, 1], grid_valid[:, 2]].astype(bool)
    pred_in_obs = pred_in[grid_inbound][in_obs]
    if len(pred_in) == 0:
        raise ValueError("DTU ObsMask filtering removed all predicted points.")
    if len(pred_in_obs) == 0:
        raise ValueError("DTU observation mask removed all predicted points.")

    gt_eval = gt
    if plane_path is not None:
        plane = loadmat(plane_path)["P"].reshape((1, 4))
        gt_h = np.concatenate([gt, np.ones_like(gt[:, :1])], axis=1)
        gt_eval = gt[(plane * gt_h).sum(axis=1) > 0]

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=downsample_density, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(gt)
    d2s, _ = nn_engine.kneighbors(pred_in_obs, n_neighbors=1, return_distance=True)
    nn_engine.fit(pred_in)
    s2d, _ = nn_engine.kneighbors(gt_eval, n_neighbors=1, return_distance=True)
    d2s = d2s[d2s < max_dist]
    s2d = s2d[s2d < max_dist]
    accuracy = float(np.mean(d2s)) if len(d2s) else float("nan")
    completeness = float(np.mean(s2d)) if len(s2d) else float("nan")
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "overall": float((accuracy + completeness) * 0.5),
        "mean_d2s": accuracy,
        "mean_s2d": completeness,
        "pred_points": int(pred_in_obs.shape[0]),
        "pred_points_sampled": int(pred.shape[0]),
        "pred_points_downsampled": int(pred_down.shape[0]),
        "gt_points": int(gt_eval.shape[0]),
    }


def _official_dtu_sample_geometry(
    obj: Any,
    downsample_density: float,
    *,
    geometry_mode: str = "mesh",
) -> np.ndarray:
    vertices = np.asarray(getattr(obj, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(obj, "faces", []), dtype=np.int64) if hasattr(obj, "faces") else np.empty((0, 3), dtype=np.int64)
    if len(vertices) == 0:
        return vertices.reshape(0, 3)
    if geometry_mode == "pcd":
        return vertices
    if geometry_mode != "mesh":
        raise ValueError(f"DTU geometry_mode must be 'mesh' or 'pcd', got {geometry_mode!r}.")
    if len(faces) == 0:
        return vertices

    tri_vert = vertices[faces]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    valid = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[valid] for arr in (l1, l2, area2, v1, v2, tri_vert)
    ]
    if len(tri_vert) == 0:
        return vertices

    thr = downsample_density * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr).astype(np.int64)[:, 0]
    n2 = np.floor(l2 / thr).astype(np.int64)[:, 0]
    sampled = []
    for i in range(len(tri_vert)):
        grid = np.mgrid[: n1[i] + 1, : n2[i] + 1].astype(np.float64)
        grid += 0.5
        grid[0] /= max(n1[i], 1e-7)
        grid[1] /= max(n2[i], 1e-7)
        bary = np.transpose(grid, (1, 2, 0))
        bary = bary[bary.sum(axis=-1) < 1]
        if len(bary):
            sampled.append(v1[i] * bary[:, :1] + v2[i] * bary[:, 1:] + tri_vert[i, 0])
    if sampled:
        return np.concatenate([vertices, *sampled], axis=0)
    return vertices


def _official_radius_downsample(points: np.ndarray, radius: float) -> np.ndarray:
    import sklearn.neighbors as skln

    if len(points) == 0:
        return points
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=radius, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(points)
    neighbors = nn_engine.radius_neighbors(points, radius=radius, return_distance=False)
    keep = np.ones(points.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(neighbors):
        if keep[curr]:
            keep[idxs] = False
            keep[curr] = True
    return points[keep]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def write_mesh_metrics(metrics: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    return output


def find_dtu_ground_truth(dtu_root: str | Path, scan_id: str | int) -> Path:
    """Find a DTU official point cloud for a scan id."""
    root = Path(dtu_root)
    scan_num = _scan_number(scan_id)
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
