"""Dual point cloud geometry export for nerfbaselines-compatible viewer.

Exports two standard PLY point clouds from any TriBench primitive:
  * geometry_viewer_points.ply — lightweight (300K–1M pts) for browser viewer
  * geometry_cg_points.ply      — high-fidelity (up to 3M pts) for CG tools

Plus optional sidecar files:
  * geometry_metadata.json   — method, checkpoint, schema, coordinate info
  * geometry_attributes.npz  — opacity, primitive_id, area, SH/DC features
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tribench.primitives.base import BasePrimitive


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _primitive_triangle_vertices(primitive: BasePrimitive) -> torch.Tensor:
    """Return triangle-soup vertices [F, 3, 3] for any triangle primitive."""
    vertices = getattr(primitive, "vertices", None)
    faces = getattr(primitive, "faces", None)

    if isinstance(vertices, torch.Tensor):
        if vertices.dim() == 3 and vertices.shape[1:] == (3, 3):
            return vertices
        if (
            vertices.dim() == 2
            and vertices.shape[1] == 3
            and isinstance(faces, torch.Tensor)
        ):
            return vertices[faces.to(device=vertices.device, dtype=torch.long)]

    verts = primitive.to_tensor()
    if verts.dim() == 2 and verts.shape[1] == 3 and isinstance(faces, torch.Tensor):
        verts = verts[faces.to(device=verts.device, dtype=torch.long)]
    return verts

def _sample_triangle_surface(
    vertices: torch.Tensor,   # [N, 3, 3]
    num_points: int,
    *,
    areas: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-weighted surface sampling of triangle soup.

    Returns:
        points: [num_points, 3]  sampled positions
        tri_ids: [num_points]    index of the source triangle for each point
    """
    N = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype
    if areas is None:
        v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
        areas = 0.5 * torch.norm(cross, dim=-1)
    else:
        areas = areas.to(device=device, dtype=dtype)
    areas = areas.clamp(min=1e-12)

    generator_device = getattr(generator, "device", None)
    use_generator = generator is not None and (
        generator_device is None or torch.device(generator_device) == device
    )

    # Area-weighted triangle selection
    probs = areas / areas.sum()
    tri_ids = torch.multinomial(
        probs,
        num_points,
        replacement=True,
        generator=generator if use_generator else None,
    )

    # Barycentric coordinates via sqrt-mapping for uniform area distribution
    rand_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    if use_generator:
        rand_kwargs["generator"] = generator
    r1 = torch.rand(num_points, **rand_kwargs)
    r2 = torch.rand(num_points, **rand_kwargs)
    sqrt_r1 = torch.sqrt(r1)
    a = 1.0 - sqrt_r1
    b = sqrt_r1 * (1.0 - r2)
    c = sqrt_r1 * r2

    sampled = vertices[tri_ids]  # [num_points, 3, 3]
    points = a.unsqueeze(-1) * sampled[:, 0] \
           + b.unsqueeze(-1) * sampled[:, 1] \
           + c.unsqueeze(-1) * sampled[:, 2]
    return points, tri_ids


def _colors_from_dc(sh_coeffs: torch.Tensor | None, num_prims: int) -> torch.Tensor | None:
    """Extract DC (SH degree-0) component as RGB, shape [N, 3] in [0, 1]."""
    if sh_coeffs is None:
        return None
    # SH DC is the first 3 coefficients (constant term)
    if sh_coeffs.shape[-1] >= 3:
        dc = sh_coeffs[:, :3].clamp(0, 1)
        return dc
    return None


def _opacity_colormap(opacity: torch.Tensor) -> torch.Tensor:
    """Map opacity [0, 1] to a coolwarm-style RGB, shape [N, 3] in [0, 1]."""
    t = opacity.clamp(0, 1)
    # Simple blue-to-red colormap
    r = (2 * t).clamp(0, 1)
    b = (2 * (1 - t)).clamp(0, 1)
    g = (1 - 2 * torch.abs(t - 0.5)).clamp(0, 1) * 0.5
    return torch.stack([r, g, b], dim=-1).float()


def _voxel_downsample(points: torch.Tensor, colors: torch.Tensor | None,
                      normals: torch.Tensor | None,
                      voxel_size: float) -> tuple[torch.Tensor, ...]:
    """Voxel-grid downsampling: keep one point per voxel cell."""
    coords = torch.floor(points / voxel_size).int()
    # Use first-occurrence per voxel
    keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
    _, idx = np.unique(keys.cpu().numpy(), return_index=True)
    idx = np.sort(idx)
    idx = torch.as_tensor(idx, dtype=torch.long, device=points.device)
    pts = points[idx]
    clr = colors[idx] if colors is not None else None
    nrm = normals[idx] if normals is not None else None
    return pts, clr, nrm


# ---------------------------------------------------------------------------
# PLY writers
# ---------------------------------------------------------------------------

def write_point_cloud_ply(
    path: str | Path,
    xyz: np.ndarray,           # [N, 3] float32
    rgb: np.ndarray | None,    # [N, 3] uint8
    normals: np.ndarray | None = None,  # [N, 3] float32
) -> None:
    """Write a standard vertex-only PLY (x y z [red green blue] [nx ny nz])."""
    from plyfile import PlyData, PlyElement

    N = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if normals is not None:
        dtype += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]

    vertex = np.zeros(N, dtype=dtype)
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    if rgb is not None:
        vertex["red"] = rgb[:, 0]
        vertex["green"] = rgb[:, 1]
        vertex["blue"] = rgb[:, 2]
    if normals is not None:
        vertex["nx"] = normals[:, 0]
        vertex["ny"] = normals[:, 1]
        vertex["nz"] = normals[:, 2]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=True).write(str(path))


def write_mesh_ply(
    path: str | Path,
    vertices: np.ndarray,       # [V, 3] float32
    faces: np.ndarray,          # [F, 3] int32
    rgb: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> None:
    """Write a standard mesh PLY with vertex + face elements."""
    from plyfile import PlyData, PlyElement

    V = vertices.shape[0]
    vtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        vtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if normals is not None:
        vtype += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    vertex = np.zeros(V, dtype=vtype)
    vertex["x"] = vertices[:, 0]
    vertex["y"] = vertices[:, 1]
    vertex["z"] = vertices[:, 2]
    if rgb is not None:
        vertex["red"] = rgb[:, 0]
        vertex["green"] = rgb[:, 1]
        vertex["blue"] = rgb[:, 2]
    if normals is not None:
        vertex["nx"] = normals[:, 0]
        vertex["ny"] = normals[:, 1]
        vertex["nz"] = normals[:, 2]

    F = faces.shape[0]
    face = np.zeros(F, dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = faces

    els = [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")]
    PlyData(els, text=True).write(str(path))


def write_mesh_glb(
    path: str | Path,
    vertices: np.ndarray,       # [V, 3] float32
    faces: np.ndarray,          # [F, 3] int32
    rgb: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> None:
    """Write a mesh as a GLB file (glTF binary).

    GLB is preferred over PLY for Unity/Blender mesh import because it
    supports indexed geometry, vertex colors, and standard coordinate
    conventions. The glTF spec uses +Y up, right-handed, meters —
    TriBench world is also +Y up, so no axis flip is needed.
    """
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_colors=rgb if rgb is not None else None,
        vertex_normals=normals if normals is not None else None,
        process=False,
    )
    # Ensure glTF metadata is set
    mesh.metadata["glTF"] = {"version": "2.0"}
    glb_data = mesh.export(file_type="glb")
    Path(path).write_bytes(glb_data)


def _extract_indexed_mesh_from_primitive(
    primitive: BasePrimitive,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Try to extract a shared-vertex indexed mesh from a primitive.

    For mesh-splatting primitives that expose ``V`` (vertices [V, 3]) and
    ``F`` (faces [F, 3]) attributes, this produces a proper indexed mesh
    with shared vertices instead of duplicated triangle-soup vertices.

    Falls back to ``_extract_mesh_from_primitive`` (triangle soup) for
    primitives that don't have indexed mesh data.
    """
    # Check for indexed mesh data. Older adapters may expose V/F while
    # TriBench primitives use vertices/faces.
    V = getattr(primitive, "V", None)
    F = getattr(primitive, "F", None)
    if V is None or F is None:
        maybe_vertices = getattr(primitive, "vertices", None)
        maybe_faces = getattr(primitive, "faces", None)
        if (
            isinstance(maybe_vertices, torch.Tensor)
            and maybe_vertices.dim() == 2
            and maybe_vertices.shape[1] == 3
            and isinstance(maybe_faces, torch.Tensor)
        ):
            V = maybe_vertices
            F = maybe_faces

    if V is not None and F is not None:
        verts = V.detach().cpu().numpy().astype(np.float32)
        faces_np = F.detach().cpu().numpy().astype(np.int32)

        # Handle quad faces by splitting into triangles
        if faces_np.shape[1] == 4:
            tris = np.empty((faces_np.shape[0] * 2, 3), dtype=np.int32)
            tris[0::2] = faces_np[:, [0, 1, 2]]
            tris[1::2] = faces_np[:, [0, 2, 3]]
            faces_np = tris
        elif faces_np.shape[1] != 3:
            return None  # Unknown face format

        # Colors: try vertex colors first, then per-face colors
        colors = primitive.get_colors()
        rgb = None
        if colors is not None:
            # If colors are per-vertex [V, 3]
            if colors.shape[0] == verts.shape[0]:
                rgb = (colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            # If per-primitive, we can't map to shared vertices cleanly
            # Fall back to no colors

        # Normals: compute from mesh if not provided
        normals = None
        if hasattr(primitive, "vertex_normals") and primitive.vertex_normals is not None:
            normals = primitive.vertex_normals.detach().cpu().numpy().astype(np.float32)

        return verts, faces_np, rgb, normals

    # Fallback: triangle-soup extraction
    return _extract_mesh_from_primitive(primitive)


# ---------------------------------------------------------------------------
# Primitive → geometry
# ---------------------------------------------------------------------------

@dataclass
class GeometryExportConfig:
    viewer_num_points: int = 500_000
    cg_num_points: int = 3_000_000
    color_mode: str = "dc"        # dc | opacity | white | rendered
    normal_mode: str = "none"     # none | primitive | estimated
    export_mesh: bool = False
    export_glb: bool = False
    voxel_downsample: bool = True
    voxel_size: float = 0.01


@dataclass
class GeometryExportResult:
    viewer_ply: str
    cg_ply: str
    mesh_ply: str | None = None
    mesh_glb_path: str | None = None
    metadata_path: str | None = None
    attributes_path: str | None = None
    num_viewer_points: int = 0
    num_cg_points: int = 0
    has_mesh: bool = False
    has_glb: bool = False


@dataclass
class ViewerPointCloudExportConfig:
    viewer_num_points: int = 500_000
    color_mode: str = "dc"
    normal_mode: str = "none"
    voxel_downsample: bool = True
    voxel_size: float = 0.01


@dataclass
class ViewerPointCloudExportResult:
    viewer_ply: str
    metadata_path: str | None = None
    num_viewer_points: int = 0


@dataclass
class OriginalGeometryExportResult:
    geometry_ply: str
    mesh_glb_path: str | None = None
    metadata_path: str | None = None
    attributes_path: str | None = None
    num_vertices: int = 0
    num_faces: int = 0
    has_glb: bool = False


def primitive_to_point_cloud(
    primitive: BasePrimitive,
    num_points: int,
    *,
    color_mode: str = "dc",
    normal_mode: str = "none",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Sample a point cloud from a primitive.

    Returns:
        xyz: [num_points, 3] float32
        rgb: [num_points, 3] float32 in [0,1] or None
        normals: [num_points, 3] float32 or None
        tri_ids: [num_points] int64  (source triangle index per point)
    """
    # Get triangle vertices [N, 3, 3]. Indexed meshes are expanded through
    # their face buffer for surface sampling, while mesh export keeps indices.
    verts = _primitive_triangle_vertices(primitive)
    assert verts.dim() == 3 and verts.shape[1:] == (3, 3), \
        f"Expected [N, 3, 3] vertices, got {verts.shape}"

    areas = primitive.compute_areas()
    areas = areas.to(device=verts.device, dtype=verts.dtype)
    xyz, tri_ids = _sample_triangle_surface(
        verts, num_points, areas=areas, generator=generator,
    )

    # --- Colors ---
    rgb = None
    if color_mode == "dc":
        colors = primitive.get_colors()
        if colors is not None:
            # Per-triangle color → per-point via tri_ids
            colors = colors.to(device=verts.device)
            rgb = colors[tri_ids].float().clamp(0, 1)
    elif color_mode == "opacity":
        opacity = primitive.get_opacity()
        if opacity is not None:
            opacity = opacity.to(device=verts.device)
            rgb = _opacity_colormap(opacity[tri_ids])
    elif color_mode == "white":
        rgb = torch.ones(num_points, 3, device=verts.device, dtype=verts.dtype)
    # "rendered" mode is handled separately (would need adapter.render)

    # --- Normals ---
    normals = None
    if normal_mode == "primitive":
        face_normals = primitive.compute_normals()  # [N, 3]
        face_normals = face_normals.to(device=verts.device)
        normals = face_normals[tri_ids]

    return xyz, rgb, normals, tri_ids


def export_viewer_geometry(
    primitive: BasePrimitive,
    output_dir: str | Path,
    *,
    method_name: str = "unknown",
    checkpoint_path: str | None = None,
    config: GeometryExportConfig | None = None,
    generator: torch.Generator | None = None,
) -> GeometryExportResult:
    """Export dual point clouds + sidecar files from a primitive.

    Produces:
      <output_dir>/geometry_viewer_points.ply
      <output_dir>/geometry_cg_points.ply
      <output_dir>/geometry_metadata.json
      <output_dir>/geometry_attributes.npz
      <output_dir>/geometry_mesh.ply  (if export_mesh and primitive supports faces)
    """
    cfg = config or GeometryExportConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    viewer_path = out / "geometry_viewer_points.ply"
    cg_path = out / "geometry_cg_points.ply"
    meta_path = out / "geometry_metadata.json"
    attr_path = out / "geometry_attributes.npz"
    mesh_path = out / "geometry_mesh.ply" if cfg.export_mesh else None

    # --- Sample viewer point cloud ---
    v_xyz, v_rgb, v_nrm, v_tri = primitive_to_point_cloud(
        primitive, cfg.viewer_num_points,
        color_mode=cfg.color_mode, normal_mode=cfg.normal_mode,
        generator=generator,
    )

    # Voxel downsample viewer cloud for browser performance
    if cfg.voxel_downsample and v_xyz.shape[0] > 0:
        v_xyz, v_rgb, v_nrm = _voxel_downsample(
            v_xyz, v_rgb, v_nrm, cfg.voxel_size,
        )
        # Rebuild tri_ids not needed after downsample for viewer cloud

    # --- Sample CG point cloud ---
    c_xyz, c_rgb, c_nrm, c_tri = primitive_to_point_cloud(
        primitive, cfg.cg_num_points,
        color_mode=cfg.color_mode, normal_mode=cfg.normal_mode,
        generator=generator,
    )

    # --- Convert to numpy for PLY ---
    def _to_np_uint8(rgb):
        if rgb is None:
            return None
        return (rgb.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    def _to_np_f32(t):
        if t is None:
            return None
        return t.cpu().numpy().astype(np.float32)

    write_point_cloud_ply(
        viewer_path, _to_np_f32(v_xyz), _to_np_uint8(v_rgb), _to_np_f32(v_nrm),
    )
    write_point_cloud_ply(
        cg_path, _to_np_f32(c_xyz), _to_np_uint8(c_rgb), _to_np_f32(c_nrm),
    )

    # --- Optional mesh ---
    has_mesh = False
    has_glb = False
    glb_path = out / "geometry_mesh.glb" if cfg.export_glb else None
    if cfg.export_mesh:
        try:
            mesh_data = _extract_indexed_mesh_from_primitive(primitive)
            if mesh_data is not None:
                m_verts, m_faces, m_rgb, m_nrm = mesh_data
                write_mesh_ply(mesh_path, m_verts, m_faces, m_rgb, m_nrm)
                has_mesh = True
                # GLB export (best-effort)
                if cfg.export_glb and glb_path is not None:
                    try:
                        write_mesh_glb(glb_path, m_verts, m_faces, m_rgb, m_nrm)
                        has_glb = True
                    except Exception:
                        pass
        except Exception:
            pass  # mesh export is best-effort

    # --- Sidecar: metadata ---
    areas = primitive.compute_areas()
    metadata = {
        "method": method_name,
        "checkpoint": checkpoint_path,
        "schema_version": "1.0",
        "coordinate_system": "tribench_world",
        "up_axis": "+y",
        "scale": 1.0,
        "assets": {
            "viewer_points": "geometry_viewer_points.ply",
            "cg_points": "geometry_cg_points.ply",
            "mesh": "geometry_mesh.ply" if has_mesh else None,
            "mesh_glb": "geometry_mesh.glb" if has_glb else None,
        },
        "num_viewer_points": int(v_xyz.shape[0]),
        "num_cg_points": int(c_xyz.shape[0]),
        "num_primitives": int(primitive.num_primitives),
        "has_mesh": has_mesh,
        "has_glb": has_glb,
        "color_mode": cfg.color_mode,
        "normal_mode": cfg.normal_mode,
        "viewer_num_points_target": cfg.viewer_num_points,
        "cg_num_points_target": cfg.cg_num_points,
        "voxel_downsample": cfg.voxel_downsample,
        "voxel_size": cfg.voxel_size,
        "area_stats": {
            "total": float(areas.sum().item()),
            "mean": float(areas.mean().item()),
            "min": float(areas.min().item()),
            "max": float(areas.max().item()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Sidecar: attributes ---
    opacity = primitive.get_opacity()
    colors = primitive.get_colors()
    attrs = {
        "opacity": opacity.cpu().numpy() if opacity is not None else None,
        "area": areas.cpu().numpy(),
        "primitive_id": np.arange(primitive.num_primitives, dtype=np.int64),
    }
    if colors is not None:
        attrs["sh_dc"] = colors.cpu().numpy()
    if hasattr(primitive, "sh_coeffs") and primitive.sh_coeffs is not None:
        attrs["sh_coeffs"] = primitive.sh_coeffs.cpu().numpy()
    if hasattr(primitive, "sigma") and primitive.sigma is not None:
        sigma = primitive.sigma
        attrs["sigma"] = sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else np.asarray(sigma)
    # CG cloud tri_ids for research attribute tracing
    attrs["cg_tri_ids"] = c_tri.cpu().numpy()
    np.savez(attr_path, **attrs)

    return GeometryExportResult(
        viewer_ply=str(viewer_path),
        cg_ply=str(cg_path),
        mesh_ply=str(mesh_path) if has_mesh else None,
        mesh_glb_path=str(glb_path) if has_glb else None,
        metadata_path=str(meta_path),
        attributes_path=str(attr_path),
        num_viewer_points=int(v_xyz.shape[0]),
        num_cg_points=int(c_xyz.shape[0]),
        has_mesh=has_mesh,
        has_glb=has_glb,
    )


def export_viewer_point_cloud(
    primitive: BasePrimitive,
    output_dir: str | Path,
    *,
    method_name: str = "unknown",
    checkpoint_path: str | None = None,
    config: ViewerPointCloudExportConfig | None = None,
    generator: torch.Generator | None = None,
) -> ViewerPointCloudExportResult:
    """Export only the lightweight point cloud needed by the web viewer."""
    cfg = config or ViewerPointCloudExportConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    viewer_path = out / "geometry_viewer_points.ply"
    meta_path = out / "geometry_viewer_metadata.json"
    xyz, rgb, normals, _ = primitive_to_point_cloud(
        primitive,
        cfg.viewer_num_points,
        color_mode=cfg.color_mode,
        normal_mode=cfg.normal_mode,
        generator=generator,
    )

    if cfg.voxel_downsample and xyz.shape[0] > 0:
        xyz, rgb, normals = _voxel_downsample(xyz, rgb, normals, cfg.voxel_size)

    def _to_np_uint8(value):
        if value is None:
            return None
        return (value.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    def _to_np_f32(value):
        if value is None:
            return None
        return value.cpu().numpy().astype(np.float32)

    write_point_cloud_ply(
        viewer_path,
        _to_np_f32(xyz),
        _to_np_uint8(rgb),
        _to_np_f32(normals),
    )

    areas = primitive.compute_areas()
    metadata = {
        "method": method_name,
        "checkpoint": checkpoint_path,
        "schema_version": "1.0",
        "export_mode": "viewer_pointcloud",
        "assets": {
            "viewer_points": "geometry_viewer_points.ply",
        },
        "num_viewer_points": int(xyz.shape[0]),
        "num_primitives": int(primitive.num_primitives),
        "color_mode": cfg.color_mode,
        "normal_mode": cfg.normal_mode,
        "viewer_num_points_target": cfg.viewer_num_points,
        "voxel_downsample": cfg.voxel_downsample,
        "voxel_size": cfg.voxel_size,
        "area_stats": {
            "total": float(areas.sum().item()),
            "mean": float(areas.mean().item()),
            "min": float(areas.min().item()),
            "max": float(areas.max().item()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return ViewerPointCloudExportResult(
        viewer_ply=str(viewer_path),
        metadata_path=str(meta_path),
        num_viewer_points=int(xyz.shape[0]),
    )


def export_original_geometry(
    primitive: BasePrimitive,
    output_dir: str | Path,
    *,
    method_name: str = "unknown",
    checkpoint_path: str | None = None,
    filename: str = "geometry_original.ply",
    export_glb: bool = False,
) -> OriginalGeometryExportResult:
    """Export the primitive's original topology as a CG-readable PLY.

    Unlike ``export_viewer_geometry()``, this does not sample, downsample, or
    target a fixed number of points. Triangle-soup primitives are written as
    triangle-soup meshes; indexed primitives keep their shared vertex topology.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh_data = _extract_indexed_mesh_from_primitive(primitive)
    if mesh_data is None:
        raise ValueError("Primitive cannot be exported as original mesh geometry")

    verts, faces, rgb, normals = mesh_data
    geometry_path = out / filename
    glb_path = out / "geometry_original.glb" if export_glb else None
    meta_path = out / "geometry_metadata.json"
    attr_path = out / "geometry_attributes.npz"

    write_mesh_ply(geometry_path, verts, faces, rgb, normals)

    has_glb = False
    if glb_path is not None:
        try:
            write_mesh_glb(glb_path, verts, faces, rgb, normals)
            has_glb = True
        except Exception:
            pass

    areas = primitive.compute_areas()
    metadata = {
        "method": method_name,
        "checkpoint": checkpoint_path,
        "schema_version": "1.0",
        "export_mode": "original",
        "coordinate_system": "tribench_world",
        "up_axis": "+y",
        "scale": 1.0,
        "assets": {
            "geometry": filename,
            "mesh_glb": "geometry_original.glb" if has_glb else None,
        },
        "num_vertices": int(verts.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_primitives": int(primitive.num_primitives),
        "has_glb": has_glb,
        "sampling": None,
        "downsampling": None,
        "area_stats": {
            "total": float(areas.sum().item()),
            "mean": float(areas.mean().item()),
            "min": float(areas.min().item()),
            "max": float(areas.max().item()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    opacity = primitive.get_opacity()
    colors = primitive.get_colors()
    attrs = {
        "opacity": opacity.cpu().numpy() if opacity is not None else None,
        "area": areas.cpu().numpy(),
        "primitive_id": np.arange(primitive.num_primitives, dtype=np.int64),
    }
    if colors is not None:
        attrs["sh_dc"] = colors.cpu().numpy()
    if hasattr(primitive, "sh_coeffs") and primitive.sh_coeffs is not None:
        attrs["sh_coeffs"] = primitive.sh_coeffs.cpu().numpy()
    if hasattr(primitive, "sigma") and primitive.sigma is not None:
        sigma = primitive.sigma
        attrs["sigma"] = (
            sigma.detach().cpu().numpy()
            if isinstance(sigma, torch.Tensor)
            else np.asarray(sigma)
        )
    np.savez(attr_path, **attrs)

    return OriginalGeometryExportResult(
        geometry_ply=str(geometry_path),
        mesh_glb_path=str(glb_path) if has_glb else None,
        metadata_path=str(meta_path),
        attributes_path=str(attr_path),
        num_vertices=int(verts.shape[0]),
        num_faces=int(faces.shape[0]),
        has_glb=has_glb,
    )


def _extract_mesh_from_primitive(
    primitive: BasePrimitive,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Try to extract a mesh (vertices + faces) from a primitive.

    For triangle-soup primitives, each triangle has 3 independent vertices,
    so the mesh is just the triangles themselves with sequential indexing.
    """
    if not hasattr(primitive, "vertices"):
        return None

    verts = primitive.vertices  # [N, 3, 3]
    if not isinstance(verts, torch.Tensor) or verts.dim() != 3 or verts.shape[1:] != (3, 3):
        return None
    N = verts.shape[0]
    # Triangle soup: each triangle uses 3 unique vertices
    mesh_verts = verts.reshape(N * 3, 3).cpu().numpy().astype(np.float32)
    faces = np.arange(N * 3, dtype=np.int32).reshape(N, 3)

    # Colors
    colors = primitive.get_colors()
    rgb = None
    if colors is not None:
        # Repeat per-vertex
        rgb = colors.repeat_interleave(3, dim=0).cpu().numpy()
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)

    # Normals
    face_normals = primitive.compute_normals()  # [N, 3]
    normals = face_normals.repeat_interleave(3, dim=0).cpu().numpy().astype(np.float32)

    return mesh_verts, faces, rgb, normals
