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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tribench.primitives.base import BasePrimitive

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
_SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
_SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


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
    """Extract SH degree-0 RGB, shape [N, 3] in [0, 1]."""
    if sh_coeffs is None:
        return None
    if sh_coeffs.shape[0] != num_prims or sh_coeffs.shape[-1] < 3:
        return None
    dc = sh_coeffs[:, :3]
    return (dc * _SH_C0 + 0.5).clamp(0, 1)


def _infer_sh_degree(flat_coeffs: torch.Tensor) -> int | None:
    """Infer SH degree from flattened [N, coeff_count * 3] coefficients."""
    if flat_coeffs.dim() != 2 or flat_coeffs.shape[-1] < 3:
        return None
    channels = int(flat_coeffs.shape[-1])
    if channels % 3 != 0:
        return None
    coeff_count = channels // 3
    degree = int(np.sqrt(coeff_count) - 1)
    if (degree + 1) ** 2 != coeff_count or degree < 0 or degree > 4:
        return None
    return degree


def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics, matching the Gaussian-splatting SH layout."""
    assert 0 <= degree <= 4
    coeff_count = (degree + 1) ** 2
    assert sh.shape[-1] >= coeff_count

    result = _SH_C0 * sh[..., 0]
    if degree > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - _SH_C1 * y * sh[..., 1] + _SH_C1 * z * sh[..., 2] - _SH_C1 * x * sh[..., 3]

        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _SH_C2[0] * xy * sh[..., 4]
                + _SH_C2[1] * yz * sh[..., 5]
                + _SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + _SH_C2[3] * xz * sh[..., 7]
                + _SH_C2[4] * (xx - yy) * sh[..., 8]
            )

            if degree > 2:
                result = (
                    result
                    + _SH_C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + _SH_C3[1] * xy * z * sh[..., 10]
                    + _SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + _SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + _SH_C3[5] * z * (xx - yy) * sh[..., 14]
                    + _SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if degree > 3:
                    result = (
                        result
                        + _SH_C4[0] * xy * (xx - yy) * sh[..., 16]
                        + _SH_C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + _SH_C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + _SH_C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + _SH_C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + _SH_C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + _SH_C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + _SH_C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + _SH_C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def _primitive_reference_points(primitive: BasePrimitive) -> torch.Tensor:
    points = getattr(primitive, "points", None)
    if isinstance(points, torch.Tensor) and points.dim() == 2 and points.shape[1] == 3:
        return points
    return _primitive_triangle_vertices(primitive).mean(dim=1)


def _colors_from_sh(
    sh_coeffs: torch.Tensor | None,
    reference_points: torch.Tensor,
    *,
    camera_center: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Bake SH colors from the origin-facing direction used by create_off.py."""
    if sh_coeffs is None:
        return None
    degree = _infer_sh_degree(sh_coeffs)
    if degree is None:
        return None

    ref = reference_points.to(device=sh_coeffs.device, dtype=sh_coeffs.dtype)
    center = (
        torch.zeros(3, device=sh_coeffs.device, dtype=sh_coeffs.dtype)
        if camera_center is None
        else camera_center.to(device=sh_coeffs.device, dtype=sh_coeffs.dtype)
    )
    dirs = ref - center
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp(min=1e-8)

    coeff_count = (degree + 1) ** 2
    sh = sh_coeffs.reshape(sh_coeffs.shape[0], coeff_count, 3).permute(0, 2, 1).contiguous()
    return (_eval_sh(degree, sh, dirs) + 0.5).clamp(0, 1)


def _primitive_display_colors(primitive: BasePrimitive) -> torch.Tensor | None:
    """Return baked per-primitive RGB in [0, 1] where possible."""
    sh_coeffs = getattr(primitive, "sh_coeffs", None)
    if isinstance(sh_coeffs, torch.Tensor):
        colors = None
        if sh_coeffs.shape[0] == primitive.num_primitives:
            colors = _colors_from_sh(sh_coeffs, _primitive_reference_points(primitive))
        else:
            colors = _vertex_sh_face_colors(primitive, sh_coeffs)
        if colors is not None:
            return colors

    colors = primitive.get_colors()
    if colors is None:
        return None
    return colors.float().clamp(0, 1)


def _vertex_sh_face_colors(
    primitive: BasePrimitive,
    sh_coeffs: torch.Tensor,
) -> torch.Tensor | None:
    """Bake per-face colors from per-vertex SH coefficients.

    Indexed meshes store one SH row per vertex; display colors are baked
    per vertex and then averaged over each triangle's three corners so the
    result aligns with per-primitive color indexing downstream.
    """
    vertices = getattr(primitive, "vertices", None)
    faces = getattr(primitive, "faces", None)
    if not (
        isinstance(vertices, torch.Tensor)
        and vertices.dim() == 2
        and vertices.shape[1] == 3
        and isinstance(faces, torch.Tensor)
        and faces.dim() == 2
        and faces.shape[1] == 3
    ):
        return None
    if sh_coeffs.shape[0] != vertices.shape[0]:
        return None
    vertex_colors = _colors_from_sh(sh_coeffs, vertices)
    if vertex_colors is None:
        return None
    device = faces.device
    if vertex_colors.device != device:
        vertex_colors = vertex_colors.to(device)
    return vertex_colors[faces].mean(dim=1)


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
    *,
    text: bool = True,
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
    PlyData(els, text=text).write(str(path))


def _quantized_material_names(
    face_rgb: np.ndarray,
    *,
    levels: int = 8,
) -> tuple[list[str], dict[str, tuple[float, float, float]]]:
    levels = max(2, int(levels))
    rgb01 = face_rgb.astype(np.float32) / 255.0
    quantized = np.rint(rgb01 * (levels - 1)).astype(np.int32).clip(0, levels - 1)

    names: list[str] = []
    materials: dict[str, tuple[float, float, float]] = {}
    denom = float(levels - 1)
    for q in quantized:
        name = f"mat_{q[0]:02d}_{q[1]:02d}_{q[2]:02d}"
        names.append(name)
        if name not in materials:
            materials[name] = (float(q[0] / denom), float(q[1] / denom), float(q[2] / denom))
    return names, materials


def write_mesh_obj(
    path: str | Path,
    vertices: np.ndarray,       # [V, 3] float32
    faces: np.ndarray,          # [F, 3] int32
    rgb: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    *,
    material_levels: int = 8,
) -> str:
    """Write a Unity/Blender-friendly OBJ with companion MTL.

    OBJ is used as the broad-compatibility mesh format. The writer emits both
    vertex-color extension values on ``v`` rows and quantized MTL materials per
    face, so tools that ignore vertex colors still get visible baked color.
    """
    obj_path = Path(path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces [F, 3], got {faces.shape}")

    rgb_u8 = None
    if rgb is not None:
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        if rgb_u8.shape != (vertices.shape[0], 3):
            rgb_u8 = None

    normals_f32 = None
    if normals is not None:
        normals_f32 = np.asarray(normals, dtype=np.float32)
        if normals_f32.shape != (vertices.shape[0], 3):
            normals_f32 = None

    if rgb_u8 is not None and faces.shape[0] > 0:
        face_rgb = np.rint(rgb_u8[faces].astype(np.float32).mean(axis=1)).astype(np.uint8)
        face_materials, materials = _quantized_material_names(
            face_rgb,
            levels=material_levels,
        )
    else:
        face_materials = ["mat_default"] * faces.shape[0]
        materials = {"mat_default": (0.8, 0.8, 0.8)}

    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("# TriBench material palette\n")
        for name in sorted(materials):
            r, g, b = materials[name]
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"Ka {r * 0.2:.6f} {g * 0.2:.6f} {b * 0.2:.6f}\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("d 1.000000\n")
            f.write("illum 1\n\n")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# TriBench CG mesh export\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("o tribench_geometry\n")
        for idx, v in enumerate(vertices):
            if rgb_u8 is not None:
                c = rgb_u8[idx].astype(np.float32) / 255.0
                f.write(
                    f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g} "
                    f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
                )
            else:
                f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")

        if normals_f32 is not None:
            for n in normals_f32:
                f.write(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n")

        active_material = None
        for face, material in zip(faces, face_materials, strict=True):
            if material != active_material:
                f.write(f"usemtl {material}\n")
                active_material = material
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            if normals_f32 is not None:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")

    return str(mtl_path)


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

        # Colors: try vertex colors first. Per-face colors are preserved by
        # triangle-soup fallback where each face owns its vertices.
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
    export_obj: bool = True
    export_glb: bool = False
    voxel_downsample: bool = True
    voxel_size: float = 0.01


@dataclass
class GeometryExportResult:
    viewer_ply: str
    cg_ply: str
    mesh_ply: str | None = None
    mesh_obj_path: str | None = None
    mesh_glb_path: str | None = None
    metadata_path: str | None = None
    attributes_path: str | None = None
    num_viewer_points: int = 0
    num_cg_points: int = 0
    has_mesh: bool = False
    has_obj: bool = False
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
class PointCloudExportResult:
    ply: str
    num_points: int = 0
    has_rgb: bool = False
    has_normals: bool = False


@dataclass
class IndexedMeshPlyExportResult:
    ply: str
    num_vertices: int = 0
    num_faces: int = 0
    has_rgb: bool = False
    has_normals: bool = False


def export_indexed_mesh_ply(
    primitive: BasePrimitive,
    output_path: str | Path,
    *,
    color_mode: str = "dc",
    normal_mode: str = "none",
) -> IndexedMeshPlyExportResult:
    """Export an indexed primitive without sampling or duplicating vertices.

    Colors are the primitive's view-independent per-vertex display colors.
    Large meshes are written as binary little-endian PLY to keep export time
    and file size practical.
    """
    vertices = getattr(primitive, "vertices", None)
    faces = getattr(primitive, "faces", None)
    if not (
        isinstance(vertices, torch.Tensor)
        and vertices.dim() == 2
        and vertices.shape[1] == 3
        and isinstance(faces, torch.Tensor)
        and faces.dim() == 2
        and faces.shape[1] == 3
    ):
        raise ValueError("Primitive does not expose indexed triangular mesh topology")

    path = Path(output_path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".ply")
    if path.suffix.lower() != ".ply":
        raise ValueError("Mesh output must use a .ply extension")
    path.parent.mkdir(parents=True, exist_ok=True)

    if color_mode == "dc":
        colors = primitive.get_colors()
    elif color_mode == "white":
        colors = torch.ones(
            vertices.shape[0], 3, device=vertices.device, dtype=vertices.dtype
        )
    elif color_mode == "opacity":
        vertex_weights = getattr(primitive, "vertex_weights", None)
        colors = (
            _opacity_colormap(vertex_weights.reshape(-1))
            if isinstance(vertex_weights, torch.Tensor)
            and vertex_weights.numel() == vertices.shape[0]
            else None
        )
    else:
        raise ValueError(f"Unsupported indexed mesh color mode: {color_mode}")

    rgb = None
    if colors is not None:
        if colors.shape != (vertices.shape[0], 3):
            raise ValueError(
                "Indexed mesh colors must have shape [num_vertices, 3], "
                f"got {tuple(colors.shape)}"
            )
        rgb = (colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    normals = None
    if normal_mode == "primitive":
        normals = getattr(primitive, "vertex_normals", None)
        if not isinstance(normals, torch.Tensor) or normals.shape != vertices.shape:
            face_vertices = vertices[faces]
            face_normals = torch.cross(
                face_vertices[:, 1] - face_vertices[:, 0],
                face_vertices[:, 2] - face_vertices[:, 0],
                dim=-1,
            )
            normals = torch.zeros_like(vertices)
            for corner in range(3):
                normals.index_add_(0, faces[:, corner], face_normals)
            normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
    elif normal_mode != "none":
        raise ValueError(f"Unsupported indexed mesh normal mode: {normal_mode}")
    normals_np = None
    if isinstance(normals, torch.Tensor) and normals.shape == vertices.shape:
        normals_np = normals.detach().cpu().numpy().astype(np.float32)

    vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    if faces_np.size:
        min_index = int(faces_np.min())
        max_index = int(faces_np.max())
        if min_index < 0 or max_index >= vertices_np.shape[0]:
            raise ValueError(
                f"Mesh face indices [{min_index}, {max_index}] are invalid for "
                f"{vertices_np.shape[0]} vertices"
            )

    write_mesh_ply(
        path,
        vertices_np,
        faces_np,
        rgb,
        normals_np,
        text=False,
    )
    return IndexedMeshPlyExportResult(
        ply=str(path),
        num_vertices=int(vertices_np.shape[0]),
        num_faces=int(faces_np.shape[0]),
        has_rgb=rgb is not None,
        has_normals=normals_np is not None,
    )


def export_point_cloud_ply(
    primitive: BasePrimitive,
    output_path: str | Path,
    *,
    num_points: int,
    color_mode: str = "dc",
    normal_mode: str = "none",
    voxel_size: float = 0.0,
    generator: torch.Generator | None = None,
) -> PointCloudExportResult:
    """Export one standard vertex-only PLY point cloud from a primitive."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if voxel_size < 0:
        raise ValueError("voxel_size must be non-negative")

    xyz, rgb, normals, _ = primitive_to_point_cloud(
        primitive,
        num_points,
        color_mode=color_mode,
        normal_mode=normal_mode,
        generator=generator,
    )
    if voxel_size > 0 and xyz.shape[0] > 0:
        xyz, rgb, normals = _voxel_downsample(xyz, rgb, normals, voxel_size)

    def _to_np_uint8(value):
        if value is None:
            return None
        return (value.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    def _to_np_f32(value):
        if value is None:
            return None
        return value.cpu().numpy().astype(np.float32)

    path = Path(output_path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".ply")
    if path.suffix.lower() != ".ply":
        raise ValueError("Point-cloud output must use a .ply extension")
    path.parent.mkdir(parents=True, exist_ok=True)

    write_point_cloud_ply(
        path,
        _to_np_f32(xyz),
        _to_np_uint8(rgb),
        _to_np_f32(normals),
    )

    return PointCloudExportResult(
        ply=str(path),
        num_points=int(xyz.shape[0]),
        has_rgb=rgb is not None,
        has_normals=normals is not None,
    )


@dataclass
class OriginalGeometryExportResult:
    geometry_ply: str | None = None
    geometry_path: str | None = None
    mesh_obj_path: str | None = None
    mesh_glb_path: str | None = None
    metadata_path: str | None = None
    attributes_path: str | None = None
    num_vertices: int = 0
    num_faces: int = 0
    has_obj: bool = False
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
        colors = _primitive_display_colors(primitive)
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
    obj_path = out / "geometry_mesh.obj" if cfg.export_mesh and cfg.export_obj else None

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
    has_obj = False
    has_glb = False
    glb_path = out / "geometry_mesh.glb" if cfg.export_glb else None
    if cfg.export_mesh:
        try:
            mesh_data = _extract_indexed_mesh_from_primitive(primitive)
            if mesh_data is not None:
                m_verts, m_faces, m_rgb, m_nrm = mesh_data
                write_mesh_ply(mesh_path, m_verts, m_faces, m_rgb, m_nrm)
                has_mesh = True
                if obj_path is not None:
                    write_mesh_obj(obj_path, m_verts, m_faces, m_rgb, m_nrm)
                    has_obj = True
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
            "mesh_obj": "geometry_mesh.obj" if has_obj else None,
            "mesh_glb": "geometry_mesh.glb" if has_glb else None,
        },
        "num_viewer_points": int(v_xyz.shape[0]),
        "num_cg_points": int(c_xyz.shape[0]),
        "num_primitives": int(primitive.num_primitives),
        "has_mesh": has_mesh,
        "has_obj": has_obj,
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
        mesh_obj_path=str(obj_path) if has_obj else None,
        mesh_glb_path=str(glb_path) if has_glb else None,
        metadata_path=str(meta_path),
        attributes_path=str(attr_path),
        num_viewer_points=int(v_xyz.shape[0]),
        num_cg_points=int(c_xyz.shape[0]),
        has_mesh=has_mesh,
        has_obj=has_obj,
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
    export_obj: bool = False,
    export_glb: bool = False,
) -> OriginalGeometryExportResult:
    """Export the primitive's original topology as a CG-readable mesh.

    Unlike ``export_viewer_geometry()``, this does not sample, downsample, or
    target a fixed number of points. Triangle-soup primitives are written as
    triangle-soup meshes; indexed primitives keep their shared vertex topology.

    The primary format is inferred from ``filename``:
      * ``.obj`` writes OBJ + MTL, recommended for Unity's built-in importer.
      * ``.ply`` writes a standard mesh PLY for research/point-cloud tooling.
      * ``.glb`` writes binary glTF when ``trimesh`` is available.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh_data = _extract_indexed_mesh_from_primitive(primitive)
    if mesh_data is None:
        raise ValueError("Primitive cannot be exported as original mesh geometry")

    verts, faces, rgb, normals = mesh_data
    geometry_path = out / filename
    suffix = geometry_path.suffix.lower()
    if suffix == "":
        geometry_path = geometry_path.with_suffix(".obj")
        suffix = ".obj"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)

    extra_obj_path = (
        out / "geometry_original.obj"
        if export_obj and suffix != ".obj"
        else None
    )
    glb_path = (
        out / "geometry_original.glb"
        if export_glb and suffix != ".glb"
        else None
    )
    meta_path = out / "geometry_metadata.json"
    attr_path = out / "geometry_attributes.npz"

    geometry_ply: str | None = None
    mesh_obj_path: str | None = None
    has_obj = False
    has_glb = False

    if suffix == ".ply":
        write_mesh_ply(geometry_path, verts, faces, rgb, normals)
        geometry_ply = str(geometry_path)
    elif suffix == ".obj":
        write_mesh_obj(geometry_path, verts, faces, rgb, normals)
        mesh_obj_path = str(geometry_path)
        has_obj = True
    elif suffix == ".glb":
        try:
            write_mesh_glb(geometry_path, verts, faces, rgb, normals)
            has_glb = True
        except Exception as exc:
            raise RuntimeError(f"Could not write GLB geometry: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported mesh extension '{suffix}'. Use .obj, .ply, or .glb; "
            "FBX export requires Autodesk FBX SDK or another external converter."
        )

    if extra_obj_path is not None:
        write_mesh_obj(extra_obj_path, verts, faces, rgb, normals)
        mesh_obj_path = str(extra_obj_path)
        has_obj = True

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
            "geometry": geometry_path.name,
            "geometry_format": suffix.lstrip("."),
            "mesh_ply": Path(geometry_ply).name if geometry_ply else None,
            "mesh_obj": Path(mesh_obj_path).name if mesh_obj_path else None,
            "mesh_glb": (
                geometry_path.name
                if suffix == ".glb" and has_glb
                else "geometry_original.glb" if has_glb and glb_path is not None else None
            ),
        },
        "num_vertices": int(verts.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_primitives": int(primitive.num_primitives),
        "has_obj": has_obj,
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
        geometry_ply=geometry_ply,
        geometry_path=str(geometry_path),
        mesh_obj_path=mesh_obj_path,
        mesh_glb_path=str(geometry_path if suffix == ".glb" else glb_path) if has_glb else None,
        metadata_path=str(meta_path),
        attributes_path=str(attr_path),
        num_vertices=int(verts.shape[0]),
        num_faces=int(faces.shape[0]),
        has_obj=has_obj,
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
    colors = _primitive_display_colors(primitive)
    rgb = None
    if colors is not None:
        # Repeat per-vertex
        rgb = colors.repeat_interleave(3, dim=0).cpu().numpy()
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)

    # Normals
    face_normals = primitive.compute_normals()  # [N, 3]
    normals = face_normals.repeat_interleave(3, dim=0).cpu().numpy().astype(np.float32)

    return mesh_verts, faces, rgb, normals
