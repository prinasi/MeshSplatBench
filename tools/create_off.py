#!/usr/bin/env python3
"""Export Unity-readable mesh assets from TriBench checkpoints.

The filename is kept as ``create_off.py`` for compatibility with older
workflows.  The default export is now ``unity_mesh.off`` with a ``COFF`` header
and per-face colors, matching the lightweight Unity loader workflow used by the
reference triangle-splatting exports.  No sampling, downsampling, clustering, or
primitive-count limit is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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


@dataclass
class UnityMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_rgb: torch.Tensor | None = None
    face_rgb: torch.Tensor | None = None
    opacity: torch.Tensor | None = None
    method: str = "unknown"
    checkpoint: str | None = None
    color_source: str = "none"
    topology: str = "indexed"
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = _as_cpu_tensor(self.vertices, dtype=torch.float32)
        self.faces = _as_cpu_tensor(self.faces, dtype=torch.long)
        if self.vertex_rgb is not None:
            self.vertex_rgb = _as_u8_rgb(self.vertex_rgb, expected=self.vertices.shape[0])
        if self.face_rgb is not None:
            self.face_rgb = _as_u8_rgb(self.face_rgb, expected=self.faces.shape[0])
        if self.opacity is not None:
            self.opacity = _as_cpu_tensor(self.opacity, dtype=torch.float32).reshape(-1)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices [V, 3], got {tuple(self.vertices.shape)}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Expected triangular faces [F, 3], got {tuple(self.faces.shape)}")

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _as_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu()
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _as_u8_rgb(value: Any, *, expected: int) -> torch.Tensor:
    rgb = _as_cpu_tensor(value)
    if rgb.ndim == 1:
        rgb = rgb.reshape(-1, 3)
    if rgb.shape != (expected, 3):
        raise ValueError(f"Expected RGB [{expected}, 3], got {tuple(rgb.shape)}")
    if rgb.dtype == torch.uint8:
        return rgb.contiguous()
    if torch.is_floating_point(rgb):
        if float(rgb.max()) <= 1.0 + 1e-6:
            rgb = rgb.clamp(0.0, 1.0) * 255.0
        return rgb.round().clamp(0, 255).to(torch.uint8).contiguous()
    return rgb.clamp(0, 255).to(torch.uint8).contiguous()


def _parse_vec3(text: str) -> torch.Tensor:
    try:
        values = [float(item) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected x,y,z") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return torch.tensor(values, dtype=torch.float32)


def _filename_with_format(filename: str | None, fmt: str) -> str:
    suffix = _format_suffix(fmt)
    if not filename:
        return f"unity_mesh{suffix}"
    path = Path(filename)
    if path.suffix:
        return str(path)
    return str(path.with_suffix(suffix))


def _format_suffix(fmt: str) -> str:
    return ".off" if fmt in {"off", "coff"} else f".{fmt}"


def _canonical_method(name: str) -> str:
    from tribench.renderers.backends import canonical_backend_name

    return canonical_backend_name(str(name))


def _torch_load(path: Path) -> Any:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.exists():
            yield path


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    text = path.stem
    for prefix in ("iteration_", "iter_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    try:
        return int(text), path.name
    except ValueError:
        return -1, path.name


def _latest(paths: Iterable[Path]) -> Path | None:
    candidates = list(paths)
    if not candidates:
        return None
    return sorted(candidates, key=_numeric_suffix_key)[-1]


def _resolve_checkpoint_path(method: str, checkpoint: str | Path) -> Path:
    base = Path(checkpoint).expanduser()
    if base.is_file():
        return base

    if method in {"triangle-splatting", "mesh-splatting"}:
        direct = list(_iter_existing([
            base / "point_cloud_state_dict.pt",
            base / "ckpt" / "point_cloud_state_dict.pt",
        ]))
        if direct:
            return direct[0]
        latest = _latest((base / "point_cloud").glob("iteration_*"))
        if latest is not None and (latest / "point_cloud_state_dict.pt").is_file():
            return latest / "point_cloud_state_dict.pt"
        latest = _latest((base / "ckpt" / "point_cloud").glob("iteration_*"))
        if latest is not None and (latest / "point_cloud_state_dict.pt").is_file():
            return latest / "point_cloud_state_dict.pt"

    if method == "2dts":
        candidates = [
            *(base / "ckpt").glob("*.ckpt"),
            *(base / "ckpt").glob("*.pth"),
            *(base / "ckpt").glob("*.pt"),
            *base.glob("*.ckpt"),
            *base.glob("*.pth"),
            *base.glob("*.pt"),
        ]
        latest = _latest(candidates)
        if latest is not None:
            return latest
        ply_latest = _latest([*(base / "point_cloud").glob("*.ply"), *base.glob("*.ply")])
        if ply_latest is not None:
            return ply_latest

    if method == "diffsoup":
        direct = list(_iter_existing([
            base / "final_params.pt",
            base / "ckpt" / "final_params.pt",
        ]))
        if direct:
            return direct[0]

    raise FileNotFoundError(f"Could not resolve a {method} checkpoint under {base}")


def _infer_sh_degree(coeff_count: int) -> int | None:
    degree = int(math.sqrt(coeff_count) - 1)
    if degree < 0 or degree > 4:
        return None
    if (degree + 1) ** 2 != coeff_count:
        return None
    return degree


def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate Gaussian-splatting SH coefficients.

    Args:
        degree: active degree, clamped to 0..4.
        sh: [N, 3, C] coefficients.
        dirs: [N, 3] normalized directions.
    """
    degree = max(0, min(int(degree), 4))
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


def _dc_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    dc = _as_cpu_tensor(features_dc, dtype=torch.float32)
    if dc.ndim >= 3 and dc.shape[-2] == 1:
        dc = dc.squeeze(-2)
    if dc.ndim != 2 or dc.shape[-1] != 3:
        raise ValueError(f"Expected DC colors [N, 1, 3] or [N, 3], got {tuple(dc.shape)}")
    return (_SH_C0 * dc + 0.5).clamp(0.0, 1.0)


def _sh_rgb(
    features_dc: torch.Tensor,
    features_rest: torch.Tensor | None,
    reference_points: torch.Tensor,
    *,
    active_degree: int | None,
    camera_center: torch.Tensor,
) -> torch.Tensor:
    dc = _as_cpu_tensor(features_dc, dtype=torch.float32)
    rest = None if features_rest is None else _as_cpu_tensor(features_rest, dtype=torch.float32)
    if rest is None or rest.numel() == 0:
        return _dc_rgb(dc)

    if dc.ndim != 3 or dc.shape[1:] != (1, 3):
        return _dc_rgb(dc)
    if rest.ndim != 3 or rest.shape[-1] != 3:
        return _dc_rgb(dc)

    coeffs = torch.cat((dc, rest), dim=1)
    degree = _infer_sh_degree(int(coeffs.shape[1]))
    if degree is None:
        return _dc_rgb(dc)
    if active_degree is not None:
        degree = min(degree, int(active_degree))

    refs = _as_cpu_tensor(reference_points, dtype=torch.float32)
    center = _as_cpu_tensor(camera_center, dtype=torch.float32)
    dirs = refs - center.reshape(1, 3)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp(min=1e-8)
    sh = coeffs.permute(0, 2, 1).contiguous()
    return (_eval_sh(degree, sh, dirs) + 0.5).clamp(0.0, 1.0)


def _opacity_rgb(opacity: torch.Tensor) -> torch.Tensor:
    t = _as_cpu_tensor(opacity, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)
    r = (2.0 * t).clamp(0.0, 1.0)
    b = (2.0 * (1.0 - t)).clamp(0.0, 1.0)
    g = (1.0 - 2.0 * torch.abs(t - 0.5)).clamp(0.0, 1.0) * 0.5
    return torch.stack([r, g, b], dim=-1)


def _choose_rgb(
    *,
    color_mode: str,
    features_dc: torch.Tensor | None,
    features_rest: torch.Tensor | None,
    reference_points: torch.Tensor,
    active_degree: int | None,
    opacity: torch.Tensor | None,
    camera_center: torch.Tensor,
) -> tuple[torch.Tensor | None, str]:
    mode = color_mode.lower()
    if mode == "none":
        return None, "none"
    if mode == "white":
        return torch.ones(reference_points.shape[0], 3, dtype=torch.float32), "white"
    if mode == "opacity":
        if opacity is None:
            return torch.ones(reference_points.shape[0], 3, dtype=torch.float32), "white(no_opacity)"
        return _opacity_rgb(opacity), "opacity"

    if features_dc is None:
        if mode == "auto" and opacity is not None:
            return _opacity_rgb(opacity), "opacity(auto)"
        return torch.ones(reference_points.shape[0], 3, dtype=torch.float32), "white(no_sh)"

    if mode == "dc":
        return _dc_rgb(features_dc), "sh_dc"
    if mode in {"auto", "sh"}:
        return _sh_rgb(
            features_dc,
            features_rest,
            reference_points,
            active_degree=active_degree,
            camera_center=camera_center,
        ), "sh" if features_rest is not None else "sh_dc"
    raise ValueError(f"Unsupported color mode: {color_mode}")


def _face_rgb_from_vertex_rgb(vertex_rgb: torch.Tensor | None, faces: torch.Tensor) -> torch.Tensor | None:
    if vertex_rgb is None:
        return None
    rgb = vertex_rgb[faces.long()].to(torch.float32).mean(dim=1).round().to(torch.uint8)
    return rgb


def _load_triangle_splatting(path: Path, *, color_mode: str, camera_center: torch.Tensor) -> UnityMesh:
    state = _torch_load(path)
    vertices_3 = _as_cpu_tensor(state["triangles_points"], dtype=torch.float32)
    face_count = int(vertices_3.shape[0])
    vertices = vertices_3.reshape(face_count * 3, 3)
    faces = torch.arange(face_count * 3, dtype=torch.long).reshape(face_count, 3)
    opacity = torch.sigmoid(_as_cpu_tensor(state.get("opacity", torch.ones(face_count)), dtype=torch.float32)).reshape(-1)
    face_rgb, color_source = _choose_rgb(
        color_mode=color_mode,
        features_dc=state.get("features_dc"),
        features_rest=state.get("features_rest"),
        reference_points=vertices_3.mean(dim=1),
        active_degree=state.get("active_sh_degree"),
        opacity=opacity,
        camera_center=camera_center,
    )
    return UnityMesh(
        vertices=vertices,
        faces=faces,
        vertex_rgb=face_rgb.repeat_interleave(3, dim=0) if face_rgb is not None else None,
        face_rgb=face_rgb,
        opacity=opacity,
        method="triangle-splatting",
        checkpoint=str(path),
        color_source=color_source,
        topology="triangle_soup",
        extras={"source_format": "point_cloud_state_dict.pt"},
    )


def _load_mesh_splatting(path: Path, *, color_mode: str, camera_center: torch.Tensor) -> UnityMesh:
    state = _torch_load(path)
    vertices = _as_cpu_tensor(state["triangles_points"], dtype=torch.float32)
    faces = _as_cpu_tensor(state["_triangle_indices"], dtype=torch.long)
    vertex_weight = state.get("vertex_weight")
    opacity = None
    if vertex_weight is not None:
        # Reuse the native-package legacy recovery so the generic preview and
        # method-specific export do not interpret the same checkpoint with
        # different opacity floors.
        from tribench.unity_assets import _mesh_splatting_opacity_floor

        opacity_floor, _ = _mesh_splatting_opacity_floor(state, path, override=None)
        vertex_opacity = opacity_floor + (1.0 - opacity_floor) * torch.sigmoid(
            _as_cpu_tensor(vertex_weight, dtype=torch.float32).reshape(-1)
        )
        # Native MeshSplatting assigns one constant opacity to a triangle: the
        # minimum of its three activated vertex weights.  The generic opaque
        # mesh path still writes alpha=255, but opacity visualisation and
        # metadata must not silently use a different reduction.
        opacity = vertex_opacity[faces.long()].amin(dim=1)

    vertex_rgb, color_source = _choose_rgb(
        color_mode=color_mode,
        features_dc=state.get("features_dc"),
        features_rest=state.get("features_rest"),
        reference_points=vertices,
        active_degree=state.get("active_sh_degree"),
        opacity=vertex_opacity if vertex_weight is not None else None,
        camera_center=camera_center,
    )
    face_rgb = _face_rgb_from_vertex_rgb(
        _as_u8_rgb(vertex_rgb, expected=vertices.shape[0]) if vertex_rgb is not None else None,
        faces,
    )
    if face_rgb is None and opacity is not None:
        face_rgb = _as_u8_rgb(_opacity_rgb(opacity), expected=faces.shape[0])
        color_source = "opacity(auto)"
    return UnityMesh(
        vertices=vertices,
        faces=faces,
        vertex_rgb=vertex_rgb,
        face_rgb=face_rgb,
        opacity=opacity,
        method="mesh-splatting",
        checkpoint=str(path),
        color_source=color_source,
        topology="indexed",
        extras={"source_format": "point_cloud_state_dict.pt"},
    )


def _unwrap_d2ts_checkpoint(loaded: Any) -> tuple[dict[str, Any], float | None]:
    if isinstance(loaded, tuple):
        state = loaded[0]
        gamma = float(loaded[3]) if len(loaded) >= 4 else None
        return state, gamma
    if isinstance(loaded, dict):
        gamma = float(loaded["gamma"]) if "gamma" in loaded else None
        return loaded, gamma
    raise TypeError(f"Unsupported 2DTS checkpoint payload: {type(loaded)!r}")


def _load_d2ts(path: Path, *, color_mode: str, camera_center: torch.Tensor) -> UnityMesh:
    if path.suffix.lower() == ".ply":
        raise ValueError("Direct CPU export from 2DTS PLY checkpoints is not implemented; use a .ckpt/.pt checkpoint.")

    state, gamma = _unwrap_d2ts_checkpoint(_torch_load(path))
    vertices_3 = _as_cpu_tensor(state["_vertex"], dtype=torch.float32)
    face_count = int(vertices_3.shape[0])
    vertices = vertices_3.reshape(face_count * 3, 3)
    faces = torch.arange(face_count * 3, dtype=torch.long).reshape(face_count, 3)
    opacity = torch.sigmoid(_as_cpu_tensor(state.get("_opacity", torch.ones(face_count)), dtype=torch.float32)).reshape(-1)

    f_dc = state.get("_f_dc")
    f_rest = state.get("_f_rest")
    face_rgb = None
    vertex_rgb = None
    color_source = "none"
    if isinstance(f_dc, torch.Tensor) and f_dc.ndim == 4:
        # Vertex-color 2DTS layout: [T, 3, C, 3].
        flat_dc = f_dc.reshape(face_count * 3, *f_dc.shape[2:])
        flat_rest = f_rest.reshape(face_count * 3, *f_rest.shape[2:]) if isinstance(f_rest, torch.Tensor) else None
        vertex_rgb, color_source = _choose_rgb(
            color_mode=color_mode,
            features_dc=flat_dc,
            features_rest=flat_rest,
            reference_points=vertices,
            active_degree=state.get("active_sh_degree"),
            opacity=opacity.repeat_interleave(3),
            camera_center=camera_center,
        )
        if vertex_rgb is not None:
            vertex_rgb_u8 = _as_u8_rgb(vertex_rgb, expected=vertices.shape[0])
            face_rgb = _face_rgb_from_vertex_rgb(vertex_rgb_u8, faces)
            vertex_rgb = vertex_rgb_u8
    else:
        face_rgb, color_source = _choose_rgb(
            color_mode=color_mode,
            features_dc=f_dc,
            features_rest=f_rest,
            reference_points=vertices_3.mean(dim=1),
            active_degree=state.get("active_sh_degree"),
            opacity=opacity,
            camera_center=camera_center,
        )
        vertex_rgb = face_rgb.repeat_interleave(3, dim=0) if face_rgb is not None else None

    return UnityMesh(
        vertices=vertices,
        faces=faces,
        vertex_rgb=vertex_rgb,
        face_rgb=face_rgb,
        opacity=opacity,
        method="2dts",
        checkpoint=str(path),
        color_source=color_source,
        topology="triangle_soup",
        extras={"source_format": path.suffix.lower().lstrip("."), "gamma": gamma},
    )


def _load_diffsoup(path: Path, *, color_mode: str, camera_center: torch.Tensor) -> UnityMesh:
    del camera_center
    state = _torch_load(path)
    vertices = _as_cpu_tensor(state["V"], dtype=torch.float32)
    faces = _as_cpu_tensor(state["F"], dtype=torch.long)
    alpha = state.get("alpha_acc")
    opacity = None
    if isinstance(alpha, torch.Tensor):
        opacity = _as_cpu_tensor(alpha, dtype=torch.float32).reshape(alpha.shape[0], -1).mean(dim=1)
    face_rgb, color_source = _choose_rgb(
        color_mode=color_mode,
        features_dc=None,
        features_rest=None,
        reference_points=torch.empty(faces.shape[0], 3),
        active_degree=None,
        opacity=opacity,
        camera_center=torch.zeros(3),
    )
    return UnityMesh(
        vertices=vertices,
        faces=faces,
        face_rgb=face_rgb,
        opacity=opacity,
        method="diffsoup",
        checkpoint=str(path),
        color_source=color_source,
        topology="indexed",
        extras={
            "source_format": "final_params.pt",
            "Rmax": int(state["Rmax"]) if "Rmax" in state else None,
            "feat_dim": int(state["feat_dim"]) if "feat_dim" in state else None,
        },
    )


def _mesh_from_adapter(
    method: str,
    checkpoint: Path,
    *,
    color_mode: str,
    camera_center: torch.Tensor,
) -> UnityMesh:
    del color_mode, camera_center
    from tribench.core.builder import build_adapter
    from tribench.core.viewer_geometry import _extract_indexed_mesh_from_primitive

    adapter = build_adapter({"type": method, "checkpoint": str(checkpoint)})
    primitive = adapter.to_primitive()
    mesh_data = _extract_indexed_mesh_from_primitive(primitive)
    if mesh_data is None:
        raise RuntimeError("Adapter fallback could not extract mesh geometry")
    vertices, faces, vertex_rgb, _normals = mesh_data
    return UnityMesh(
        vertices=torch.from_numpy(vertices),
        faces=torch.from_numpy(faces),
        vertex_rgb=torch.from_numpy(vertex_rgb) if vertex_rgb is not None else None,
        face_rgb=_face_rgb_from_vertex_rgb(torch.from_numpy(vertex_rgb), torch.from_numpy(faces))
        if vertex_rgb is not None else None,
        opacity=primitive.get_opacity() if hasattr(primitive, "get_opacity") else None,
        method=method,
        checkpoint=str(checkpoint),
        color_source="adapter",
        topology="adapter",
        extras={"source_format": "adapter_fallback"},
    )


def load_unity_mesh(
    method: str,
    checkpoint: str | Path,
    *,
    color_mode: str,
    camera_center: torch.Tensor,
    use_adapter_fallback: bool,
) -> UnityMesh:
    method = _canonical_method(method)
    ckpt = _resolve_checkpoint_path(method, checkpoint)
    loaders = {
        "triangle-splatting": _load_triangle_splatting,
        "mesh-splatting": _load_mesh_splatting,
        "2dts": _load_d2ts,
        "diffsoup": _load_diffsoup,
    }
    try:
        return loaders[method](ckpt, color_mode=color_mode, camera_center=camera_center)
    except Exception:
        if not use_adapter_fallback:
            raise
        return _mesh_from_adapter(
            method,
            ckpt,
            color_mode=color_mode,
            camera_center=camera_center,
        )


def write_coff(
    path: str | Path,
    mesh: UnityMesh,
    *,
    chunk_rows: int = 1_000_000,
) -> None:
    """Write an OFF file with a COFF header and per-face RGBA colors."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy().astype(np.float32, copy=False)
    faces = mesh.faces.numpy().astype(np.int64, copy=False)
    if mesh.face_rgb is not None:
        face_rgb = mesh.face_rgb.numpy().astype(np.uint8, copy=False)
    elif mesh.vertex_rgb is not None:
        face_rgb = _face_rgb_from_vertex_rgb(mesh.vertex_rgb, mesh.faces).numpy()
    else:
        face_rgb = np.full((faces.shape[0], 3), 255, dtype=np.uint8)

    with out.open("w", encoding="ascii") as f:
        f.write("COFF\n")
        f.write(f"{vertices.shape[0]} {faces.shape[0]} 0\n")
        np.savetxt(f, vertices, fmt="%.9g %.9g %.9g")
        for start in range(0, faces.shape[0], chunk_rows):
            end = min(start + chunk_rows, faces.shape[0])
            count = end - start
            rows = np.empty((count, 8), dtype=np.int64)
            rows[:, 0] = 3
            rows[:, 1:4] = faces[start:end]
            rows[:, 4:7] = face_rgb[start:end].astype(np.int64)
            rows[:, 7] = 255
            np.savetxt(f, rows, fmt="%d %d %d %d %d %d %d %d")


def _mesh_for_vertex_color_format(mesh: UnityMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices = mesh.vertices.numpy().astype(np.float32, copy=False)
    faces = mesh.faces.numpy().astype(np.int32, copy=False)
    if mesh.vertex_rgb is not None:
        return vertices, faces, mesh.vertex_rgb.numpy().astype(np.uint8, copy=False)
    if mesh.face_rgb is None:
        return vertices, faces, None

    face_rgb = mesh.face_rgb
    expanded_vertices = mesh.vertices[mesh.faces.long()].reshape(mesh.num_faces * 3, 3)
    expanded_faces = torch.arange(mesh.num_faces * 3, dtype=torch.int32).reshape(mesh.num_faces, 3)
    expanded_rgb = face_rgb.repeat_interleave(3, dim=0)
    return (
        expanded_vertices.numpy().astype(np.float32, copy=False),
        expanded_faces.numpy().astype(np.int32, copy=False),
        expanded_rgb.numpy().astype(np.uint8, copy=False),
    )


def write_ply(path: str | Path, mesh: UnityMesh) -> None:
    from tribench.core.viewer_geometry import write_mesh_ply

    vertices, faces, rgb = _mesh_for_vertex_color_format(mesh)
    write_mesh_ply(path, vertices, faces, rgb, None)


def write_obj(path: str | Path, mesh: UnityMesh, *, material_levels: int) -> str:
    from tribench.core.viewer_geometry import write_mesh_obj

    vertices, faces, rgb = _mesh_for_vertex_color_format(mesh)
    return write_mesh_obj(path, vertices, faces, rgb, None, material_levels=material_levels)


def write_glb(path: str | Path, mesh: UnityMesh) -> None:
    from tribench.core.viewer_geometry import write_mesh_glb

    vertices, faces, rgb = _mesh_for_vertex_color_format(mesh)
    write_mesh_glb(path, vertices, faces, rgb, None)


def write_metadata(
    path: str | Path,
    mesh: UnityMesh,
    *,
    config_path: str | None,
    output_asset: Path,
    fmt: str,
    camera_center: torch.Tensor,
) -> None:
    meta = {
        "schema_version": "1.0",
        "exporter": "tools/create_off.py",
        "export_mode": "unity_original_mesh",
        "method": mesh.method,
        "config": config_path,
        "checkpoint": mesh.checkpoint,
        "asset": output_asset.name,
        "format": fmt,
        "coff_face_color_layout": fmt in {"off", "coff"},
        "num_vertices": mesh.num_vertices,
        "num_faces": mesh.num_faces,
        "topology": mesh.topology,
        "color_source": mesh.color_source,
        "camera_center_for_baked_sh": [float(v) for v in camera_center.tolist()],
        "sampling": None,
        "downsampling": None,
        "primitive_limit": None,
        "coordinate_system": "tribench_world",
        "up_axis": "+y",
        "scale": 1.0,
        "extras": mesh.extras,
    }
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _resolve_from_config(
    config: str | None,
    *,
    method: str | None,
    checkpoint: str | None,
    output_dir: str | None,
) -> tuple[str, str, Path, Any | None]:
    from tribench.cli.config import adapter_config, load_cli_config, output_dir as config_output_dir

    cfg = load_cli_config(config) if config else None
    adapter_cfg = adapter_config(cfg, method=method, checkpoint=checkpoint)
    resolved_method = adapter_cfg.get("type")
    resolved_checkpoint = adapter_cfg.get("checkpoint")
    if resolved_method is None or resolved_checkpoint is None:
        raise ValueError("Could not resolve adapter type and checkpoint")

    if output_dir:
        out_dir = Path(output_dir).expanduser()
    elif cfg is not None:
        run_dir = config_output_dir(cfg)
        if run_dir is not None:
            out_dir = Path(run_dir).expanduser() / "unity_mesh"
        else:
            out_dir = Path(resolved_checkpoint).expanduser()
            if out_dir.is_file():
                out_dir = out_dir.parent
            out_dir = out_dir / "unity_mesh"
    else:
        out_dir = Path(resolved_checkpoint).expanduser()
        if out_dir.is_file():
            out_dir = out_dir.parent
        out_dir = out_dir / "unity_mesh"

    return str(resolved_method), str(resolved_checkpoint), out_dir, cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export original TriBench topology as a Unity-readable mesh. "
            "Default output is COFF-style .off with per-face colors."
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
        help="Checkpoint path. Overrides config adapter.checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory. Default with --config: <output.dir>/unity_mesh",
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
        choices=("off", "coff", "ply", "obj", "glb"),
        default="off",
        help="Unity mesh format. off/coff writes a COFF header with per-face colors.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("auto", "sh", "dc", "opacity", "white", "none"),
        default="auto",
        help="How to bake static colors for Unity mesh viewing.",
    )
    parser.add_argument(
        "--camera",
        type=_parse_vec3,
        default=torch.zeros(3),
        help="Camera center x,y,z used when baking SH colors. Default: 0,0,0.",
    )
    parser.add_argument(
        "--obj-material-levels",
        type=int,
        default=8,
        help="Quantization levels per channel for OBJ/MTL face materials.",
    )
    parser.add_argument(
        "--no-adapter-fallback",
        action="store_true",
        help="Disable slower adapter fallback when direct CPU checkpoint parsing fails.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write unity_mesh_metadata.json.",
    )
    args = parser.parse_args()

    if args.method and not args.checkpoint:
        parser.error("--checkpoint is required when using --method")

    method, checkpoint, output_dir, _cfg = _resolve_from_config(
        args.config,
        method=args.method,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )

    filename = _filename_with_format(args.filename, args.format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fmt = output_path.suffix.lower().lstrip(".")
    if fmt == "coff":
        fmt = "coff"
    elif fmt == "off":
        fmt = "off"
    elif fmt not in {"ply", "obj", "glb"}:
        parser.error("Output extension must be .off, .coff, .ply, .obj, or .glb")

    print(f"[create_off] method={method}")
    print(f"[create_off] checkpoint={checkpoint}")
    print(f"[create_off] output={output_path}")
    print("[create_off] export_policy=original_topology,no_sampling,no_downsampling,no_limit")

    mesh = load_unity_mesh(
        method,
        checkpoint,
        color_mode=args.color_mode,
        camera_center=args.camera,
        use_adapter_fallback=not args.no_adapter_fallback,
    )

    if fmt in {"off", "coff"}:
        write_coff(output_path, mesh)
    elif fmt == "ply":
        write_ply(output_path, mesh)
    elif fmt == "obj":
        write_obj(output_path, mesh, material_levels=args.obj_material_levels)
    elif fmt == "glb":
        write_glb(output_path, mesh)
    else:  # pragma: no cover - guarded above
        raise AssertionError(fmt)

    metadata_path = output_dir / "unity_mesh_metadata.json"
    if not args.no_metadata:
        write_metadata(
            metadata_path,
            mesh,
            config_path=args.config,
            output_asset=output_path,
            fmt=fmt,
            camera_center=args.camera,
        )

    print(f"[create_off] saved:    {output_path}")
    if not args.no_metadata:
        print(f"[create_off] metadata: {metadata_path}")
    print(f"[create_off] vertices: {mesh.num_vertices}")
    print(f"[create_off] faces:    {mesh.num_faces}")
    print(f"[create_off] topology: {mesh.topology}")
    print(f"[create_off] colors:   {mesh.color_source}")


if __name__ == "__main__":
    main()
