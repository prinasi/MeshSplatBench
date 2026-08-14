"""Method-aware Unity asset packages.

``.triasset`` is deliberately a directory instead of a generic mesh format.
It keeps the renderer inputs that are not representable by OFF/COFF or GLB
(SH coefficients, opacity, splat widths, neural features, and MLP weights).
The package is consumed by the Unity TriBench package; standard CG previews
remain a separate, explicitly lossy export path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "1.0"
EXPORT_CONTRACT_REVISION = 2
PACKAGE_SUFFIX = ".triasset"


@dataclass
class BufferSpec:
    """Description of one little-endian, contiguous raw tensor buffer."""

    name: str
    file: str
    dtype: str
    shape: list[int]
    semantic: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "dtype": self.dtype,
            "shape": self.shape,
            "semantic": self.semantic,
        }


@dataclass
class TriAssetPackage:
    """In-memory package description returned by :func:`export_triasset`."""

    path: Path
    manifest_path: Path
    method: str
    buffers: dict[str, BufferSpec] = field(default_factory=dict)


def canonical_method(method: str) -> str:
    """Normalize public method aliases without importing CUDA-backed adapters."""

    value = method.strip().lower().replace("_", "-")
    aliases = {
        "2dts": "2dts",
        "d2ts": "2dts",
        "triangle-splatting": "triangle-splatting",
        "trianglesplatting": "triangle-splatting",
        "mesh-splatting": "mesh-splatting",
        "meshsplatting": "mesh-splatting",
        "diffsoup": "diffsoup",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        names = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported Unity export method {method!r}; expected one of: {names}") from exc


def resolve_checkpoint(method: str, checkpoint: str | Path) -> Path:
    """Resolve the native checkpoint layouts used by the four bundled methods."""

    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    if method in {"triangle-splatting", "mesh-splatting"}:
        direct = [
            path / "point_cloud_state_dict.pt",
            path / "ckpt" / "point_cloud_state_dict.pt",
        ]
        for candidate in direct:
            if candidate.is_file():
                return candidate
        candidates = list((path / "point_cloud").glob("iteration_*/point_cloud_state_dict.pt"))
        candidates += list((path / "ckpt" / "point_cloud").glob("iteration_*/point_cloud_state_dict.pt"))
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)

    if method == "2dts":
        candidates = []
        for root in (path / "ckpt", path):
            for suffix in ("*.ckpt", "*.pth", "*.pt"):
                candidates.extend(root.glob(suffix))
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)

    if method == "diffsoup":
        for candidate in (path / "final_params.pt", path / "ckpt" / "final_params.pt"):
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"Could not resolve a {method} checkpoint under {path}")


def export_triasset(
    method: str,
    checkpoint: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
    mesh_opacity_floor: float | None = None,
    export_topology: str = "indexed",
    d2ts_gamma_rescale: bool | None = None,
    background_color: str | None = None,
) -> TriAssetPackage:
    """Export the complete renderer state required for a Unity-native port.

    No CUDA extension or model class is imported: checkpoints are decoded on
    CPU, which makes this command safe to run on a workstation before Unity is
    installed.  The package contains raw tensors rather than baked colours.
    """

    method = canonical_method(method)
    export_topology = _canonical_export_topology(export_topology)
    if export_topology != "indexed" and method != "mesh-splatting":
        raise ValueError("--export-topology is only supported for mesh-splatting")
    background_color = _deployment_background_color(method, background_color)
    checkpoint_path = resolve_checkpoint(method, checkpoint)
    final_output_path = _package_path(output)
    if final_output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Unity asset already exists: {final_output_path}; use --force to replace it"
            )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_path.name}.tmp-", dir=final_output_path.parent
        )
    )
    buffers_dir = output_path / "buffers"
    buffers_dir.mkdir()

    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        writer = _PackageWriter(output_path)
        if method == "triangle-splatting":
            renderer = _export_triangle_splatting(writer, payload)
        elif method == "mesh-splatting":
            renderer = _export_mesh_splatting(
                writer,
                payload,
                checkpoint_path,
                opacity_floor_override=mesh_opacity_floor,
                export_topology=export_topology,
            )
        elif method == "2dts":
            renderer = _export_d2ts(writer, payload, gamma_rescale_override=d2ts_gamma_rescale)
        else:
            renderer = _export_diffsoup(writer, payload)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "export_contract_revision": EXPORT_CONTRACT_REVISION,
            "asset_kind": "tribench-unity-native",
            "method": method,
            "source": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
            },
            "coordinate_system": {
                "name": "tribench_world",
                "up_axis": "+y",
                "handedness": "source-defined",
                "unit_scale": 1.0,
            },
            "rendering": {
                # Triangle-splatting is trained and evaluated against PIL RGB / 255
                # tensors. These are native image code values, not a decoded
                # linear-light representation; marking them linear makes a Unity
                # Linear project apply an extra sRGB transfer on output.
                "color_space": "native_image_code_values",
                "requires_method_renderer": True,
                "generic_mesh_equivalent": False,
                "unity_profile_name": "method-aware",
                "cuda_equivalent": False,
                "background_color": background_color,
                "general_purpose": _general_purpose_contract(method),
                **renderer,
            },
            "buffers": {name: spec.to_dict() for name, spec in writer.buffers.items()},
            # JsonUtility does not deserialize dictionaries. Keep this redundant
            # list so the zero-dependency Unity loader can parse the same contract.
            "buffer_list": [
                {"name": name, **spec.to_dict()}
                for name, spec in writer.buffers.items()
            ],
        }
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output_path, ignore_errors=True)
        raise

    # Preserve the old complete package until the replacement has been built.
    # The staging directory is a sibling, so the final rename stays on one
    # filesystem. A failed decode therefore never destroys the last good asset.
    if final_output_path.exists():
        shutil.rmtree(final_output_path)
    output_path.rename(final_output_path)
    manifest_path = final_output_path / "manifest.json"
    return TriAssetPackage(final_output_path, manifest_path, method, writer.buffers)


class _PackageWriter:
    def __init__(self, root: Path):
        self.root = root
        self.buffers: dict[str, BufferSpec] = {}

    def tensor(self, name: str, value: Any, semantic: str, *, dtype: torch.dtype | None = None) -> None:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.detach().cpu().contiguous()
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        supported_dtypes = {torch.float32, torch.int32}
        uint32_dtype = getattr(torch, "uint32", None)
        if uint32_dtype is not None:
            supported_dtypes.add(uint32_dtype)
        if tensor.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported Unity buffer dtype for {name}: {tensor.dtype}")
        file_name = f"buffers/{name}.bin"
        with (self.root / file_name).open("wb") as handle:
            tensor.numpy().tofile(handle)
        self.buffers[name] = BufferSpec(
            name=name,
            file=file_name,
            dtype=_dtype_name(tensor.dtype),
            shape=[int(v) for v in tensor.shape],
            semantic=semantic,
        )


def _export_triangle_splatting(writer: _PackageWriter, state: Any) -> dict[str, Any]:
    state = _expect_dict(state, "triangle-splatting")
    triangles = _required_tensor(state, "triangles_points", "triangle-splatting")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"triangle-splatting triangles_points must be [N,3,3], got {tuple(triangles.shape)}")
    count = int(triangles.shape[0])
    writer.tensor("positions", triangles.reshape(-1, 3), "triangle-soup positions")
    writer.tensor("indices", torch.arange(count * 3, dtype=torch.int32).reshape(count, 3), "triangle indices")
    writer.tensor("opacity_logits", _required_tensor(state, "opacity", "triangle-splatting"), "per-triangle opacity logits")
    writer.tensor("sigma_logits", _required_tensor(state, "sigma", "triangle-splatting"), "per-triangle splat sigma logits")
    writer.tensor("sh_dc", _required_tensor(state, "features_dc", "triangle-splatting"), "per-triangle SH DC")
    writer.tensor("sh_rest", _required_tensor(state, "features_rest", "triangle-splatting"), "per-triangle SH residual")
    return {
        "renderer": "triangle-splatting",
        "primitive_topology": "triangle-soup",
        "primitive_count": count,
        "opacity_activation": "sigmoid",
        "sigma_activation": "0.01+exp",
        "native_coverage_space": "screen_space_projected_incenter",
        "native_compositing": "tile_depth_sorted_front_to_back",
        "requires_per_camera_depth_sort": True,
        "unity_method_aware_requires_per_camera_depth_sort": True,
        "unity_method_aware_status": "portability-baseline",
        "active_sh_degree": int(state.get("active_sh_degree", 0)),
    }


def _export_mesh_splatting(
    writer: _PackageWriter,
    state: Any,
    checkpoint_path: Path,
    *,
    opacity_floor_override: float | None = None,
    export_topology: str = "indexed",
) -> dict[str, Any]:
    state = _expect_dict(state, "mesh-splatting")
    vertices = _required_tensor(state, "triangles_points", "mesh-splatting")
    faces = _required_tensor(state, "_triangle_indices", "mesh-splatting")
    vertex_weight_logits = _required_tensor(state, "vertex_weight", "mesh-splatting")
    # Native MeshSplatting stores its global log-sigma as a Python float in
    # legacy point-cloud checkpoints and as a tensor in some newer variants.
    # Preserve either representation as a raw float buffer.
    sigma_logits = _required_value(state, "sigma", "mesh-splatting")
    writer.tensor("sigma_logits", sigma_logits, "mesh splat sigma logits")
    sh_dc = _required_tensor(state, "features_dc", "mesh-splatting")
    sh_rest = _required_tensor(state, "features_rest", "mesh-splatting")
    opacity_floor, source = _mesh_splatting_opacity_floor(
        state, checkpoint_path, override=opacity_floor_override
    )
    export_topology = _canonical_export_topology(export_topology)
    materialized_soup = export_topology == "soup"
    if materialized_soup:
        corner_vertices = _mesh_corner_vertex_indices(faces, vertices.shape[0])
        corner_count = int(corner_vertices.numel())
        writer.tensor("positions", vertices[corner_vertices], "materialized triangle-soup positions")
        writer.tensor(
            "indices",
            torch.arange(corner_count, dtype=torch.int32).reshape(-1, 3),
            "materialized triangle-soup sequential indices",
            dtype=torch.int32,
        )
        writer.tensor(
            "vertex_weight_logits",
            vertex_weight_logits[corner_vertices],
            "materialized per-corner opacity weight logits",
        )
        writer.tensor("sh_dc", sh_dc[corner_vertices], "materialized per-corner SH DC")
        writer.tensor("sh_rest", sh_rest[corner_vertices], "materialized per-corner SH residual")
        export_topology_label = "materialized-soup"
        primitive_topology = "triangle-soup"
        vertex_count = corner_count
        topology_expansion = float(corner_count) / float(vertices.shape[0]) if int(vertices.shape[0]) > 0 else 0.0
    else:
        writer.tensor("positions", vertices, "indexed-mesh positions")
        writer.tensor("indices", faces, "triangle indices", dtype=torch.int32)
        writer.tensor("vertex_weight_logits", vertex_weight_logits, "per-vertex opacity weight logits")
        writer.tensor("sh_dc", sh_dc, "per-vertex SH DC")
        writer.tensor("sh_rest", sh_rest, "per-vertex SH residual")
        export_topology_label = "indexed"
        primitive_topology = "indexed-triangles"
        vertex_count = int(vertices.shape[0])
        topology_expansion = 1.0
    triangle_opacity = mesh_splatting_triangle_opacity(
        vertex_weight_logits, faces, opacity_floor=opacity_floor
    )
    writer.tensor(
        "triangle_opacity",
        triangle_opacity,
        "per-triangle activated opacity: minimum of the three activated vertex weights",
    )
    sigma = math.exp(_scalar_float(sigma_logits, "mesh-splatting sigma"))
    terminal_solid_eligible = opacity_floor >= 0.999 and sigma <= 1e-3
    return {
        "renderer": "mesh-splatting",
        "primitive_topology": primitive_topology,
        "export_topology": export_topology_label,
        "source_vertex_count": int(vertices.shape[0]),
        "exported_vertex_count": vertex_count,
        "topology_vertex_expansion": topology_expansion,
        "primitive_count": int(faces.shape[0]),
        "opacity_activation": "opacity_floor+(1-opacity_floor)*sigmoid",
        "opacity_floor": opacity_floor,
        "opacity_floor_source": source,
        "triangle_opacity_reduction": "min_after_activation",
        "triangle_opacity_buffer": "triangle_opacity",
        "sigma_activation": "exp",
        "sigma": sigma,
        "terminal_solid_eligible": terminal_solid_eligible,
        "terminal_solid_condition": "opacity_floor>=0.999 and exp(sigma_logits)<=0.001",
        "native_coverage_space": "screen_space_projected_incenter",
        "native_compositing": "tile_depth_sorted_front_to_back",
        "requires_per_camera_depth_sort": True,
        "unity_method_aware_requires_per_camera_depth_sort": not terminal_solid_eligible,
        "unity_method_aware_status": (
            "opaque-terminal-solid" if terminal_solid_eligible else "portability-baseline"
        ),
        "active_sh_degree": int(state.get("active_sh_degree", 0)),
    }


def _canonical_export_topology(value: str | None) -> str:
    normalized = (value or "indexed").strip().lower().replace("_", "-")
    if normalized in {"indexed", "mesh", "indexed-mesh"}:
        return "indexed"
    if normalized in {"soup", "materialized-soup", "triangle-soup", "deindexed", "deindexed-soup"}:
        return "soup"
    raise ValueError("--export-topology must be indexed/mesh or soup/materialized-soup")


def _mesh_corner_vertex_indices(faces: torch.Tensor, vertex_count: int) -> torch.Tensor:
    indices = faces.detach().cpu().long()
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"mesh-splatting faces must be [F,3], got {tuple(indices.shape)}")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= int(vertex_count)):
        raise ValueError("mesh-splatting faces reference an invalid vertex")
    return indices.reshape(-1)


def mesh_splatting_triangle_opacity(
    vertex_weight_logits: torch.Tensor,
    faces: torch.Tensor,
    *,
    opacity_floor: float,
    chunk_faces: int = 1_000_000,
) -> torch.Tensor:
    """Reproduce native MeshSplatting's constant per-triangle opacity.

    The CUDA rasterizer activates each vertex weight first and then uses the
    minimum of the three values for every fragment of the triangle.  Exporting
    the reduced value avoids relying on provoking-vertex or interpolation
    behaviour in a graphics API.
    """

    if not 0.0 <= opacity_floor < 1.0:
        raise ValueError("opacity_floor must be in [0, 1)")
    logits = vertex_weight_logits.detach().cpu().to(dtype=torch.float32).reshape(-1)
    indices = faces.detach().cpu()
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"mesh-splatting faces must be [F,3], got {tuple(indices.shape)}")
    if chunk_faces <= 0:
        raise ValueError("chunk_faces must be positive")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= logits.numel()):
        raise ValueError("mesh-splatting faces reference an invalid vertex opacity")
    vertex_opacity = opacity_floor + (1.0 - opacity_floor) * torch.sigmoid(logits)
    triangle_opacity = torch.empty(indices.shape[0], dtype=torch.float32)
    for start in range(0, indices.shape[0], chunk_faces):
        end = min(start + chunk_faces, indices.shape[0])
        triangle_opacity[start:end] = vertex_opacity[indices[start:end].long()].amin(dim=1)
    return triangle_opacity


def _mesh_splatting_opacity_floor(
    state: dict[str, Any], checkpoint_path: Path, *, override: float | None
) -> tuple[float, str]:
    """Recover MeshSplatting's non-serialized terminal alpha floor.

    New checkpoints retain the exact value.  Legacy point-cloud checkpoints
    omitted it, so their default-paper schedule can be reconstructed from the
    ``iteration_<N>`` directory.  A caller may override this inference for a
    non-default schedule.
    """
    if override is not None:
        if not 0.0 <= override < 1.0:
            raise ValueError("mesh_opacity_floor must be in [0, 1)")
        return float(override), "export_override"
    if "opacity_floor" in state:
        raw_value = state["opacity_floor"]
        value = (
            float(raw_value.detach().cpu().reshape(-1)[0])
            if isinstance(raw_value, torch.Tensor)
            else float(raw_value)
        )
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Invalid mesh-splatting opacity_floor {value}")
        return value, "checkpoint"
    match = re.fullmatch(r"iteration_(\d+)", checkpoint_path.parent.name)
    if match is None:
        return 0.0, "legacy_unknown_default_zero"
    iteration = int(match.group(1))
    start, end, initial, final = 5_000, 24_000, 0.1, 0.9999
    if iteration <= start:
        return 0.0, "legacy_default_schedule"
    ratio = min(1.0, (iteration - start) / float(end - start))
    return initial + (final - initial) * ratio, "legacy_default_schedule"


def _required_value(state: dict[str, Any], key: str, method: str) -> Any:
    if key not in state:
        raise KeyError(f"{method} checkpoint missing value {key!r}")
    return state[key]


def _scalar_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        result = float(value.detach().cpu().reshape(-1)[0])
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _export_d2ts(
    writer: _PackageWriter,
    payload: Any,
    *,
    gamma_rescale_override: bool | None = None,
) -> dict[str, Any]:
    if isinstance(payload, tuple):
        state = _expect_dict(payload[0], "2dts")
        gamma = float(payload[3]) if len(payload) > 3 else 1.0
    else:
        state = _expect_dict(payload, "2dts")
        gamma = float(state.get("gamma", 1.0))
    checkpoint_gamma_rescale = state.get("gamma_rescale")
    if gamma_rescale_override is not None:
        gamma_rescale = bool(gamma_rescale_override)
        gamma_rescale_source = "export_override"
    elif checkpoint_gamma_rescale is not None:
        gamma_rescale = bool(checkpoint_gamma_rescale)
        gamma_rescale_source = "checkpoint"
    else:
        # Native TriBench 2DTS configs enable gamma_rescale when gamma is
        # annealed.  Old tuple checkpoints did not serialize the flag.
        gamma_rescale = gamma > 1.0
        gamma_rescale_source = "legacy_inferred_from_gamma"
    gamma_vertex_rescale = _d2ts_gamma_vertex_rescale(gamma) if gamma_rescale else 1.0
    triangles = _required_tensor(state, "_vertex", "2dts")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"2dts _vertex must be [N,3,3], got {tuple(triangles.shape)}")
    count = int(triangles.shape[0])
    f_dc = _required_tensor(state, "_f_dc", "2dts")
    f_rest = _required_tensor(state, "_f_rest", "2dts")
    # Legacy 2DTS checkpoints never serialise active_sh_degree (the trainer
    # only records it in its logs).  Derive the full degree from the stored
    # coefficient count, mirroring D2TSAdapter.load_checkpoint so Unity
    # renders the same SH bands as the native path.
    coeff_dim = f_rest.shape[-2] + 1
    max_sh_degree = int(math.sqrt(coeff_dim) - 1)
    writer.tensor("positions", triangles.reshape(-1, 3), "triangle-soup positions")
    writer.tensor("indices", torch.arange(count * 3, dtype=torch.int32).reshape(count, 3), "triangle indices")
    writer.tensor("opacity_logits", _required_tensor(state, "_opacity", "2dts"), "per-triangle opacity logits")
    writer.tensor("sh_dc", f_dc, "per-triangle or per-vertex SH DC")
    writer.tensor("sh_rest", f_rest, "per-triangle or per-vertex SH residual")
    return {
        "renderer": "2dts",
        "primitive_topology": "triangle-soup",
        "primitive_count": count,
        "opacity_activation": "sigmoid",
        "active_sh_degree": int(state.get("active_sh_degree", max_sh_degree)),
        "gamma": gamma,
        "gamma_rescale": gamma_rescale,
        "gamma_rescale_source": gamma_rescale_source,
        "gamma_vertex_rescale": gamma_vertex_rescale,
        "vertex_color": bool(f_dc.ndim == 4),
        "native_coverage_space": "screen_space_barycentric",
        "native_compositing": "depth_sorted_front_to_back",
        "requires_per_camera_depth_sort": True,
        "unity_method_aware_requires_per_camera_depth_sort": True,
        "unity_method_aware_status": "portability-baseline",
    }


def _d2ts_gamma_vertex_rescale(gamma: float) -> float:
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"2dts gamma must be finite and positive, got {gamma}")
    beta = 1.0 / gamma
    return 1.0 / math.sqrt((2.0**beta) * beta * math.gamma(beta))


def _export_diffsoup(writer: _PackageWriter, state: Any) -> dict[str, Any]:
    state = _expect_dict(state, "diffsoup")
    faces = _required_tensor(state, "F", "diffsoup")
    writer.tensor("positions", _required_tensor(state, "V", "diffsoup"), "indexed-mesh positions")
    writer.tensor("indices", faces, "triangle indices", dtype=torch.int32)
    writer.tensor("features", _required_tensor(state, "feat_acc", "diffsoup"), "per-triangle multiresolution neural features")
    writer.tensor("alpha", _required_tensor(state, "alpha_acc", "diffsoup"), "per-triangle multiresolution alpha")
    color_mlp = state.get("color_mlp")
    if not isinstance(color_mlp, dict):
        raise KeyError("DiffSoup checkpoint missing color_mlp state dict")
    mlp_buffers: dict[str, str] = {}
    for key, value in sorted(color_mlp.items()):
        if isinstance(value, torch.Tensor):
            name = "color_mlp__" + _safe_name(key)
            writer.tensor(name, value, f"DiffSoup ColorMLP parameter {key}")
            mlp_buffers[key] = name
    if not mlp_buffers:
        raise KeyError("DiffSoup color_mlp state dict contains no tensors")
    return {
        "renderer": "diffsoup",
        "primitive_topology": "indexed-triangles",
        "primitive_count": int(faces.shape[0]),
        "rmin": int(state.get("Rmin", 0)),
        "rmax": int(state["Rmax"]),
        "feature_dim": int(state["feat_dim"]),
        "alpha_test": "deterministic-0.5-for-benchmark",
        "unity_method_aware_requires_per_camera_depth_sort": False,
        "unity_method_aware_status": "requires-native-image-validation",
        "color_mlp_buffers": mlp_buffers,
    }


def _deployment_background_color(method: str, value: str | None) -> str:
    """Return the explicit evaluation background stored in the asset contract."""

    if value is None:
        return "white" if method == "2dts" else "black"
    normalized = str(value).strip().lower()
    if normalized not in {"black", "white"}:
        raise ValueError("background_color must be black or white for deterministic Unity evaluation")
    return normalized


def _general_purpose_contract(method: str) -> dict[str, Any]:
    """Describe whether a fair ordinary Unity Mesh condition can be constructed.

    DiffSoup has no SH-DC/vertex-colour field. Rendering it as a white mesh is an
    import diagnostic, not a comparable general-purpose appearance condition.
    """

    if method == "diffsoup":
        return {
            "supported": False,
            "appearance": "none",
            "reason": "requires a fixed-budget texture or vertex-colour bake",
        }
    return {
        "supported": True,
        "appearance": "sh_dc_vertex_color",
        "view_dependent_appearance": False,
        "opaque_depth_tested": True,
    }


def _package_path(output: str | Path) -> Path:
    path = Path(output).expanduser()
    return path if path.suffix == PACKAGE_SUFFIX else path.with_name(path.name + PACKAGE_SUFFIX)


def _expect_dict(value: Any, method: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected {method} checkpoint state dict, got {type(value)!r}")
    return value


def _required_tensor(state: dict[str, Any], key: str, method: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{method} checkpoint missing tensor {key!r}")
    return value


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name)


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.int32: "int32",
    }
    uint32_dtype = getattr(torch, "uint32", None)
    if uint32_dtype is not None:
        names[uint32_dtype] = "uint32"
    return names[dtype]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
