#!/usr/bin/env python3
"""Calculate local connectivity metrics for MeshSplatting meshes.

The default input is a TriBench MeshSplatting config.  The script resolves the
configured checkpoint, reads the indexed mesh stored in
``point_cloud_state_dict.pt`` (``triangles_points`` + ``_triangle_indices``),
and reports topology metrics that are less easily hidden by global averages.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    source_type: str
    source_path: Path
    source_format: str


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    text = path.stem
    for prefix in ("iteration_", "iter_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    try:
        return int(text), path.name
    except ValueError:
        return -1, path.name


def _latest(paths: Iterable[Path]) -> Path | None:
    candidates = list(paths)
    if not candidates:
        return None
    return sorted(candidates, key=_numeric_suffix_key)[-1]


def _canonical_method(name: str) -> str:
    from tribench.renderers.backends import canonical_backend_name

    return canonical_backend_name(str(name))


def _resolve_checkpoint_path(method: str, checkpoint: str | Path) -> Path:
    method = _canonical_method(method)
    base = Path(checkpoint).expanduser()
    if base.is_file():
        return base

    if method == "mesh-splatting":
        direct = [
            base / "point_cloud_state_dict.pt",
            base / "ckpt" / "point_cloud_state_dict.pt",
        ]
        for path in direct:
            if path.is_file():
                return path

        latest = _latest((base / "point_cloud").glob("iteration_*"))
        if latest is not None and (latest / "point_cloud_state_dict.pt").is_file():
            return latest / "point_cloud_state_dict.pt"

        latest = _latest((base / "ckpt" / "point_cloud").glob("iteration_*"))
        if latest is not None and (latest / "point_cloud_state_dict.pt").is_file():
            return latest / "point_cloud_state_dict.pt"

    raise FileNotFoundError(f"Could not resolve a {method} checkpoint under {base}")


def _as_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def load_mesh_splatting_checkpoint(checkpoint: str | Path) -> MeshData:
    """Load MeshSplatting's indexed triangle mesh from a checkpoint."""
    import torch

    path = Path(checkpoint).expanduser()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if "triangles_points" not in state or "_triangle_indices" not in state:
        raise KeyError(
            "MeshSplatting checkpoint must contain 'triangles_points' and "
            "'_triangle_indices'."
        )
    vertices = _as_numpy(state["triangles_points"], dtype=np.float64)
    faces = _as_numpy(state["_triangle_indices"], dtype=np.int64)
    return MeshData(
        vertices=vertices,
        faces=faces,
        source_type="checkpoint",
        source_path=path,
        source_format="point_cloud_state_dict.pt",
    )


def load_mesh_file(mesh_file: str | Path) -> MeshData:
    """Load an exported triangular mesh file through trimesh."""
    import trimesh

    path = Path(mesh_file).expanduser()
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry found in mesh file: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        raise TypeError(f"Unsupported mesh payload in {path}: {type(loaded)!r}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces [F, 3], got {faces.shape}")
    return MeshData(
        vertices=vertices,
        faces=faces,
        source_type="mesh_file",
        source_path=path,
        source_format=path.suffix.lower().lstrip(".") or "mesh",
    )


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.size == 0:
        empty_edges = np.zeros((0, 2), dtype=np.int64)
        return empty_edges, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges.sort(axis=1)
    unique_edges, inverse, counts = np.unique(
        edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return unique_edges, inverse.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def _connected_components(
    num_faces: int,
    edge_inverse: np.ndarray,
    edge_counts: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    if num_faces == 0:
        return 0, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    face_ids = np.repeat(np.arange(num_faces, dtype=np.int64), 3)
    order = np.argsort(edge_inverse, kind="mergesort")
    sorted_edge_ids = edge_inverse[order]
    sorted_face_ids = face_ids[order]

    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_edge_ids)) + 1))
    first_faces = sorted_face_ids[starts]
    repeat_counts = edge_counts - 1

    linked_mask = np.ones(sorted_face_ids.shape[0], dtype=bool)
    linked_mask[starts] = False
    cols = sorted_face_ids[linked_mask]
    rows = np.repeat(first_faces, repeat_counts)

    if rows.size == 0:
        labels = np.arange(num_faces, dtype=np.int64)
        sizes = np.ones(num_faces, dtype=np.int64)
        return num_faces, labels, sizes

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(num_faces, num_faces),
    )
    component_count, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=component_count).astype(np.int64, copy=False)
    return int(component_count), labels.astype(np.int64, copy=False), sizes


def _is_manifold_vertex_link(link_edges: np.ndarray) -> bool:
    """Return whether a vertex link is one circle or one open interval."""
    if link_edges.shape[0] == 0:
        return False

    unique_link_edges = np.unique(link_edges, axis=0)
    if unique_link_edges.shape[0] != link_edges.shape[0]:
        return False

    nodes, inverse = np.unique(unique_link_edges.reshape(-1), return_inverse=True)
    if nodes.shape[0] == 0:
        return False
    pairs = inverse.reshape(-1, 2)
    degree = np.bincount(pairs.reshape(-1), minlength=nodes.shape[0])
    if int(degree.max(initial=0)) > 2:
        return False

    endpoint_count = int(np.count_nonzero(degree == 1))
    if endpoint_count not in {0, 2}:
        return False

    parent = np.arange(nodes.shape[0], dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in pairs:
        union(int(a), int(b))

    roots = {find(i) for i in range(nodes.shape[0]) if degree[i] > 0}
    return len(roots) == 1


def _non_manifold_vertices(
    num_vertices: int,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect vertices whose incident faces are not a single disk-like fan."""
    non_manifold = np.zeros(num_vertices, dtype=bool)
    used = np.zeros(num_vertices, dtype=bool)
    if faces.size == 0:
        non_manifold[:] = True
        return non_manifold, used

    face_count = faces.shape[0]
    owners = np.empty(face_count * 3, dtype=np.int64)
    link_edges = np.empty((face_count * 3, 2), dtype=np.int64)

    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]
    owners[0::3] = a
    owners[1::3] = b
    owners[2::3] = c
    link_edges[0::3] = np.sort(np.stack((b, c), axis=1), axis=1)
    link_edges[1::3] = np.sort(np.stack((a, c), axis=1), axis=1)
    link_edges[2::3] = np.sort(np.stack((a, b), axis=1), axis=1)

    order = np.argsort(owners, kind="mergesort")
    owners = owners[order]
    link_edges = link_edges[order]

    starts = np.concatenate(([0], np.flatnonzero(np.diff(owners)) + 1))
    ends = np.concatenate((starts[1:], [owners.shape[0]]))
    grouped_vertices = owners[starts]
    used[grouped_vertices] = True
    non_manifold[~used] = True

    for vertex, start, end in zip(grouped_vertices, starts, ends, strict=True):
        if not _is_manifold_vertex_link(link_edges[start:end]):
            non_manifold[int(vertex)] = True

    return non_manifold, used


def calculate_connectivity_metrics(
    mesh: MeshData,
    *,
    config_path: str | Path | None = None,
    method: str | None = None,
    dataset: str | None = None,
    scene: str | None = None,
    include_vertex_manifoldness: bool = True,
    top_components: int = 10,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces [F, 3], got {faces.shape}")

    num_vertices = int(vertices.shape[0])
    num_faces = int(faces.shape[0])

    if num_faces == 0:
        valid_index_mask = np.zeros((0,), dtype=bool)
    else:
        valid_index_mask = np.all((faces >= 0) & (faces < num_vertices), axis=1)
    valid_face_count = int(np.count_nonzero(valid_index_mask))
    invalid_face_count = num_faces - valid_face_count
    valid_faces = faces[valid_index_mask]
    repeated_vertex_mask = (
        (valid_faces[:, 0] == valid_faces[:, 1])
        | (valid_faces[:, 1] == valid_faces[:, 2])
        | (valid_faces[:, 2] == valid_faces[:, 0])
        if valid_faces.size
        else np.zeros((0,), dtype=bool)
    )
    degenerate_face_count = int(np.count_nonzero(repeated_vertex_mask))
    analysis_faces = valid_faces[~repeated_vertex_mask]

    areas = _face_areas(vertices, analysis_faces)
    total_area = float(areas.sum(dtype=np.float64))
    zero_area_mask = areas <= 1e-12
    zero_area_count = int(np.count_nonzero(zero_area_mask))

    unique_edges, edge_inverse, edge_counts = _unique_edges(analysis_faces)
    num_edges = int(unique_edges.shape[0])
    boundary_edge_mask = edge_counts == 1
    non_manifold_edge_mask = edge_counts > 2
    boundary_edge_count = int(np.count_nonzero(boundary_edge_mask))
    non_manifold_edge_count = int(np.count_nonzero(non_manifold_edge_mask))
    manifold_edge_count = int(np.count_nonzero(edge_counts == 2))

    if num_edges:
        valence = np.bincount(unique_edges.reshape(-1), minlength=num_vertices)
        boundary_vertices = np.unique(unique_edges[boundary_edge_mask].reshape(-1))
    else:
        valence = np.zeros(num_vertices, dtype=np.int64)
        boundary_vertices = np.zeros((0,), dtype=np.int64)
    used_vertex_mask = valence > 0
    used_vertex_count = int(np.count_nonzero(used_vertex_mask))
    isolated_vertex_count = int(num_vertices - used_vertex_count)
    boundary_vertex_count = int(boundary_vertices.shape[0])

    component_count, component_labels, component_face_counts = _connected_components(
        analysis_faces.shape[0],
        edge_inverse,
        edge_counts,
    )
    if component_count:
        component_areas = np.bincount(
            component_labels,
            weights=areas,
            minlength=component_count,
        ).astype(np.float64, copy=False)
        largest_component_id = int(np.argmax(component_face_counts))
        largest_component_faces = int(component_face_counts[largest_component_id])
        largest_component_area = float(component_areas[largest_component_id])
        component_order = np.argsort(-component_face_counts, kind="stable")
    else:
        component_areas = np.zeros((0,), dtype=np.float64)
        largest_component_id = -1
        largest_component_faces = 0
        largest_component_area = 0.0
        component_order = np.zeros((0,), dtype=np.int64)

    if include_vertex_manifoldness:
        non_manifold_vertex_mask, incident_vertex_mask = _non_manifold_vertices(
            num_vertices,
            analysis_faces,
        )
        non_manifold_vertex_count = int(np.count_nonzero(non_manifold_vertex_mask))
        incident_vertex_count = int(np.count_nonzero(incident_vertex_mask))
    else:
        non_manifold_vertex_count = None
        incident_vertex_count = None

    top_component_rows: list[dict[str, Any]] = []
    for rank, component_id in enumerate(component_order[: max(top_components, 0)], start=1):
        face_count = int(component_face_counts[component_id])
        area = float(component_areas[component_id])
        top_component_rows.append(
            {
                "rank": rank,
                "component_id": int(component_id),
                "face_count": face_count,
                "face_ratio": _safe_ratio(face_count, num_faces),
                "valid_face_ratio": _safe_ratio(face_count, analysis_faces.shape[0]),
                "area": area,
                "area_ratio": _safe_ratio(area, total_area),
            }
        )

    metrics: dict[str, Any] = {
        "vertex_face_ratio": _safe_ratio(num_vertices, num_faces),
        "average_vertex_valence": _safe_ratio(float(valence.sum()), num_vertices),
        "average_used_vertex_valence": _safe_ratio(
            float(valence[used_vertex_mask].sum()),
            used_vertex_count,
        ),
        "boundary_edge_ratio": _safe_ratio(boundary_edge_count, num_edges),
        "non_manifold_edge_ratio": _safe_ratio(non_manifold_edge_count, num_edges),
        "boundary_vertex_ratio": _safe_ratio(boundary_vertex_count, num_vertices),
        "largest_component_ratio": _safe_ratio(largest_component_faces, num_faces),
        "largest_component_valid_face_ratio": _safe_ratio(
            largest_component_faces,
            analysis_faces.shape[0],
        ),
        "largest_component_area_ratio": _safe_ratio(largest_component_area, total_area),
        "isolated_vertex_ratio": _safe_ratio(isolated_vertex_count, num_vertices),
        "invalid_face_ratio": _safe_ratio(invalid_face_count, num_faces),
        "degenerate_face_ratio": _safe_ratio(degenerate_face_count, num_faces),
        "zero_area_face_ratio": _safe_ratio(zero_area_count, analysis_faces.shape[0]),
    }
    if non_manifold_vertex_count is not None:
        metrics["non_manifold_vertex_ratio"] = _safe_ratio(
            non_manifold_vertex_count,
            num_vertices,
        )

    counts: dict[str, Any] = {
        "vertices": num_vertices,
        "faces": num_faces,
        "valid_faces": valid_face_count,
        "analyzed_faces": int(analysis_faces.shape[0]),
        "edges": num_edges,
        "used_vertices": used_vertex_count,
        "isolated_vertices": isolated_vertex_count,
        "boundary_edges": boundary_edge_count,
        "manifold_edges": manifold_edge_count,
        "non_manifold_edges": non_manifold_edge_count,
        "boundary_vertices": boundary_vertex_count,
        "invalid_faces": invalid_face_count,
        "degenerate_faces": degenerate_face_count,
        "zero_area_faces": zero_area_count,
    }
    if non_manifold_vertex_count is not None:
        counts["non_manifold_vertices"] = non_manifold_vertex_count
        counts["incident_vertices_for_vertex_check"] = incident_vertex_count

    area_stats = {
        "total": total_area,
        "mean": float(areas.mean()) if areas.size else 0.0,
        "min": float(areas.min()) if areas.size else 0.0,
        "max": float(areas.max()) if areas.size else 0.0,
    }
    edge_stats = {
        "max_face_incidence": int(edge_counts.max(initial=0)),
        "boundary_edge_count": boundary_edge_count,
        "boundary_edge_ratio": metrics["boundary_edge_ratio"],
        "non_manifold_edge_count": non_manifold_edge_count,
        "non_manifold_edge_ratio": metrics["non_manifold_edge_ratio"],
    }
    vertex_stats = {
        "average_valence": metrics["average_vertex_valence"],
        "average_used_valence": metrics["average_used_vertex_valence"],
        "boundary_vertex_count": boundary_vertex_count,
        "boundary_vertex_ratio": metrics["boundary_vertex_ratio"],
        "isolated_vertex_count": isolated_vertex_count,
        "isolated_vertex_ratio": metrics["isolated_vertex_ratio"],
    }
    if non_manifold_vertex_count is not None:
        vertex_stats["non_manifold_vertex_count"] = non_manifold_vertex_count
        vertex_stats["non_manifold_vertex_ratio"] = metrics["non_manifold_vertex_ratio"]

    components = {
        "count": component_count,
        "single_face_component_count": int(np.count_nonzero(component_face_counts == 1)),
        "largest_component_id": largest_component_id,
        "largest_component_faces": largest_component_faces,
        "largest_component_ratio": metrics["largest_component_ratio"],
        "largest_component_valid_face_ratio": metrics["largest_component_valid_face_ratio"],
        "largest_component_area": largest_component_area,
        "largest_component_area_ratio": metrics["largest_component_area_ratio"],
        "top": top_component_rows,
    }

    quality_flags = {
        "edge_manifold": non_manifold_edge_count == 0,
        "closed_edge_manifold": boundary_edge_count == 0 and non_manifold_edge_count == 0,
        "single_component": component_count == 1,
    }
    if non_manifold_vertex_count is not None:
        quality_flags["vertex_manifold"] = non_manifold_vertex_count == 0
        quality_flags["closed_manifold_candidate"] = (
            boundary_edge_count == 0
            and non_manifold_edge_count == 0
            and non_manifold_vertex_count == 0
        )

    return {
        "schema_version": "1.0",
        "method": method,
        "dataset": dataset,
        "scene": scene,
        "config": str(config_path) if config_path is not None else None,
        "source": {
            "type": mesh.source_type,
            "path": str(mesh.source_path),
            "format": mesh.source_format,
        },
        "counts": counts,
        "metrics": metrics,
        "edges": edge_stats,
        "vertices": vertex_stats,
        "components": components,
        "area": area_stats,
        "quality_flags": quality_flags,
        "definitions": {
            "face_adjacency": "faces are adjacent when they share an undirected edge",
            "boundary_edge": "an undirected edge incident to exactly one analyzed face",
            "non_manifold_edge": "an undirected edge incident to more than two analyzed faces",
            "non_manifold_vertex": (
                "a vertex whose incident-face link is not one connected circle "
                "or one connected open interval"
            ),
            "analyzed_faces": "valid faces with three distinct vertex indices",
        },
    }


def _section(cfg: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if cfg is None:
        return {}
    from tribench.core.config import Config

    return Config(cfg.get(name, {})).to_dict()


def _resolve_from_config(
    config: str | Path | None,
    *,
    checkpoint: str | None,
    mesh_file: str | None,
    source: str | None,
    output: str | None,
) -> tuple[MeshData, dict[str, Any], Path | None]:
    from tribench.cli.config import adapter_config, load_cli_config, output_dir

    cfg = load_cli_config(config) if config is not None else None
    adapter_cfg = adapter_config(cfg, checkpoint=checkpoint) if cfg is not None else {}
    dataset_cfg = _section(cfg, "dataset")
    connectivity_cfg = _section(cfg, "connectivity")

    selected_source = source
    if selected_source is None:
        selected_source = "mesh-file" if mesh_file is not None else "checkpoint"

    if selected_source == "mesh-file":
        mesh_path = mesh_file or connectivity_cfg.get("mesh") or connectivity_cfg.get("mesh_file")
        if mesh_path is None and cfg is not None:
            mesh_cfg = _section(cfg, "mesh")
            output_cfg = _section(cfg, "output")
            mesh_path = mesh_cfg.get("output") or output_cfg.get("mesh_file")
        if mesh_path is None:
            raise ValueError("--mesh-file is required when --source=mesh-file")
        mesh = load_mesh_file(mesh_path)
    else:
        method = adapter_cfg.get("type")
        checkpoint_value = adapter_cfg.get("checkpoint")
        if config is None:
            method = "mesh-splatting"
            checkpoint_value = checkpoint
        if method is None or checkpoint_value is None:
            raise ValueError("Could not resolve MeshSplatting checkpoint from config.")
        method = _canonical_method(str(method))
        if method != "mesh-splatting":
            raise ValueError(
                "Checkpoint connectivity loading is currently implemented for "
                f"mesh-splatting, got {method!r}."
            )
        resolved_checkpoint = _resolve_checkpoint_path(method, checkpoint_value)
        mesh = load_mesh_splatting_checkpoint(resolved_checkpoint)

    resolved_method = adapter_cfg.get("type")
    if resolved_method is not None:
        resolved_method = _canonical_method(str(resolved_method))
    elif selected_source == "checkpoint":
        resolved_method = "mesh-splatting"

    metadata = {
        "method": resolved_method,
        "dataset": dataset_cfg.get("name") or dataset_cfg.get("type"),
        "scene": dataset_cfg.get("scene"),
        "config": str(config) if config is not None else None,
    }

    output_path: Path | None
    if output:
        output_path = Path(output).expanduser()
    elif connectivity_cfg.get("output") is not None:
        output_path = Path(str(connectivity_cfg["output"])).expanduser()
    elif cfg is not None and output_dir(cfg) is not None:
        output_path = Path(str(output_dir(cfg))).expanduser() / "connectivity_metrics.json"
    else:
        output_path = None

    return mesh, metadata, output_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate boundary-edge, non-manifold, and connected-component "
            "metrics for a MeshSplatting triangular mesh."
        )
    )
    parser.add_argument("--config", "-c", type=str, help="TriBench MeshSplatting config YAML.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Override config adapter.checkpoint or provide a standalone MeshSplatting checkpoint.",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        help="Analyze an exported triangular mesh file instead of the MeshSplatting checkpoint.",
    )
    parser.add_argument(
        "--source",
        choices=("checkpoint", "mesh-file"),
        help="Geometry source. Default: checkpoint, or mesh-file when --mesh-file is set.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Metrics JSON output path. Default with --config: <output.dir>/connectivity_metrics.json.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print metrics JSON to stdout.",
    )
    parser.add_argument(
        "--skip-vertex-check",
        action="store_true",
        help="Skip the more expensive non-manifold vertex one-ring check.",
    )
    parser.add_argument(
        "--top-components",
        type=int,
        default=10,
        help="Number of largest connected components to include in details.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    args = parser.parse_args()

    if args.config is None and args.checkpoint is None and args.mesh_file is None:
        parser.error("Provide --config, --checkpoint, or --mesh-file.")
    if args.config is None and args.checkpoint is not None and args.source == "mesh-file":
        parser.error("--source=mesh-file requires --mesh-file.")

    mesh, metadata, output_path = _resolve_from_config(
        args.config,
        checkpoint=args.checkpoint,
        mesh_file=args.mesh_file,
        source=args.source,
        output=args.output,
    )
    result = calculate_connectivity_metrics(
        mesh,
        config_path=metadata.get("config"),
        method=metadata.get("method"),
        dataset=metadata.get("dataset"),
        scene=metadata.get("scene"),
        include_vertex_manifoldness=not args.skip_vertex_check,
        top_components=args.top_components,
    )

    text = json.dumps(result, indent=args.indent, default=_json_default)
    if output_path is not None and not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"[calculate_connectivity] saved: {output_path}")
    print(text)


if __name__ == "__main__":
    main()
