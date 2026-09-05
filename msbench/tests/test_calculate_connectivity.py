from __future__ import annotations

import numpy as np

from tools.calculate_connectivity import MeshData, calculate_connectivity_metrics


def _mesh(vertices: list[list[float]], faces: list[list[int]]) -> MeshData:
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        source_type="test",
        source_path=__file__,
        source_format="memory",
    )


def test_square_patch_has_boundary_but_manifold_vertices():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["counts"]["edges"] == 5
    assert result["counts"]["boundary_edges"] == 4
    assert result["counts"]["non_manifold_edges"] == 0
    assert result["counts"]["non_manifold_vertices"] == 0
    assert result["components"]["count"] == 1
    assert result["metrics"]["largest_component_ratio"] == 1.0
    assert result["quality_flags"]["edge_manifold"] is True
    assert result["quality_flags"]["closed_edge_manifold"] is False


def test_three_faces_on_one_edge_are_non_manifold():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["counts"]["non_manifold_edges"] == 1
    assert result["counts"]["non_manifold_vertices"] == 2
    assert result["edges"]["max_face_incidence"] == 3
    assert result["quality_flags"]["edge_manifold"] is False
    assert result["quality_flags"]["vertex_manifold"] is False


def test_largest_component_ratio_reports_fragmentation():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["components"]["count"] == 2
    assert result["components"]["single_face_component_count"] == 2
    assert result["metrics"]["largest_component_ratio"] == 0.5
