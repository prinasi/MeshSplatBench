"""Tests for Mesh Splatting training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from msbench.trainers.mesh_splatting_method import MeshSplattingTrainingMethod


class _FakeMeshModel:
    def __init__(self, num_triangles: int) -> None:
        self._triangle_indices = torch.arange(num_triangles * 3).view(num_triangles, 3)
        self.vertices = torch.zeros(num_triangles * 3, 3)
        self.vertex_weight = torch.ones(num_triangles * 3, 1)
        self.image_size = torch.zeros(num_triangles)
        self.importance_score = torch.ones(num_triangles)
        self.optimizer = object()
        self.add_calls: list[int | None] = []

    def opacity_activation(self, value):
        return value

    @property
    def get_vertex_weight(self):
        return self.vertex_weight

    def prune_triangles(self, mask):
        pass

    def _prune_vertices(self, mask):
        pass

    def add_new_gs(self, step, *, cap_max, splitt_large_triangles, max_split_candidates=None):
        self.add_calls.append(max_split_candidates)

    def enforce_max_primitives(self, max_primitives):
        return 0


def _method_with_fake_mesh(num_triangles: int, *, max_points: int = 1_000_000):
    method = MeshSplattingTrainingMethod.__new__(MeshSplattingTrainingMethod)
    method._ensure_initialized = lambda: None
    method._model = _FakeMeshModel(num_triangles)
    method._optimizer = method._model.optimizer
    method._train_cameras = [object()] * 10
    method._need_delaunay = False
    method._prune_threshold = 0.0
    method._opt = SimpleNamespace(
        max_points=max_points,
        run_restricted_delaunay=10_000,
        densify_until_iter=5_000,
        densification_interval=500,
        densify_from_iter=500,
        start_pruning=10_000,
        prune_size=1_400,
        splitt_large_triangles=100,
        start_opacity_floor=10_000,
        final_opacity_iter=24_000,
    )
    return method


def test_mesh_training_method_dataset_args_include_dtu_eval_mode(tmp_path):
    method = MeshSplattingTrainingMethod(
        dataset=tmp_path / "scan24",
        output_dir=tmp_path / "out",
        white_background=True,
        dtu_eval_mode="foreground",
    )

    args = method._build_dataset_args()

    assert args.white_background is True
    assert args.dtu_eval_mode == "foreground"


def test_mesh_training_method_accepts_dtu_eval_mode_from_extra_args(tmp_path):
    method = MeshSplattingTrainingMethod(
        dataset=tmp_path / "scan24",
        output_dir=tmp_path / "out",
        extra_args={"dtu_eval_mode": "foreground"},
    )

    assert method._build_dataset_args().dtu_eval_mode == "foreground"


def test_mesh_update_structure_skips_densify_at_vertex_cap():
    """Densification is skipped when vertices >= max_points (official vertex safeguard)."""
    method = _method_with_fake_mesh(10, max_points=30)

    update = method.update_structure(1_000)

    assert update is None
    assert method._model.add_calls == []


def test_mesh_update_structure_densifies_when_under_vertex_cap():
    """Densification proceeds when vertices < max_points."""
    method = _method_with_fake_mesh(10, max_points=1_000_000)

    update = method.update_structure(1_000)

    assert update["type"] == "densify"
    assert method._model.add_calls == [None]


def test_mesh_model_enforce_max_primitives_prunes_faces_and_unused_vertices():
    from msbench.vendor.mesh_splatting.scene.triangle_model import TriangleModel

    model = TriangleModel.__new__(TriangleModel)
    model._triangle_indices = torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ], dtype=torch.int32)
    model.vertices = torch.randn(6, 3)
    model.vertex_weight = torch.ones(6, 1)
    model._features_dc = torch.zeros(6, 1, 3)
    model._features_rest = torch.zeros(6, 15, 3)
    model.importance_score = torch.tensor([1.0, 10.0, 5.0, 0.5])
    model.image_size = torch.zeros(4)
    model.pixel_count = torch.zeros(4)
    model.optimizer = torch.optim.Adam([
        {"params": [model.vertices], "name": "vertices"},
        {"params": [model.vertex_weight], "name": "vertex_weight"},
        {"params": [model._features_dc], "name": "f_dc"},
        {"params": [model._features_rest], "name": "f_rest"},
    ])

    removed = model.enforce_max_primitives(2)
    assert removed == 2
    assert model._triangle_indices.shape[0] == 2
    assert model.vertices.shape[0] == 4
