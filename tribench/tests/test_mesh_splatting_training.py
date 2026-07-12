"""Tests for Mesh Splatting training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tribench.trainers.mesh_splatting_method import MeshSplattingTrainingMethod


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


def _method_with_fake_mesh(num_triangles: int, *, max_primitives: int | None):
    method = MeshSplattingTrainingMethod.__new__(MeshSplattingTrainingMethod)
    method._ensure_initialized = lambda: None
    method._model = _FakeMeshModel(num_triangles)
    method._optimizer = method._model.optimizer
    method._train_cameras = [object()] * 10
    method._need_delaunay = False
    method._prune_threshold = 0.0
    method._opt = SimpleNamespace(
        max_points=1_000_000,
        max_primitives=max_primitives,
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


def test_mesh_update_structure_skips_densify_at_primitive_cap():
    method = _method_with_fake_mesh(10, max_primitives=10)

    update = method.update_structure(1_000)

    assert update is None
    assert method._model.add_calls == []


def test_mesh_update_structure_limits_split_candidates_by_primitive_budget():
    method = _method_with_fake_mesh(10, max_primitives=13)

    update = method.update_structure(1_000)

    assert update["type"] == "densify"
    assert method._model.add_calls == [1]
