"""Tests for Triangle Splatting training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tribench.trainers.triangle_splatting_method import TriangleSplattingTrainingMethod


class _FakeTriangleModel:
    def __init__(self, num_triangles: int) -> None:
        self._triangles = torch.zeros(num_triangles, 3, 3)
        self.importance_score = torch.ones(num_triangles)
        self._opacity = torch.ones(num_triangles, 1)
        self.triangle_area = torch.ones(num_triangles)
        self.image_size = torch.zeros(num_triangles)
        self.optimizer = object()
        self.add_calls = 0
        self.prune_calls = 0

    @property
    def get_triangles_points(self):
        return self._triangles

    @property
    def get_opacity(self):
        return self._opacity

    def add_new_gs(self, *, cap_max, oddGroup=True, dead_mask=None):
        self.add_calls += 1

    def remove_final_points(self, mask):
        self.prune_calls += 1


def _training_method_with_fake_model(num_triangles: int, *, max_shapes: int):
    method = TriangleSplattingTrainingMethod.__new__(TriangleSplattingTrainingMethod)
    method._ensure_initialized = lambda: None
    method._model = _FakeTriangleModel(num_triangles)
    method._optimizer = method._model.optimizer
    method._train_cameras = [object()] * 10
    method._new_round = False
    method._removed_them = False
    method._opacity_now = True
    method._last_structure_update = None
    method._opt = SimpleNamespace(
        max_shapes=max_shapes,
        densify_until_iter=15_000,
        densification_interval=500,
        densify_from_iter=500,
        importance_threshold=0.022,
        opacity_dead=0.014,
        outdoor=False,
        proba_distr=2,
    )
    return method


def test_update_structure_skips_densify_at_triangle_cap():
    method = _training_method_with_fake_model(10, max_shapes=10)

    update = method.update_structure(1_000)

    assert update is None
    assert method._model.add_calls == 0
    assert method._model.prune_calls == 0


def test_update_structure_runs_densify_below_triangle_cap():
    method = _training_method_with_fake_model(9, max_shapes=10)

    update = method.update_structure(1_000)

    assert update["type"] == "densify"
    assert method._model.add_calls == 1
    assert method._model.prune_calls == 0
