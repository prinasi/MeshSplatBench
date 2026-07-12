"""Tests for Triangle Splatting training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tribench.trainers.triangle_splatting_method import TriangleSplattingTrainingMethod
from tribench.vendor.triangle_splatting.scene.triangle_model import TriangleModel


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


def test_triangle_model_reset_training_statistics_initializes_tensor_buffers():
    model = TriangleModel.__new__(TriangleModel)
    model._triangles_points = torch.zeros(7, 3, 3)
    model.triangle_area = 0
    model.image_size = 0
    model.importance_score = 0

    model.reset_training_statistics()

    assert torch.equal(model.triangle_area, torch.zeros(7))
    assert torch.equal(model.image_size, torch.zeros(7))
    assert torch.equal(model.importance_score, torch.zeros(7))


def test_triangle_model_enforce_max_primitives_prunes_optimizer_state():
    model = TriangleModel.__new__(TriangleModel)
    model._triangles_points = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).view(4, 3, 3))
    model._features_dc = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model._features_rest = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model._opacity = torch.nn.Parameter(torch.zeros(4, 1))
    model._sigma = torch.nn.Parameter(torch.zeros(4, 1))
    model._mask = torch.nn.Parameter(torch.ones(4, 1))
    model.triangle_area = torch.ones(4)
    model.image_size = torch.zeros(4)
    model.importance_score = torch.tensor([0.0, 10.0, 1.0, 5.0])
    model.max_scaling = torch.zeros(4)
    model.max_radii2D = torch.zeros(4)
    model.max_density_factor = torch.zeros(4)
    model.denom = torch.zeros(4, 1)
    model.opacity_activation = torch.sigmoid
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [model._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [model._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [model._triangles_points], "lr": 0.0, "name": "triangles_points"},
            {"params": [model._sigma], "lr": 0.0, "name": "sigma"},
            {"params": [model._mask], "lr": 0.0, "name": "mask"},
        ],
        lr=0.0,
    )

    removed = model.enforce_max_primitives(2)

    assert removed == 2
    assert model.get_triangles_points.shape[0] == 2
    assert torch.equal(model.get_triangles_points[:, 0, 0], torch.tensor([9.0, 27.0]))
    assert model.get_num_points_per_triangle.tolist() == [3, 3]
    assert model.get_number_of_points == 2


def test_training_method_dataset_args_include_dtu_eval_mode(tmp_path):
    method = TriangleSplattingTrainingMethod(
        dataset=tmp_path / "scan24",
        output_dir=tmp_path / "out",
        white_background=True,
        dtu_eval_mode="foreground",
    )

    args = method._build_dataset_args()

    assert args.white_background is True
    assert args.dtu_eval_mode == "foreground"


def test_training_method_accepts_dtu_eval_mode_from_extra_args(tmp_path):
    method = TriangleSplattingTrainingMethod(
        dataset=tmp_path / "scan24",
        output_dir=tmp_path / "out",
        extra_args={"dtu_eval_mode": "foreground"},
    )

    assert method._build_dataset_args().dtu_eval_mode == "foreground"
