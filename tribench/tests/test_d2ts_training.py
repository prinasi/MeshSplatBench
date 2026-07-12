"""Tests for 2DTS training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_2dts_model_enforce_max_primitives_prunes_optimizer_state():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")

    from tribench.vendor.d2ts.diff_recon.models.VanillaTS_model import VanillaTSModel

    model = VanillaTSModel.__new__(VanillaTSModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_update=SimpleNamespace(max_primitives=2))
    model.use_vertex_color = False
    model._vertex = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).view(4, 3, 3))
    model._opacity = torch.nn.Parameter(torch.zeros(4, 1))
    model._f_dc = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model._f_rest = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model.gradient_accum = torch.zeros(4)
    model.gradient_denom = torch.zeros(4)
    model.max_radii2D = torch.zeros(4)
    model.contrib_sum = torch.zeros(4)
    model.contrib_max = torch.tensor([0.0, 10.0, 1.0, 5.0])
    model.contrib_denom = torch.zeros(4)
    model.opacity_floor = 0.0
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model._vertex], "lr": 0.0, "name": "vertex"},
            {"params": [model._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [model._f_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [model._f_rest], "lr": 0.0, "name": "f_rest"},
        ],
        lr=0.0,
    )

    removed = model.enforce_max_primitives()

    assert removed == 2
    assert model.get_vertex.shape[0] == 2
    assert torch.equal(model.get_vertex[:, 0, 0], torch.tensor([9.0, 27.0]))
    assert model.contrib_max.tolist() == [10.0, 5.0]
