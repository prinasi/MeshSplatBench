"""Tests for the generic training loop."""

from __future__ import annotations

import torch

from tribench.trainers.hooks import TrainingMethod
from tribench.trainers.loop import TrainingConfig, TrainingLoop


class _StepRecordingMethod(TrainingMethod):
    def __init__(self) -> None:
        self.seen_steps: list[int] = []

    def sample_batch(self):
        self.seen_steps.append(self._step)
        return None, torch.tensor([0.0])

    def render_train(self, cameras):
        return None

    def compute_loss(self, output, gt_images):
        return {"total": torch.tensor(1.0, requires_grad=True)}

    def optimizer_step(self) -> None:
        pass

    def update_structure(self, step: int):
        return None


def test_training_loop_synchronizes_method_step(tmp_path):
    method = _StepRecordingMethod()
    loop = TrainingLoop(
        method,
        TrainingConfig(max_steps=3, eval_interval=99, output_dir=str(tmp_path)),
    )

    loop.train()

    assert method.seen_steps == [1, 2, 3]
