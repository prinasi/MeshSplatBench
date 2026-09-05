"""Tests for the generic training loop."""

from __future__ import annotations

import torch

from msbench.trainers.checkpoints import find_latest_point_cloud_checkpoint
from msbench.trainers.hooks import TrainingMethod
from msbench.trainers.loop import TrainingConfig, TrainingLoop


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


class _ResumingStepRecordingMethod(_StepRecordingMethod):
    def __init__(self, start_step: int) -> None:
        super().__init__()
        self.start_step = start_step

    def resume_from_checkpoint(self, output_dir, max_steps: int) -> int:
        return self.start_step


def test_training_loop_synchronizes_method_step(tmp_path):
    method = _StepRecordingMethod()
    loop = TrainingLoop(
        method,
        TrainingConfig(max_steps=3, eval_interval=99, output_dir=str(tmp_path)),
    )

    loop.train()

    assert method.seen_steps == [1, 2, 3]


def test_find_latest_point_cloud_checkpoint_ignores_incomplete_dirs(tmp_path):
    complete = tmp_path / "point_cloud" / "iteration_1000"
    complete.mkdir(parents=True)
    (complete / "point_cloud_state_dict.pt").write_bytes(b"checkpoint")
    incomplete = tmp_path / "point_cloud" / "iteration_2000"
    incomplete.mkdir()

    latest = find_latest_point_cloud_checkpoint(tmp_path)

    assert latest == (1000, complete)


def test_training_loop_resumes_after_completed_step(tmp_path):
    method = _ResumingStepRecordingMethod(start_step=2)
    loop = TrainingLoop(
        method,
        TrainingConfig(max_steps=4, eval_interval=99, output_dir=str(tmp_path)),
    )

    summary = loop.train()

    assert method.seen_steps == [3, 4]
    assert summary["start_step"] == 2
    assert summary["trained_steps"] == 2
