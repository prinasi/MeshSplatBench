"""Tests for Mesh Splatting training control logic."""

from __future__ import annotations

from tribench.trainers.mesh_splatting_method import MeshSplattingTrainingMethod


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
