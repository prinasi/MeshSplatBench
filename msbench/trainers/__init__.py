"""Training infrastructure for MeshSplatBench."""

from msbench.trainers.hooks import TrainingMethod
from msbench.trainers.loop import TrainingLoop, TrainingConfig
from msbench.trainers.losses import L1Loss, SSIMLoss, LPIPSLoss, CombinedLoss
from msbench.trainers.registry import (
    register_training_method,
    get_training_method,
    list_training_methods,
)

__all__ = [
    "TrainingMethod",
    "TrainingLoop",
    "TrainingConfig",
    "L1Loss",
    "SSIMLoss",
    "LPIPSLoss",
    "CombinedLoss",
    "register_training_method",
    "get_training_method",
    "list_training_methods",
]
