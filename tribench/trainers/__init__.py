"""Training infrastructure for TriBench."""

from tribench.trainers.hooks import TrainingMethod
from tribench.trainers.loop import TrainingLoop, TrainingConfig
from tribench.trainers.losses import L1Loss, SSIMLoss, LPIPSLoss, CombinedLoss
from tribench.trainers.registry import (
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
