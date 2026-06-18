"""Training infrastructure for TriangleBench."""

from trianglebench.trainers.hooks import TrainingMethod
from trianglebench.trainers.loop import TrainingLoop, TrainingConfig
from trianglebench.trainers.losses import L1Loss, SSIMLoss, LPIPSLoss, CombinedLoss
from trianglebench.trainers.registry import (
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
