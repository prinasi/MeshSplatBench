"""Core abstractions for TriangleBench."""

from trianglebench.core.cameras import CameraBatch
from trianglebench.core.stats import ModelStats
from trianglebench.core.registry import register, get_adapter, list_methods

__all__ = [
    "CameraBatch",
    "ModelStats",
    "register",
    "get_adapter",
    "list_methods",
]
