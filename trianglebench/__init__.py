"""TriangleBench: A toolkit for triangle-splatting-style radiance field research."""

__version__ = "0.1.0"

from trianglebench.core.cameras import CameraBatch
from trianglebench.core.stats import ModelStats

__all__ = [
    "__version__",
    "CameraBatch",
    "ModelStats",
]
