"""MeshSplatBench: a config-driven benchmark for triangle/splatting reconstruction."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CameraBatch",
    "Config",
    "load_config",
    "ModelStats",
]


def __getattr__(name: str):
    if name == "CameraBatch":
        from msbench.core.cameras import CameraBatch

        return CameraBatch
    if name == "Config":
        from msbench.core.config import Config

        return Config
    if name == "load_config":
        from msbench.core.config import load_config

        return load_config
    if name == "ModelStats":
        from msbench.core.stats import ModelStats

        return ModelStats
    raise AttributeError(name)
