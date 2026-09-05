"""Core abstractions for MeshSplatBench."""

__all__ = [
    "CameraBatch",
    "Config",
    "ModelStats",
    "build_adapter",
    "build_dataset",
    "build_from_cfg",
    "build_training_loop",
    "build_training_method",
    "register",
    "get_adapter",
    "load_config",
    "list_methods",
]


def __getattr__(name: str):
    if name == "CameraBatch":
        from msbench.core.cameras import CameraBatch

        return CameraBatch
    if name == "Config":
        from msbench.core.config import Config

        return Config
    if name == "ModelStats":
        from msbench.core.stats import ModelStats

        return ModelStats
    if name == "build_adapter":
        from msbench.core.builder import build_adapter

        return build_adapter
    if name == "build_dataset":
        from msbench.core.builder import build_dataset

        return build_dataset
    if name == "build_from_cfg":
        from msbench.core.builder import build_from_cfg

        return build_from_cfg
    if name == "build_training_loop":
        from msbench.core.builder import build_training_loop

        return build_training_loop
    if name == "build_training_method":
        from msbench.core.builder import build_training_method

        return build_training_method
    if name == "register":
        from msbench.core.registry import register

        return register
    if name == "get_adapter":
        from msbench.core.registry import get_adapter

        return get_adapter
    if name == "load_config":
        from msbench.core.config import load_config

        return load_config
    if name == "list_methods":
        from msbench.core.registry import list_methods

        return list_methods
    raise AttributeError(name)
