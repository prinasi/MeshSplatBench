"""Config-driven builders for TriBench components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from tribench.core.config import Config, resolve_dataset_config

T = TypeVar("T")


def build_from_cfg(
    cfg: Mapping[str, Any],
    registry: Mapping[str, type[T]] | Callable[[str], type[T]],
    *,
    default_args: Mapping[str, Any] | None = None,
) -> T:
    """Instantiate an object from a config mapping.

    The config must contain a ``type`` key. Remaining keys are passed as keyword
    arguments, with an optional nested ``params`` mapping merged in. This mirrors
    the MMDetection habit of declarative component construction while keeping
    the surface area small.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError(f"cfg must be a mapping, got {type(cfg).__name__}")

    plain_cfg = Config(cfg).to_dict()
    if "type" in plain_cfg:
        obj_type = plain_cfg.pop("type")
    else:
        obj_type = plain_cfg.pop("name", None)
    if obj_type is None:
        raise KeyError("Config section must include a 'type' key.")

    params = plain_cfg.pop("params", {}) or {}
    if not isinstance(params, Mapping):
        raise TypeError("'params' must be a mapping when provided.")

    kwargs = dict(default_args or {})
    kwargs.update(params)
    kwargs.update(plain_cfg)
    cls = _lookup(obj_type, registry)
    return cls(**kwargs)


def build_adapter(cfg: Mapping[str, Any]):
    """Build and optionally load a renderer adapter from config.

    Example::

        adapter:
          type: triangle-splatting
          checkpoint: outputs/model
          render_params:
            gamma_rescale: true
            ste_threshold: 0.3
            sort_level: 2
    """
    from tribench.core.registry import get_adapter

    plain_cfg = Config(cfg).to_dict()
    checkpoint = plain_cfg.pop("checkpoint", None)
    render_params = plain_cfg.pop("render_params", None)
    adapter = build_from_cfg(plain_cfg, get_adapter)
    if render_params and hasattr(adapter, "configure"):
        adapter.configure(**render_params)
    if checkpoint is not None:
        adapter.load_checkpoint(str(Path(checkpoint).expanduser()))
    return adapter


def build_dataset(cfg: Mapping[str, Any]):
    """Build a dataset from a config mapping."""
    from tribench.core.datasets import load_dataset

    plain_cfg = resolve_dataset_config(Config(cfg).to_dict())
    dataset_type = plain_cfg.pop("type", plain_cfg.pop("dataset_type", "auto"))
    root = plain_cfg.pop("root", plain_cfg.pop("dataset_path", None))
    plain_cfg.pop("eval_split", None)
    plain_cfg.pop("llffhold", None)
    plain_cfg.pop("scene", None)
    plain_cfg.pop("name", None)
    if root is None:
        raise KeyError("Dataset config must include 'root' or 'dataset_path'.")
    return load_dataset(root, dataset_type=dataset_type, **plain_cfg)


def build_training_method(cfg: Mapping[str, Any], *, default_args: Mapping[str, Any] | None = None):
    """Build a ``TrainingMethod`` from config."""
    import inspect

    from tribench.trainers.registry import get_training_method

    plain_cfg = Config(cfg).to_dict()
    obj_type = plain_cfg.get("type", plain_cfg.get("name"))
    if obj_type is None:
        raise KeyError("Training method config must include a 'type' key.")
    method_cls = get_training_method(obj_type)
    known_params = set(inspect.signature(method_cls.__init__).parameters) - {"self"}
    if "extra_args" not in known_params:
        return build_from_cfg(plain_cfg, get_training_method, default_args=default_args)

    params = dict(default_args or {})
    params.update(plain_cfg.get("params", {}) or {})
    params.update({k: v for k, v in plain_cfg.items() if k not in {"type", "name", "params"}})
    init_kwargs: dict[str, Any] = {}
    extra_args: dict[str, Any] = {}
    for key, value in params.items():
        if key in known_params:
            init_kwargs[key] = value
        else:
            extra_args[key] = value
    if extra_args:
        init_kwargs["extra_args"] = extra_args
    return method_cls(**init_kwargs)


def build_training_loop(
    cfg: Mapping[str, Any],
    method,
    *,
    output_dir: str | Path | None = None,
):
    """Build a ``TrainingLoop`` and its ``TrainingConfig`` from config."""
    from tribench.trainers.loop import TrainingConfig, TrainingLoop

    plain_cfg = Config(cfg).to_dict()
    if output_dir is not None:
        plain_cfg.setdefault("output_dir", str(output_dir))
    loop_config = TrainingConfig(**plain_cfg)
    return TrainingLoop(method, loop_config)


def _lookup(name: str, registry: Mapping[str, type[T]] | Callable[[str], type[T]]) -> type[T]:
    if callable(registry):
        return registry(name)
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Component '{name}' not found. Available: {available}") from exc
