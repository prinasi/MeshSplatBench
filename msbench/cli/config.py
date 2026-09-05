"""Shared helpers for config-driven CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from msbench.core.config import Config, merge_dicts, resolve_dataset_config


def load_cli_config(path: str | Path | None) -> Config | None:
    """Load a MeshSplatBench CLI config when one was provided."""
    if path is None:
        return None
    return Config.fromfile(path)


def section(cfg: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    """Return a config section as a plain dict."""
    if cfg is None:
        return {}
    return Config(cfg.get(name, {})).to_dict()


def merged_section(
    cfg: Mapping[str, Any] | None,
    name: str,
    *,
    nested: str | None = None,
) -> dict[str, Any]:
    """Return a section optionally merged with one nested sub-section."""
    data = section(cfg, name)
    if nested is None:
        return data
    nested_data = data.pop(nested, {}) or {}
    if not isinstance(nested_data, Mapping):
        return data
    return merge_dicts(data, nested_data)


def output_dir(cfg: Mapping[str, Any] | None, default: str | Path | None = None) -> str | None:
    """Resolve the run output directory from config."""
    output_cfg = section(cfg, "output")
    value = output_cfg.get("dir", default)
    return str(value) if value is not None else None


def metrics_file(cfg: Mapping[str, Any] | None, default: str = "metrics.json") -> str:
    """Resolve the image-metrics output path."""
    eval_cfg = section(cfg, "eval")
    output_cfg = section(cfg, "output")
    value = eval_cfg.get("output") or eval_cfg.get("metrics_file") or output_cfg.get("metrics_file")
    if value is None:
        out_dir = output_cfg.get("dir")
        value = Path(out_dir) / default if out_dir else default
    return str(value)


def adapter_config(
    cfg: Mapping[str, Any] | None,
    *,
    method: str | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Resolve adapter config, falling back to trainer/output for train configs."""
    if cfg is None:
        if method is None:
            raise ValueError("Adapter method is required.")
        data: dict[str, Any] = {"type": method}
        if checkpoint is not None:
            data["checkpoint"] = checkpoint
        return data

    data = section(cfg, "adapter")
    trainer_cfg = section(cfg, "trainer")
    out_dir = output_dir(cfg)
    if not data:
        data = {}
    if method is not None:
        data["type"] = method
    else:
        data.setdefault("type", trainer_cfg.get("type") or trainer_cfg.get("name"))
    if checkpoint is not None:
        data["checkpoint"] = checkpoint
    else:
        data.setdefault("checkpoint", out_dir)
    return data


def dataset_config(
    cfg: Mapping[str, Any] | None,
    *,
    dataset: str | None = None,
    dataset_type: str | None = None,
    split: str | None = None,
    image_dir: str | None = None,
    resolution: int | None = None,
    eval_every: int | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Resolve dataset config for render/eval/profile style commands."""
    data = section(cfg, "dataset")
    stage_cfg = section(cfg, stage) if stage else {}

    if dataset is not None:
        data["root"] = dataset
    if dataset_type is not None:
        data["type"] = dataset_type
    if image_dir is not None:
        data["image_dir"] = image_dir
    if resolution is not None:
        data["resolution"] = resolution
    if eval_every is not None:
        data["eval_every"] = eval_every
    data["split"] = stage_cfg.get("split", split if split is not None else data.get("split", "test"))

    data = resolve_dataset_config(data)

    # Training configs may carry train-only keys; dataset builders should not
    # receive them for evaluation/rendering.
    for key in ("eval_split", "llffhold", "scene", "name"):
        data.pop(key, None)
    return data
