"""Configuration loading utilities for TriBench.

TriBench configs are plain YAML mappings with optional ``_base_`` inheritance.
They are intentionally small compared with MMEngine/MMDetection configs, but
keep the same useful shape: declare components in YAML and build them through
registries.
"""

from __future__ import annotations

from copy import deepcopy
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised in minimal environments.
    _yaml = None


class Config(dict):
    """Dictionary-backed config with attribute access.

    Nested mappings are recursively wrapped as ``Config`` objects. The class is
    deliberately lightweight so it can be passed anywhere a normal ``dict`` is
    expected.
    """

    @classmethod
    def fromfile(cls, filename: str | Path) -> "Config":
        """Load a YAML config file, resolving optional ``_base_`` entries."""
        return cls(load_config(filename))

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        payload = dict(data or {})
        payload.update(kwargs)
        for key, value in payload.items():
            self[key] = self._wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return this config as plain Python containers."""
        return _to_plain_dict(self)

    def dump(self, filename: str | Path) -> None:
        """Write this config as YAML."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_safe_dump(self.to_dict()))


def load_config(filename: str | Path) -> dict[str, Any]:
    """Load a YAML config file with recursive ``_base_`` inheritance.

    ``_base_`` can be a string path or a list of paths. Relative base paths are
    resolved from the child config's directory. Child keys recursively override
    base keys.
    """
    path = Path(filename).expanduser().resolve()
    return finalize_config(_load_config(path))


def _load_config(path: Path) -> dict[str, Any]:
    """Load and merge config files without applying final conveniences."""
    data = _read_yaml_mapping(path)
    base_entry = data.pop("_base_", None)

    if base_entry is None:
        return data

    base_paths = base_entry if isinstance(base_entry, list) else [base_entry]
    merged: dict[str, Any] = {}
    for base in base_paths:
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, _load_config(base_path.resolve()))
    return merge_dicts(merged, data)


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings and return a new plain dict."""
    result = deepcopy(_to_plain_dict(base))
    for key, value in override.items():
        plain_value = _to_plain_dict(value)
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(plain_value, dict)
        ):
            result[key] = merge_dicts(result[key], plain_value)
        else:
            result[key] = deepcopy(plain_value)
    return result


def apply_overrides(config: Mapping[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dot-list overrides such as ``trainer.max_steps=1000``."""
    result = deepcopy(_to_plain_dict(config))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = target.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set nested key below non-mapping: {part}")
            target = node
        target[parts[-1]] = _coerce_scalar(raw_value)
    return result


def finalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve inherited config conveniences.

    Finalization currently supports two small conventions:

    - Scene-specific triangle caps via ``triangle_caps``.
    - String templates such as ``outputs/{method}/{dataset}/{scene}``.
    - Dataset roots split into ``dataset.root`` plus ``dataset.scene``.
    """
    resolved = _apply_scene_triangle_caps(_to_plain_dict(config))
    resolved = _apply_max_primitive_limit(resolved)
    resolved = _apply_dtu_defaults(resolved)
    resolved = _format_config_templates(resolved)
    resolved = _resolve_dataset_scene(resolved)
    return resolved


def resolve_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return one dataset config with ``root`` resolved against ``scene``."""
    data = {"dataset": _to_plain_dict(config)}
    return finalize_config(data)["dataset"]


def save_config_snapshot(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Save the fully resolved config to ``output_dir/config.yaml``."""
    path = Path(output_dir).expanduser() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_dump(_to_plain_dict(config)))
    return path


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_config_templates(config: dict[str, Any]) -> dict[str, Any]:
    context = _format_context(config)
    return _format_value(config, context)


def _apply_scene_triangle_caps(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        return config

    scene = dataset_cfg.get("scene")
    if scene is None:
        return config

    cap_maps = []
    for key in ("triangle_caps", "triangle_limits"):
        value = config.get(key)
        if isinstance(value, Mapping):
            cap_maps.append(value)

    cap = None
    for cap_map in cap_maps:
        if scene in cap_map:
            cap = cap_map[scene]
            break
        scene_name = Path(str(scene)).name
        if scene_name in cap_map:
            cap = cap_map[scene_name]
            break
        for pattern, value in cap_map.items():
            pattern_text = str(pattern)
            if fnmatchcase(str(scene), pattern_text) or fnmatchcase(scene_name, pattern_text):
                cap = value
                break
        if cap is not None:
            break
    if cap is None:
        return config

    trainer_cfg = dict(config.get("trainer", {}) or {})
    trainer_cfg.setdefault("max_shapes", cap)
    config["trainer"] = trainer_cfg
    return config


def _apply_max_primitive_limit(config: dict[str, Any]) -> dict[str, Any]:
    trainer_cfg = config.get("trainer", {})
    if not isinstance(trainer_cfg, Mapping):
        return config

    trainer = dict(trainer_cfg)
    method_type = str(trainer.get("type", trainer.get("name", ""))).lower().replace("_", "-")
    method_key = method_type.replace("-", "_")

    # 1. Look for method-specific cap first
    cap_value = None
    candidate_keys = [
        f"max_primitives_{method_key}",
        f"max_primitives_{method_type}",
        f"max_shapes_{method_key}",
        f"max_shapes_{method_type}",
    ]
    for key in candidate_keys:
        if key in trainer and trainer[key] is not None:
            cap_value = trainer[key]
            break

    if cap_value is None:
        aliases = [method_key, method_type]
        if method_type in {"2dts", "d2ts"}:
            aliases.extend(["2dts", "d2ts"])
        elif method_type in {"triangle-splatting", "triangle_splatting"}:
            aliases.extend(["triangle_splatting", "triangle-splatting"])
        elif method_type in {"mesh-splatting", "mesh_splatting"}:
            aliases.extend(["mesh_splatting", "mesh-splatting"])
        for alias in aliases:
            block = config.get(alias)
            if isinstance(block, Mapping):
                val = block.get("max_primitives", block.get("max_shapes"))
                if val is not None:
                    cap_value = val
                    break

    if cap_value is None:
        for key in candidate_keys:
            if key in config and config[key] is not None:
                cap_value = config[key]
                break

    # 2. Fall back to generic max_primitives (mesh-splatting is excluded;
    #    it uses only the native max_points vertex safeguard).
    if cap_value is None:
        if method_type in {"mesh-splatting", "mesh_splatting"}:
            return config
        cap_value = trainer.get("max_primitives", config.get("max_primitives"))

    if cap_value is None:
        return config

    try:
        cap = int(cap_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trainer.max_primitives must be an integer, got {cap_value!r}") from exc
    if cap <= 0:
        raise ValueError(f"trainer.max_primitives must be positive, got {cap!r}")

    trainer["max_primitives"] = cap

    if method_type in {"triangle-splatting", "triangle_splatting"}:
        existing = trainer.get("max_shapes")
        trainer["max_shapes"] = cap if existing is None else min(int(existing), cap)
    elif method_type in {"2dts", "d2ts"}:
        d2ts = dict(config.get("d2ts", {}) or {})
        existing = d2ts.get("target_point_num")
        d2ts["target_point_num"] = cap if existing is None else min(int(existing), cap)
        config["d2ts"] = d2ts
    elif method_type == "diffsoup":
        for key in ("n_points", "target_prims"):
            existing = trainer.get(key)
            trainer[key] = cap if existing is None else min(int(existing), cap)

    config["trainer"] = trainer

    if cap > 0 and method_type in {"triangle-splatting", "2dts", "d2ts"}:
        canonical_name = "2dts" if method_type in {"2dts", "d2ts"} else method_type
        old_prefix = f"outputs/{canonical_name}"
        new_prefix = f"outputs/abl/{canonical_name}-{cap}"
        config = _retarget_ablation_paths(config, old_prefix, new_prefix)

    return config


def _retarget_ablation_paths(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, str):
        if value == old_prefix or value.startswith(old_prefix + "/"):
            return new_prefix + value[len(old_prefix):]
        return value
    if isinstance(value, Mapping):
        return {k: _retarget_ablation_paths(v, old_prefix, new_prefix) for k, v in value.items()}
    if isinstance(value, list):
        return [_retarget_ablation_paths(item, old_prefix, new_prefix) for item in value]
    return value


def _apply_dtu_defaults(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        return config
    dataset_type = str(dataset_cfg.get("type", "")).lower()
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    if dataset_type != "dtu" and dataset_name != "dtu":
        return config
    dataset = dict(dataset_cfg)
    mode = str(dataset.get("dtu_eval_mode", "full")).lower()
    if mode == "fg":
        mode = "foreground"
    if mode not in {"full", "foreground"}:
        raise ValueError(f"dtu_eval_mode must be 'full' or 'foreground', got {mode!r}")
    dataset["dtu_eval_mode"] = mode
    config["dataset"] = dataset

    trainer_cfg = config.get("trainer", {})
    if isinstance(trainer_cfg, Mapping):
        trainer = dict(trainer_cfg)
        adapter_cfg = config.get("adapter", {})
        adapter_type = (
            str(adapter_cfg.get("type", "")).lower()
            if isinstance(adapter_cfg, Mapping)
            else ""
        )
        trainer_type = str(trainer.get("type", "")).lower()
        method_type = trainer_type or adapter_type
        foreground_loss_methods = {"diffsoup"}
        trainer.setdefault(
            "foreground_training",
            mode == "foreground" and method_type in foreground_loss_methods,
        )
        if mode == "foreground":
            trainer["white_background"] = True
        else:
            trainer.setdefault("white_background", False)
        config["trainer"] = trainer

    adapter_cfg = config.get("adapter", {})
    if isinstance(adapter_cfg, Mapping):
        adapter = dict(adapter_cfg)
        render_params = dict(adapter.get("render_params", {}) or {})
        render_params["bg_color"] = "white" if mode == "foreground" else "black"
        adapter["render_params"] = render_params
        config["adapter"] = adapter
    return config


def _format_context(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    trainer_cfg = config.get("trainer", {})
    if not isinstance(trainer_cfg, Mapping):
        trainer_cfg = {}
    adapter_cfg = config.get("adapter", {})
    if not isinstance(adapter_cfg, Mapping):
        adapter_cfg = {}

    scene = dataset_cfg.get("scene", "")
    root = dataset_cfg.get("root", dataset_cfg.get("dataset_path", ""))
    dataset_name = dataset_cfg.get("name") or dataset_cfg.get("dataset")
    if dataset_name is None and root:
        dataset_name = Path(str(root).replace("{scene}", "")).name
    dataset_template_name = _dataset_template_name(dataset_cfg, dataset_name)
    raw_method = adapter_cfg.get("type") or adapter_cfg.get("name") or trainer_cfg.get("type") or trainer_cfg.get("name")
    raw_method_str = str(raw_method) if raw_method is not None else ""
    canonical_method = raw_method_str.lower().replace("_", "-")
    if canonical_method == "d2ts":
        canonical_method = "2dts"

    method_template_val = raw_method_str
    cap = trainer_cfg.get("max_primitives")
    if (
        cap is not None
        and int(cap) > 0
        and canonical_method in {"triangle-splatting", "2dts"}
    ):
        method_template_val = f"abl/{canonical_method}-{int(cap)}"

    max_steps = trainer_cfg.get("max_steps", "")

    return {
        "scene": str(scene) if scene is not None else "",
        "dataset": str(dataset_template_name) if dataset_template_name is not None else "",
        "dataset_name": str(dataset_name) if dataset_name is not None else "",
        "dtu_eval_mode": str(dataset_cfg.get("dtu_eval_mode", "")),
        "method": str(method_template_val),
        "method_name": str(raw_method_str),
        "max_steps": str(max_steps) if max_steps is not None else "",
    }


def _dataset_template_name(dataset_cfg: Mapping[str, Any], dataset_name: Any) -> Any:
    dataset_type = str(dataset_cfg.get("type", "")).lower()
    name_text = str(dataset_name).lower() if dataset_name is not None else ""
    if dataset_type != "dtu" and name_text != "dtu":
        return dataset_name

    mode = str(dataset_cfg.get("dtu_eval_mode", "full")).lower()
    if mode in {"foreground", "fg"}:
        return "dtu-fg"
    return "dtu-full"


def _format_value(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _format_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, str) and "{" in value and "}" in value:
        return value.format_map(_SafeFormatDict(context))
    return value


def _resolve_dataset_scene(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        return config

    dataset = dict(dataset_cfg)
    scene = dataset.get("scene")
    key = "root" if "root" in dataset else "dataset_path" if "dataset_path" in dataset else None
    if scene and key is not None:
        root = Path(str(dataset[key])).expanduser()
        scene_name = Path(str(scene)).name
        if root.name != scene_name:
            root = root / scene_name
        dataset[key] = str(root)
    config["dataset"] = dataset
    return config


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = _safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")
    return data


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _safe_load(text: str) -> Any:
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _mini_yaml_load(text)


def _safe_dump(data: Any) -> str:
    if _yaml is not None:
        return _yaml.safe_dump(data, sort_keys=False)
    return _mini_yaml_dump(data)


def _mini_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by TriBench example configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending: list[tuple[int, Any, str]] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        while pending and indent <= pending[-1][0]:
            pending.pop()
        while stack and indent <= stack[-1][0]:
            stack.pop()

        if content.startswith("- "):
            parent = stack[-1][1]
            if not isinstance(parent, list):
                if not pending:
                    raise ValueError(f"YAML list item has no parent: {raw_line}")
                _, owner, key = pending.pop()
                parent = []
                owner[key] = parent
                stack.append((indent - 1, parent))
            item_text = content[2:].strip()
            if not item_text:
                item: Any = {}
            elif ":" in item_text:
                key, value = item_text.split(":", 1)
                item = {key.strip(): _parse_mini_yaml_value(value.strip())}
            else:
                item = _parse_mini_yaml_value(item_text)
            parent.append(item)
            if isinstance(item, dict):
                stack.append((indent, item))
            continue

        if ":" not in content:
            raise ValueError(f"Expected key/value YAML line: {raw_line}")

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"Cannot assign key inside YAML list scalar: {raw_line}")

        if value:
            parent[key] = _parse_mini_yaml_value(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            pending.append((indent, parent, key))
            stack.append((indent, child))

    return root


def _parse_mini_yaml_value(value: str) -> Any:
    if value in {"[]", "{}"}:
        return [] if value == "[]" else {}
    return _coerce_scalar(value)


def _mini_yaml_dump(data: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(data, Mapping):
        for key, value in data.items():
            if isinstance(value, Mapping):
                lines.append(f"{prefix}{key}:")
                lines.append(_mini_yaml_dump(value, indent + 2).rstrip())
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(_mini_yaml_dump(value, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_mini_yaml_scalar(value)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}-")
                lines.append(_mini_yaml_dump(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_mini_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_format_mini_yaml_scalar(data)}")
    return "\n".join(line for line in lines if line) + "\n"


def _format_mini_yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)
