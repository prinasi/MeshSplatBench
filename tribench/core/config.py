"""Configuration loading utilities for TriBench.

TriBench configs are plain YAML mappings with optional ``_base_`` inheritance.
They are intentionally small compared with MMEngine/MMDetection configs, but
keep the same useful shape: declare components in YAML and build them through
registries.
"""

from __future__ import annotations

from copy import deepcopy
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
        merged = merge_dicts(merged, load_config(base_path))
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


def save_config_snapshot(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Save the fully resolved config to ``output_dir/config.yaml``."""
    path = Path(output_dir).expanduser() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_dump(_to_plain_dict(config)))
    return path


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
