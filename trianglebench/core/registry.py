"""Method and adapter registry for TriangleBench.

Adapters are discovered via the ``trianglebench.adapters`` entry-point group,
so both built-in and third-party adapters register the same way — through
``pyproject.toml`` (or ``setup.cfg``) without modifying any TriangleBench
source file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trianglebench.renderers.base import RendererAdapter
from trianglebench.renderers.backends import canonical_backend_name

_REGISTRY: dict[str, type[RendererAdapter]] = {}
_PLUGINS_DISCOVERED = False

#: Entry point group that packages use to register renderer adapters.
ENTRY_POINT_GROUP = "trianglebench.adapters"


def register(name: str, adapter_cls: type | None = None):
    """Register a renderer adapter.

    Can be used as a decorator or a direct function call::

        @register("my_method")
        class MyAdapter(RendererAdapter):
            ...

        # or equivalently:
        register("my_method", MyAdapter)
    """

    def _register(cls):
        _REGISTRY[name] = cls
        return cls

    if adapter_cls is not None:
        _REGISTRY[name] = adapter_cls
        return adapter_cls

    return _register


def _discover_plugins() -> None:
    """Load all entry points in the ``trianglebench.adapters`` group.

    Each entry point should point to a module whose top-level code calls
    ``register()`` (typically via a ``@register`` decorator on a
    ``RendererAdapter`` subclass).  Built-in adapters declare their entry
    points in TriangleBench's own ``pyproject.toml``.
    """
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    _PLUGINS_DISCOVERED = True

    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            ep.load()
        except Exception:
            # Silently skip broken entry points so one bad plugin
            # does not prevent the rest of the framework from working.
            pass


def get_adapter(name: str) -> type[RendererAdapter]:
    """Get a registered renderer adapter class.

    Args:
        name: Method name (e.g., '2dts', 'triangle-splatting').

    Returns:
        The adapter class.

    Raises:
        KeyError: If the method is not registered.
    """
    _discover_plugins()

    try:
        lookup_name = canonical_backend_name(name)
    except KeyError:
        lookup_name = name

    if lookup_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Method '{name}' not found in registry. "
            f"Available methods: {available or '(none registered)'}"
        )
    return _REGISTRY[lookup_name]


def list_methods() -> list[str]:
    """List all registered method names."""
    _discover_plugins()
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a method is registered."""
    _discover_plugins()
    try:
        name = canonical_backend_name(name)
    except KeyError:
        pass
    return name in _REGISTRY
