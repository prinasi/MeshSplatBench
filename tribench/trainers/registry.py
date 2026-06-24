"""Training method registry for TriBench.

Training methods are registered by name and discovered via entry points,
allowing third-party packages to provide new training pipelines without
modifying TriBench source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tribench.trainers.hooks import TrainingMethod

_TRAINING_METHODS: dict[str, type[TrainingMethod]] = {}
_PLUGINS_DISCOVERED = False

#: Preferred entry point group that third-party packages use to register methods.
ENTRY_POINT_GROUP = "tribench.training_methods"


def register_training_method(name: str, cls: type | None = None):
    """Register a training method.

    Can be used as a decorator or a direct function call::

        @register_training_method("my-method")
        class MyTrainingMethod(TrainingMethod):
            ...

        # or equivalently:
        register_training_method("my-method", MyTrainingMethod)
    """

    def _do_register(klass):
        _TRAINING_METHODS[name] = klass
        return klass

    if cls is not None:
        _TRAINING_METHODS[name] = cls
        return cls

    return _do_register


def _discover_plugins() -> None:
    """Load built-in training methods and entry points.

    Each entry point should point to a module whose top-level code calls
    ``register_training_method`` (typically via a ``@register_training_method``
    decorator on a ``TrainingMethod`` subclass).
    """
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    _PLUGINS_DISCOVERED = True

    try:
        __import__("tribench.trainers.triangle_splatting_method")
    except Exception:
        pass

    try:
        __import__("tribench.trainers.mesh_splatting_method")
    except Exception:
        pass

    try:
        __import__("tribench.trainers.diffsoup_native")
    except Exception:
        pass

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


def get_training_method(name: str) -> type[TrainingMethod]:
    """Look up a registered training method by name.

    Aliases are resolved through
    :func:`tribench.renderers.backends.canonical_backend_name`.
    """
    _discover_plugins()

    # Resolve aliases (e.g. "triangle_splatting" → "triangle-splatting").
    try:
        from tribench.renderers.backends import canonical_backend_name

        lookup = canonical_backend_name(name)
    except KeyError:
        lookup = name

    if lookup not in _TRAINING_METHODS:
        available = ", ".join(sorted(_TRAINING_METHODS)) or "(none)"
        raise KeyError(
            f"Training method '{name}' not found. Available: {available}"
        )
    return _TRAINING_METHODS[lookup]


def list_training_methods() -> list[str]:
    """Return sorted names of all registered training methods."""
    _discover_plugins()
    return sorted(_TRAINING_METHODS)


def is_training_method_registered(name: str) -> bool:
    """Check whether a training method is registered under *name*."""
    _discover_plugins()
    try:
        from tribench.renderers.backends import canonical_backend_name

        name = canonical_backend_name(name)
    except KeyError:
        pass
    return name in _TRAINING_METHODS
