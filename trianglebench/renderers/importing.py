"""Compatibility import helpers for bundled renderer backends."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from trianglebench.renderers.backends import BackendSpec, BackendUnavailableError


def purge_modules(roots: tuple[str, ...]) -> None:
    """Compatibility no-op.

    TriangleBench now vendors renderer modules under unique package names, so
    there are no shared top-level modules to evict from ``sys.modules``.
    """


@contextmanager
def backend_import_context(
    spec: BackendSpec,
    repo_root: str | Path | None = None,
) -> Iterator[None]:
    """Compatibility context for old adapter code.

    Bundled backends no longer need temporary ``sys.path`` mutation.
    """
    yield


def import_backend_module(
    spec: BackendSpec,
    module_name: str | None = None,
    repo_root: str | Path | None = None,
) -> ModuleType:
    """Import a bundled backend module.

    If ``module_name`` is omitted, the backend's canonical package is imported.
    """
    target = module_name or spec.package
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Cannot import bundled module '{target}' for backend '{spec.name}'. "
            "Install TriangleBench with its bundled CUDA extensions built."
        ) from exc


def ensure_module_from_backend(
    module: ModuleType,
    spec: BackendSpec,
    repo_root: str | Path | None = None,
) -> None:
    """Validate that a module is inside the TriangleBench vendored namespace."""
    if not module.__name__.startswith("trianglebench.vendor."):
        raise BackendUnavailableError(
            f"Imported '{module.__name__}', but backend '{spec.name}' expects "
            "a bundled trianglebench.vendor module."
        )

