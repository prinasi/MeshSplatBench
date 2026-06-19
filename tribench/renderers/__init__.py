"""Renderer adapters for triangle splatting methods.

Adapter modules register themselves with :mod:`tribench.core.registry`.
They are imported lazily so importing ``tribench.renderers.backends`` does
not recursively import adapters while the registry is still initializing.
"""

from __future__ import annotations

from tribench.renderers.base import RendererAdapter

__all__ = [
    "RendererAdapter",
    "D2TSAdapter",
    "TriangleSplattingAdapter",
    "MeshSplattingAdapter",
    "DiffSoupAdapter",
]


def __getattr__(name: str):
    if name == "D2TSAdapter":
        from tribench.renderers.d2ts_adapter import D2TSAdapter

        return D2TSAdapter
    if name == "TriangleSplattingAdapter":
        from tribench.renderers.triangle_splatting_adapter import TriangleSplattingAdapter

        return TriangleSplattingAdapter
    if name == "MeshSplattingAdapter":
        from tribench.renderers.mesh_splatting_adapter import MeshSplattingAdapter

        return MeshSplattingAdapter
    if name == "DiffSoupAdapter":
        from tribench.renderers.diffsoup_adapter import DiffSoupAdapter

        return DiffSoupAdapter
    raise AttributeError(name)
