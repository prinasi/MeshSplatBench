"""Primitive types for triangle splatting methods."""

from trianglebench.primitives.base import BasePrimitive
from trianglebench.primitives.triangle import IndependentTriangle
from trianglebench.primitives.mesh_triangle import IndexedMeshTriangle
from trianglebench.primitives.convex_triangle import ConvexTriangle

__all__ = [
    "BasePrimitive",
    "IndependentTriangle",
    "IndexedMeshTriangle",
    "ConvexTriangle",
]
