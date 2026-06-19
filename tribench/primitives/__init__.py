"""Primitive types for triangle splatting methods."""

from tribench.primitives.base import BasePrimitive
from tribench.primitives.triangle import IndependentTriangle
from tribench.primitives.mesh_triangle import IndexedMeshTriangle
from tribench.primitives.convex_triangle import ConvexTriangle

__all__ = [
    "BasePrimitive",
    "IndependentTriangle",
    "IndexedMeshTriangle",
    "ConvexTriangle",
]
