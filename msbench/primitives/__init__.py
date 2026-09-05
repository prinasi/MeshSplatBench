"""Primitive types for triangle splatting methods."""

from msbench.primitives.base import BasePrimitive
from msbench.primitives.triangle import IndependentTriangle
from msbench.primitives.mesh_triangle import IndexedMeshTriangle
from msbench.primitives.convex_triangle import ConvexTriangle

__all__ = [
    "BasePrimitive",
    "IndependentTriangle",
    "IndexedMeshTriangle",
    "ConvexTriangle",
]
