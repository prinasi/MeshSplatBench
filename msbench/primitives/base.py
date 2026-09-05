"""Base class for triangle primitive representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BasePrimitive(ABC):
    """Abstract base class for triangle primitives.
    
    All triangle splatting methods represent their geometry using
    some form of triangles. This base class defines the common interface.
    """

    @property
    @abstractmethod
    def num_primitives(self) -> int:
        """Number of triangle primitives."""
        ...

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        """Total number of unique vertices."""
        ...

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """Export primitive geometry as a tensor.
        
        Returns:
            Tensor representation of the primitives.
            Shape depends on the primitive type.
        """
        ...

    @abstractmethod
    def compute_areas(self) -> torch.Tensor:
        """Compute the area of each primitive.
        
        Returns:
            Tensor of areas, shape [num_primitives].
        """
        ...

    @abstractmethod
    def compute_normals(self) -> torch.Tensor:
        """Compute the normal vector of each primitive.
        
        Returns:
            Tensor of normals, shape [num_primitives, 3].
        """
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> BasePrimitive:
        """Move primitives to a device."""
        ...

    def get_opacity(self) -> torch.Tensor | None:
        """Get per-primitive opacity values, if available."""
        return None

    def get_colors(self) -> torch.Tensor | None:
        """Get per-primitive colors, if available."""
        return None

    def extra_stats(self) -> dict[str, Any]:
        """Return primitive-type-specific statistics."""
        return {}
