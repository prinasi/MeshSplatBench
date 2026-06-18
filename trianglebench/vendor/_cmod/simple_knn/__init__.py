"""Vendored simple-knn CUDA wrapper."""

from __future__ import annotations

import torch

from . import _C


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """Return the mean squared distance to each point's 3 nearest neighbors."""
    return _C.distCUDA2(points)


def nearestNeighbor(points: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """Return nearest-neighbor indices, optionally grouped by batch."""
    return _C.nearestNeighbor(points, batch_size)

