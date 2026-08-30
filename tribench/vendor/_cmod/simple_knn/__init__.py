"""Vendored simple-knn CUDA wrapper."""

from __future__ import annotations

try:
    from . import _C
except ImportError:
    _C = None


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """Return the mean squared distance to each point's 3 nearest neighbors."""
    if _C is None:
        if points.shape[0] <= 1:
            return torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
        k = min(4, points.shape[0])
        dist = torch.cdist(points.float(), points.float())
        topk = torch.topk(dist, k=k, largest=False).values[:, 1:]
        return (topk ** 2).mean(dim=-1)
    return _C.distCUDA2(points)


def nearestNeighbor(points: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """Return nearest-neighbor indices, optionally grouped by batch."""
    if _C is None:
        raise RuntimeError("simple_knn CUDA extension is not compiled.")
    return _C.nearestNeighbor(points, batch_size)

