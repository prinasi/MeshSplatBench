"""Convex triangle primitive with smooth coverage (Triangle Splatting-style)."""

from __future__ import annotations

from typing import Any

import torch

from tribench.primitives.base import BasePrimitive


class ConvexTriangle(BasePrimitive):
    """Triangle point primitive with smooth convex coverage function.
    
    Used by Triangle Splatting, each primitive is associated with a triangle
    but uses a sigma-parameterized smooth coverage function for differentiable
    rendering.
    
    Attributes:
        points: Triangle point positions (e.g., centroids), shape [T, 3].
        vertices: Triangle vertex positions, shape [T, 3, 3].
        sigma: Per-triangle coverage scale, shape [T].
        opacity: Per-triangle opacity, shape [T].
        sh_coeffs: Optional SH coefficients for view-dependent color.
    """

    def __init__(
        self,
        points: torch.Tensor,
        vertices: torch.Tensor,
        sigma: torch.Tensor,
        opacity: torch.Tensor | None = None,
        sh_coeffs: torch.Tensor | None = None,
    ):
        """Initialize convex triangles.
        
        Args:
            points: Triangle point positions, shape [T, 3].
            vertices: Triangle vertex positions, shape [T, 3, 3].
            sigma: Per-triangle sigma for coverage function, shape [T].
            opacity: Per-triangle opacity, shape [T]. Defaults to ones.
            sh_coeffs: SH coefficients, shape [T, C].
        """
        assert points.dim() == 2 and points.shape[1] == 3, (
            f"Expected points shape [T, 3], got {points.shape}"
        )
        assert vertices.dim() == 3 and vertices.shape[1:] == (3, 3), (
            f"Expected vertices shape [T, 3, 3], got {vertices.shape}"
        )
        self.points = points
        self.vertices = vertices
        self.sigma = sigma
        self.opacity = opacity if opacity is not None else torch.ones(points.shape[0], device=points.device)
        self.sh_coeffs = sh_coeffs

    @property
    def num_primitives(self) -> int:
        return self.points.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0] * 3

    def to_tensor(self) -> torch.Tensor:
        return self.vertices.clone()

    def compute_areas(self) -> torch.Tensor:
        v0 = self.vertices[:, 0, :]
        v1 = self.vertices[:, 1, :]
        v2 = self.vertices[:, 2, :]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=-1)
        return 0.5 * torch.norm(cross, dim=-1)

    def compute_normals(self) -> torch.Tensor:
        v0 = self.vertices[:, 0, :]
        v1 = self.vertices[:, 1, :]
        v2 = self.vertices[:, 2, :]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=-1)
        norms = torch.norm(cross, dim=-1, keepdim=True).clamp(min=1e-8)
        return cross / norms

    def to(self, device: torch.device | str) -> ConvexTriangle:
        return ConvexTriangle(
            points=self.points.to(device),
            vertices=self.vertices.to(device),
            sigma=self.sigma.to(device),
            opacity=self.opacity.to(device),
            sh_coeffs=self.sh_coeffs.to(device) if self.sh_coeffs is not None else None,
        )

    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    def get_colors(self) -> torch.Tensor | None:
        if self.sh_coeffs is not None:
            return self.sh_coeffs[:, :3] if self.sh_coeffs.shape[-1] >= 3 else None
        return None

    def extra_stats(self) -> dict[str, Any]:
        stats = {
            "sigma_mean": self.sigma.mean().item(),
            "sigma_std": self.sigma.std().item(),
            "sigma_min": self.sigma.min().item(),
            "sigma_max": self.sigma.max().item(),
        }
        if self.sh_coeffs is not None:
            stats["sh_degree"] = (self.sh_coeffs.shape[-1] // 3) - 1
        return stats
