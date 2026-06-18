"""Independent triangle soup primitive (2DTS-style)."""

from __future__ import annotations

from typing import Any

import torch

from trianglebench.primitives.base import BasePrimitive


class IndependentTriangle(BasePrimitive):
    """Independent triangle soup representation.
    
    Each triangle is defined by 3 independent vertices with no shared topology.
    This is the output format of 2DTS: (N, 3, 3) tensor of vertex positions.
    
    Attributes:
        vertices: Triangle vertices, shape [T, 3, 3].
        opacity: Per-triangle opacity, shape [T].
        sh_coeffs: Optional spherical harmonics coefficients.
        gamma: Compactness parameter (for 2DTS gamma annealing).
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        opacity: torch.Tensor | None = None,
        sh_coeffs: torch.Tensor | None = None,
        gamma: float = 1.0,
        back_culling: bool = False,
    ):
        """Initialize independent triangles.
        
        Args:
            vertices: Triangle vertices, shape [T, 3, 3].
            opacity: Per-triangle opacity, shape [T]. Defaults to ones.
            sh_coeffs: SH coefficients, shape [T, C] where C depends on SH degree.
            gamma: Compactness parameter for coverage function.
            back_culling: Whether to enable back-face culling.
        """
        assert vertices.dim() == 3 and vertices.shape[1:] == (3, 3), (
            f"Expected vertices shape [T, 3, 3], got {vertices.shape}"
        )
        self.vertices = vertices
        self.opacity = opacity if opacity is not None else torch.ones(vertices.shape[0], device=vertices.device)
        self.sh_coeffs = sh_coeffs
        self.gamma = gamma
        self.back_culling = back_culling

    @property
    def num_primitives(self) -> int:
        return self.vertices.shape[0]

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

    def to(self, device: torch.device | str) -> IndependentTriangle:
        return IndependentTriangle(
            vertices=self.vertices.to(device),
            opacity=self.opacity.to(device),
            sh_coeffs=self.sh_coeffs.to(device) if self.sh_coeffs is not None else None,
            gamma=self.gamma,
            back_culling=self.back_culling,
        )

    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    def get_colors(self) -> torch.Tensor | None:
        if self.sh_coeffs is not None:
            # Return DC component (first 3 channels) as color approximation
            return self.sh_coeffs[:, :3] if self.sh_coeffs.shape[-1] >= 3 else None
        return None

    def extra_stats(self) -> dict[str, Any]:
        stats = {
            "gamma": self.gamma,
            "back_culling": self.back_culling,
        }
        if self.sh_coeffs is not None:
            stats["sh_degree"] = (self.sh_coeffs.shape[-1] // 3) - 1
        return stats
