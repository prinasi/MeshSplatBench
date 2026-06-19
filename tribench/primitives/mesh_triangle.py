"""Indexed mesh triangle primitive (MeshSplatting-style)."""

from __future__ import annotations

from typing import Any

import torch

from tribench.primitives.base import BasePrimitive


class IndexedMeshTriangle(BasePrimitive):
    """Indexed mesh with shared vertices (MeshSplatting-style).
    
    Triangles share vertices through an index buffer, forming a mesh topology.
    This representation supports vertex-level attributes like weights and
    displacement, which is key to MeshSplatting's approach.
    
    Attributes:
        vertices: Vertex positions, shape [V, 3].
        faces: Triangle face indices, shape [F, 3].
        vertex_weights: Per-vertex weights, shape [V] or [V, D].
        opacity: Per-face opacity, shape [F].
        sigma: Global or per-face sigma parameter.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_weights: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
        sigma: float | torch.Tensor = 1.0,
    ):
        """Initialize indexed mesh triangles.
        
        Args:
            vertices: Vertex positions, shape [V, 3].
            faces: Triangle face indices, shape [F, 3], values in [0, V).
            vertex_weights: Per-vertex weights for MeshSplatting, shape [V] or [V, D].
            opacity: Per-face opacity, shape [F]. Defaults to ones.
            sigma: Global sigma or per-face sigma tensor.
        """
        assert vertices.dim() == 2 and vertices.shape[1] == 3, (
            f"Expected vertices shape [V, 3], got {vertices.shape}"
        )
        assert faces.dim() == 2 and faces.shape[1] == 3, (
            f"Expected faces shape [F, 3], got {faces.shape}"
        )
        self.vertices = vertices
        self.faces = faces
        self.vertex_weights = vertex_weights
        self.opacity = opacity if opacity is not None else torch.ones(faces.shape[0], device=vertices.device)
        self.sigma = sigma

    @property
    def num_primitives(self) -> int:
        return self.faces.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def get_face_vertices(self) -> torch.Tensor:
        """Get the 3D positions of vertices for each face.
        
        Returns:
            Face vertex positions, shape [F, 3, 3].
        """
        return self.vertices[self.faces]

    def to_tensor(self) -> torch.Tensor:
        """Export as triangle soup tensor [F, 3, 3]."""
        return self.get_face_vertices()

    def compute_areas(self) -> torch.Tensor:
        fv = self.get_face_vertices()
        v0 = fv[:, 0, :]
        v1 = fv[:, 1, :]
        v2 = fv[:, 2, :]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=-1)
        return 0.5 * torch.norm(cross, dim=-1)

    def compute_normals(self) -> torch.Tensor:
        fv = self.get_face_vertices()
        v0 = fv[:, 0, :]
        v1 = fv[:, 1, :]
        v2 = fv[:, 2, :]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=-1)
        norms = torch.norm(cross, dim=-1, keepdim=True).clamp(min=1e-8)
        return cross / norms

    def to(self, device: torch.device | str) -> IndexedMeshTriangle:
        sigma = self.sigma
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device)
        return IndexedMeshTriangle(
            vertices=self.vertices.to(device),
            faces=self.faces.to(device),
            vertex_weights=self.vertex_weights.to(device) if self.vertex_weights is not None else None,
            opacity=self.opacity.to(device),
            sigma=sigma,
        )

    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    def extra_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        sigma = self.sigma
        if isinstance(sigma, torch.Tensor):
            stats["sigma_mean"] = sigma.mean().item()
            stats["sigma_std"] = sigma.std().item()
        else:
            stats["sigma"] = float(sigma)

        if self.vertex_weights is not None:
            w = self.vertex_weights
            if w.dim() == 1:
                stats["vertex_weight_mean"] = w.mean().item()
                stats["vertex_weight_std"] = w.std().item()
            else:
                stats["vertex_weight_dim"] = w.shape[-1]

        return stats
