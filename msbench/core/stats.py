"""Model statistics for triangle splatting methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import numpy as np


@dataclass
class ModelStats:
    """Statistics for a triangle splatting model.
    
    Contains both generic fields applicable to all methods and
    triangle-specific diagnostic fields.
    """

    # ---- Generic fields ----
    method: str = ""
    scene: str = ""
    primitive_count: int = 0
    vertex_count: int = 0
    face_count: int = 0
    trainable_param_count: int = 0
    checkpoint_size_mb: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    train_step_ms: float = 0.0
    render_fps: float = 0.0
    peak_memory_forward_gb: float = 0.0
    peak_memory_train_gb: float = 0.0

    # ---- Triangle-specific fields ----
    triangle_area_mean: float = 0.0
    triangle_area_median: float = 0.0
    triangle_area_p95: float = 0.0
    opacity_mean: float = 0.0
    opacity_median: float = 0.0
    opacity_histogram: list[float] = field(default_factory=list)
    visible_primitive_ratio: float = 0.0
    screen_radii_mean: float = 0.0
    screen_radii_p95: float = 0.0
    depth_valid_ratio: float = 0.0
    normal_valid_ratio: float = 0.0

    # ---- Backend-specific extra fields ----
    backend_extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON string.
        
        Args:
            path: If provided, write to this file path.
            
        Returns:
            JSON string.
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=_json_default)
        if path is not None:
            Path(path).write_text(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelStats:
        """Create ModelStats from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelStats:
        """Load ModelStats from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for non-standard types."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_triangle_areas(vertices: torch.Tensor) -> torch.Tensor:
    """Compute areas of triangles from vertex positions.
    
    Args:
        vertices: Triangle vertices, shape [T, 3, 3] (T triangles, 3 vertices, xyz).
        
    Returns:
        Triangle areas, shape [T].
    """
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = torch.cross(edge1, edge2, dim=-1)
    return 0.5 * torch.norm(cross, dim=-1)


def compute_triangle_normals(vertices: torch.Tensor) -> torch.Tensor:
    """Compute face normals of triangles.
    
    Args:
        vertices: Triangle vertices, shape [T, 3, 3].
        
    Returns:
        Unit normals, shape [T, 3].
    """
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = torch.cross(edge1, edge2, dim=-1)
    norms = torch.norm(cross, dim=-1, keepdim=True).clamp(min=1e-8)
    return cross / norms


def compute_triangle_stats(
    vertices: torch.Tensor,
    opacity: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute triangle-specific statistics from model data.
    
    Args:
        vertices: Triangle vertices, shape [T, 3, 3].
        opacity: Per-triangle opacity, shape [T]. Optional.
        
    Returns:
        Dictionary with triangle_area_mean, triangle_area_median, etc.
    """
    stats: dict[str, Any] = {}

    areas = compute_triangle_areas(vertices)
    areas_np = areas.cpu().numpy()

    stats["triangle_area_mean"] = float(np.mean(areas_np))
    stats["triangle_area_median"] = float(np.median(areas_np))
    stats["triangle_area_p95"] = float(np.percentile(areas_np, 95))

    if opacity is not None:
        opacity_np = opacity.cpu().numpy().flatten()
        stats["opacity_mean"] = float(np.mean(opacity_np))
        stats["opacity_median"] = float(np.median(opacity_np))
        # 10-bin histogram
        hist, _ = np.histogram(opacity_np, bins=10, range=(0, 1))
        stats["opacity_histogram"] = hist.tolist()

    return stats
