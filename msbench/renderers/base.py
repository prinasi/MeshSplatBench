"""Base renderer adapter interface for MeshSplatBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from msbench.core.cameras import CameraBatch


@dataclass
class RenderOutput:
    """Standardized output from all renderer adapters.
    
    Attributes:
        rgb: Rendered RGB image, shape [H, W, 3] or [B, H, W, 3].
        alpha: Optional alpha/opacity map, shape [H, W] or [B, H, W].
        depth: Optional depth map, shape [H, W] or [B, H, W].
        normal: Optional normal map, shape [H, W, 3] or [B, H, W, 3].
        radii: Optional screen-space radii, shape [N].
        visibility: Optional visibility mask, shape [N].
        extras: Backend-specific extra outputs (distortion maps, contribution counts, etc.).
    """

    rgb: torch.Tensor
    alpha: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    radii: torch.Tensor | None = None
    visibility: torch.Tensor | None = None
    extras: dict[str, Any] | None = None


class RendererAdapter(ABC):
    """Abstract base class for renderer adapters.
    
    Every supported triangle splatting method implements one adapter
    that converts CameraBatch to its native format and returns RenderOutput.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name identifier."""
        ...

    @property
    def device(self) -> torch.device:
        """Device the model is loaded on."""
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        """Return backend availability diagnostics.

        Adapters backed by an external repository override this method.
        """
        return {"name": self.name, "repo_exists": None, "modules": []}

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a trained model checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        """
        ...

    @abstractmethod
    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        """Load a scene from a dataset directory.
        
        Args:
            dataset_path: Path to the dataset root.
            split: Dataset split to load.
        """
        ...

    @abstractmethod
    def model_stats(self) -> dict[str, Any]:
        """Compute model statistics.
        
        Returns:
            Dictionary with generic and method-specific statistics.
        """
        ...

    def profile_metadata(self) -> dict[str, int | float | None]:
        """Return the model fields included in every profiling result."""
        stats = self.model_stats()
        primitive_count = stats.get("primitive_count")
        checkpoint_size_mb = stats.get("checkpoint_size_mb")
        return {
            "primitive_count": int(primitive_count) if primitive_count is not None else None,
            "checkpoint_size_mb": (
                float(checkpoint_size_mb) if checkpoint_size_mb is not None else None
            ),
        }

    @abstractmethod
    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        """Render images from the given cameras.
        
        Args:
            cameras: CameraBatch with camera parameters.
            mode: Rendering mode ('eval' for no gradients, 'train' for gradient computation).
            
        Returns:
            RenderOutput with rendered images and optional auxiliary outputs.
        """
        ...

    def profile_forward(self, cameras: CameraBatch, repeats: int = 100) -> dict[str, float]:
        """Profile the forward pass.
        
        Args:
            cameras: CameraBatch for rendering.
            repeats: Number of profiling repetitions.
            
        Returns:
            Timing summary dictionary.
        """
        from msbench.core.profiler import profile_forward
        return profile_forward(self, cameras, repeats=repeats)

    def profile_forward_backward(self, cameras: CameraBatch, repeats: int = 100) -> dict[str, dict[str, float]]:
        """Profile forward and backward passes.
        
        Args:
            cameras: CameraBatch for rendering.
            repeats: Number of profiling repetitions.
            
        Returns:
            Dictionary with forward, backward, and total timing summaries.
        """
        from msbench.core.profiler import profile_forward_backward
        return profile_forward_backward(self, cameras, repeats=repeats)

    def to_primitive(self) -> Any:
        """Extract the model's primitive representation.
        
        Returns:
            A BasePrimitive subclass instance representing the model's geometry.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_primitive() is not implemented. "
            "This method should convert the internal model to a MeshSplatBench primitive type."
        )
