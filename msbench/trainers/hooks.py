"""Training method hooks for MeshSplatBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from msbench.core.cameras import CameraBatch
from msbench.renderers.base import RenderOutput


class TrainingMethod(ABC):
    """Abstract training method interface.
    
    Each triangle splatting method has different training behavior:
    - Different densification policies
    - Different pruning and opacity schedules
    - Different regularizers
    - Different mesh topology updates
    - Different optimizer state handling
    
    This hook-based interface allows common logging and profiling
    without forcing identical training semantics.
    """

    @abstractmethod
    def sample_batch(self) -> tuple[CameraBatch, torch.Tensor]:
        """Sample a training batch.
        
        Returns:
            Tuple of (CameraBatch, ground-truth images [B, H, W, 3]).
        """
        ...

    @abstractmethod
    def render_train(self, cameras: CameraBatch) -> RenderOutput:
        """Render training views.
        
        Args:
            cameras: CameraBatch from sample_batch().
            
        Returns:
            RenderOutput with rendered images.
        """
        ...

    @abstractmethod
    def compute_loss(self, output: RenderOutput, gt_images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute training loss.
        
        Args:
            output: RenderOutput from render_train().
            gt_images: Ground-truth images.
            
        Returns:
            Dictionary with loss components (e.g., 'l1', 'ssim', 'total').
        """
        ...

    @abstractmethod
    def optimizer_step(self) -> None:
        """Perform one optimizer step."""
        ...

    @abstractmethod
    def update_structure(self, step: int) -> dict[str, Any] | None:
        """Update model structure (densification, pruning, topology changes).
        
        Args:
            step: Current training step.
            
        Returns:
            Optional dictionary with structure update statistics.
        """
        ...

    def on_step_end(self, step: int, losses: dict[str, float]) -> None:
        """Called at the end of each training step. Override for custom logging."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch. Override for checkpoint saving, etc."""
        pass

    def get_lr(self) -> dict[str, float]:
        """Get current learning rates for all parameter groups."""
        return {}

    def set_step(self, step: int) -> None:
        """Synchronize the current global training step with the method."""
        self._step = step

    def resume_from_checkpoint(self, output_dir: str | Path, max_steps: int) -> int:
        """Optionally resume from an existing checkpoint.

        Returns the last completed training step. The default implementation
        keeps existing trainers opt-in and preserves from-scratch behavior.
        """
        return 0
