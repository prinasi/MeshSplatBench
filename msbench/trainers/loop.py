"""Training loop orchestrator for MeshSplatBench."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from msbench.trainers.hooks import TrainingMethod


@dataclass
class TrainingConfig:
    """Configuration for the training loop.
    
    Attributes:
        max_steps: Maximum number of training steps.
        log_interval: Steps between log outputs.
        eval_interval: Steps between evaluation runs.
        save_interval: Steps between checkpoint saves.
        output_dir: Directory for outputs.
    """

    max_steps: int = 30000
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./outputs"


class TrainingLoop:
    """Generic training loop that delegates to TrainingMethod hooks.
    
    This orchestrator handles:
    - Step iteration and timing
    - Logging of losses and learning rates
    - Periodic evaluation and checkpoint saving
    - Structure update callbacks
    
    It does NOT enforce any specific training semantics — those are
    handled by the TrainingMethod implementation.
    """

    def __init__(
        self,
        method: TrainingMethod,
        config: TrainingConfig | None = None,
    ):
        """Initialize the training loop.
        
        Args:
            method: TrainingMethod implementation with hooks.
            config: Training configuration.
        """
        self.method = method
        self.config = config or TrainingConfig()
        self._step = 0
        self._history: list[dict[str, float]] = []

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    @property
    def history(self) -> list[dict[str, float]]:
        """Training loss history."""
        return self._history

    def train(self) -> dict[str, Any]:
        """Run the full training loop.
        
        Returns:
            Dictionary with training summary (final losses, total time, etc.).
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        step_times = []
        start_step = self.method.resume_from_checkpoint(output_dir, cfg.max_steps)
        start_step = max(0, min(int(start_step), cfg.max_steps))
        self._step = start_step

        if start_step > 0:
            print(f"Resuming training from step {start_step}.")
        print(f"Starting training for {cfg.max_steps} steps...")
        print(f"Output directory: {output_dir}")

        for step in range(start_step + 1, cfg.max_steps + 1):
            self._step = step
            self.method.set_step(step)
            t0 = time.perf_counter()

            # Sample batch
            cameras, gt_images = self.method.sample_batch()

            # Forward pass
            output = self.method.render_train(cameras)

            # Compute loss
            losses = self.method.compute_loss(output, gt_images)
            total_loss = losses["total"]

            # Backward pass
            total_loss.backward()

            # Step end callback
            loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            self.method.on_step_end(step, loss_values)

            step_time = time.perf_counter() - t0
            step_times.append(step_time)

            # Record history
            self._history.append(loss_values)

            # Logging
            if step % cfg.log_interval == 0:
                lr_info = self.method.get_lr()
                self._log_step(step, loss_values, step_time, lr_info)

            # Checkpointing
            if step % cfg.save_interval == 0 or step == cfg.max_steps:
                with torch.no_grad():
                    self.method.on_epoch_end(step)

            # Structure update
            update_info = self.method.update_structure(step)
            if update_info is not None:
                self._log_structure_update(step, update_info)

            # Optimizer step
            self.method.optimizer_step()

        total_time = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0

        summary = {
            "total_steps": cfg.max_steps,
            "start_step": start_step,
            "trained_steps": max(cfg.max_steps - start_step, 0),
            "total_time_s": total_time,
            "avg_step_time_ms": avg_step_time * 1000,
            "final_losses": self._history[-1] if self._history else {},
        }

        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Average step time: {avg_step_time * 1000:.2f}ms")

        return summary

    def _log_step(
        self,
        step: int,
        losses: dict[str, float],
        step_time: float,
        lr_info: dict[str, float],
    ) -> None:
        """Log training step information."""
        loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
        lr_str = " | ".join(f"lr_{k}: {v:.2e}" for k, v in lr_info.items()) if lr_info else ""
        parts = [f"Step {step:6d}", loss_str, f"{step_time * 1000:.1f}ms"]
        if lr_str:
            parts.append(lr_str)
        print(" | ".join(parts))

    def _log_structure_update(self, step: int, update_info: dict[str, Any]) -> None:
        """Log topology changes such as densification and pruning."""
        update_type = update_info.get("type", "structure")
        before = update_info.get("triangles_before", "?")
        after = update_info.get("triangles_after", "?")
        delta = update_info.get("delta", "?")
        delta_text = f"{delta:+}" if isinstance(delta, (int, float)) else str(delta)
        parts = [
            f"Step {step:6d}",
            f"{update_type}: {before} -> {after} ({delta_text})",
        ]
        dead_keys = [
            ("dead_total", "dead"),
            ("dead_opacity", "opacity"),
            ("dead_importance", "importance"),
            ("dead_area", "area"),
            ("dead_image_size", "image_size"),
        ]
        dead_parts = [
            f"{label}: {update_info[key]}"
            for key, label in dead_keys
            if key in update_info
        ]
        if dead_parts:
            parts.append(" | ".join(dead_parts))
        print(" | ".join(parts))
