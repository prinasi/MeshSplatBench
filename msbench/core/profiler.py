"""Profiling utilities for timing and memory measurement."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

PROFILE_SCHEMA_VERSION = 1


def _cuda_device(device: str | torch.device | None = None) -> torch.device | None:
    """Return the CUDA device used for a measurement, if any."""
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else None
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return resolved


def _cuda_sync(device: str | torch.device | None = None) -> None:
    """Synchronize the selected CUDA device, if any."""
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)


def _time_fn(
    fn: Callable,
    repeats: int,
    warmup: int = 5,
    *,
    device: str | torch.device | None = None,
) -> list[float]:
    """Time a function call using CUDA events when available.
    
    Args:
        fn: Function to time (no arguments).
        repeats: Number of timed repetitions.
        warmup: Number of warmup runs (not measured).
        
    Returns:
        List of elapsed times in milliseconds.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    cuda_device = _cuda_device(device)
    for _ in range(warmup):
        fn()
        _cuda_sync(cuda_device)

    timings = []

    for _ in range(repeats):
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize(cuda_device)
                timings.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return timings


def _summarize_timings(timings: list[float]) -> dict[str, float]:
    """Compute summary statistics for timing measurements.
    
    Args:
        timings: List of elapsed times in milliseconds.
        
    Returns:
        Dictionary with mean_ms, std_ms, p50_ms, p95_ms, min_ms, max_ms, fps.
    """
    import numpy as np

    arr = np.array(timings)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


def profile_forward(
    renderer: Any,
    cameras: Any,
    repeats: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    """Profile renderer forward pass.
    
    Args:
        renderer: A RendererAdapter instance.
        cameras: CameraBatch for rendering.
        repeats: Number of timed repetitions.
        warmup: Number of warmup runs.
        
    Returns:
        Timing summary dictionary.
    """

    def forward_fn():
        with torch.no_grad():
            renderer.render(cameras, mode="eval")

    timings = _time_fn(forward_fn, repeats, warmup, device=renderer.device)
    return _summarize_timings(timings)


def _zero_grad(renderer: Any) -> None:
    model = getattr(renderer, "_model", None)
    optimizer = getattr(model, "optimizer", None)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    elif isinstance(model, torch.nn.Module):
        model.zero_grad(set_to_none=True)


def _timed_call(
    fn: Callable[[], Any],
    device: str | torch.device | None,
) -> tuple[Any, float]:
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        with torch.cuda.device(cuda_device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn()
            end_event.record()
            torch.cuda.synchronize(cuda_device)
            return result, float(start_event.elapsed_time(end_event))

    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000.0


def profile_forward_backward(
    renderer: Any,
    cameras: Any,
    repeats: int = 100,
    warmup: int = 5,
) -> dict[str, dict[str, float]]:
    """Profile renderer forward and backward passes.
    
    Args:
        renderer: A RendererAdapter instance (must support gradient computation).
        cameras: CameraBatch for rendering.
        repeats: Number of timed repetitions.
        warmup: Number of warmup runs.
        
    Returns:
        Dictionary with 'forward', 'backward', and 'total' timing summaries.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    forward_timings: list[float] = []
    backward_timings: list[float] = []
    for index in range(warmup + repeats):
        _zero_grad(renderer)

        def forward_fn():
            output = renderer.render(cameras, mode="train")
            loss = output.rgb.mean()
            if not loss.requires_grad:
                raise RuntimeError(
                    f"{renderer.__class__.__name__} does not expose a differentiable train render."
                )
            return loss

        loss, forward_ms = _timed_call(forward_fn, renderer.device)
        _, backward_ms = _timed_call(loss.backward, renderer.device)
        _zero_grad(renderer)
        if index >= warmup:
            forward_timings.append(forward_ms)
            backward_timings.append(backward_ms)

    total_timings = [fwd + bwd for fwd, bwd in zip(forward_timings, backward_timings)]

    return {
        "forward": _summarize_timings(forward_timings),
        "backward": _summarize_timings(backward_timings),
        "total": _summarize_timings(total_timings),
    }


def profile_backward(
    renderer: Any,
    cameras: Any,
    repeats: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    """Profile only the backward portion of differentiable train renders."""
    return profile_forward_backward(renderer, cameras, repeats, warmup)["backward"]


def measure_peak_memory(
    fn: Callable,
    *,
    device: str | torch.device | None = None,
) -> dict[str, float | str]:
    """Measure peak CUDA memory usage of a function.
    
    Args:
        fn: Function to measure.
        
    Returns:
        Dictionary with peak_memory_allocated_gb, peak_memory_reserved_gb.
    """
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        note = "CUDA not available" if not torch.cuda.is_available() else "Renderer device is not CUDA"
        return {
            "peak_memory_allocated_gb": 0.0,
            "peak_memory_reserved_gb": 0.0,
            "note": note,
        }

    with torch.cuda.device(cuda_device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

        fn()
        torch.cuda.synchronize(cuda_device)

        peak_allocated = torch.cuda.max_memory_allocated(cuda_device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(cuda_device) / (1024**3)

    return {
        "peak_memory_allocated_gb": float(peak_allocated),
        "peak_memory_reserved_gb": float(peak_reserved),
    }


def profile_renderer(
    renderer: Any,
    cameras: Any,
    *,
    repeats: int = 100,
    warmup: int = 5,
    include_backward: bool = False,
) -> dict[str, Any]:
    """Collect the adapter-independent MeshSplatBench renderer profile schema."""
    forward = profile_forward(renderer, cameras, repeats=repeats, warmup=warmup)
    def memory_fn():
        with torch.no_grad():
            renderer.render(cameras, mode="eval")

    memory = measure_peak_memory(memory_fn, device=renderer.device)
    metadata = renderer.profile_metadata()
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "forward_latency_ms": forward["mean_ms"],
        "fps": forward["fps"],
        "peak_cuda_memory_gb": memory["peak_memory_allocated_gb"],
        "primitive_count": metadata.get("primitive_count"),
        "checkpoint_size_mb": metadata.get("checkpoint_size_mb"),
        "forward": forward,
        "memory": memory,
    }
    if include_backward:
        backward = profile_backward(renderer, cameras, repeats=repeats, warmup=warmup)
        result["backward_latency_ms"] = backward["mean_ms"]
        result["backward"] = backward
    return result
