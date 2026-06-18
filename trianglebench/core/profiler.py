"""Profiling utilities for timing and memory measurement."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


def _cuda_sync() -> None:
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn: Callable, repeats: int, warmup: int = 5) -> list[float]:
    """Time a function call using CUDA events when available.
    
    Args:
        fn: Function to time (no arguments).
        repeats: Number of timed repetitions.
        warmup: Number of warmup runs (not measured).
        
    Returns:
        List of elapsed times in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
        _cuda_sync()

    timings = []
    use_cuda = torch.cuda.is_available()

    for _ in range(repeats):
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        else:
            _cuda_sync()
            t0 = time.perf_counter()
            fn()
            _cuda_sync()
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

    timings = _time_fn(forward_fn, repeats, warmup)
    return _summarize_timings(timings)


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

    def forward_fn():
        output = renderer.render(cameras, mode="train")
        loss = output.rgb.mean()
        return loss

    # Profile forward only
    forward_timings = _time_fn(lambda: forward_fn(), repeats, warmup)

    # Profile forward + backward
    def forward_backward_fn():
        loss = forward_fn()
        loss.backward()

    fb_timings = _time_fn(forward_backward_fn, repeats, warmup)

    # Compute backward = (forward+backward) - forward
    forward_summary = _summarize_timings(forward_timings)
    fb_summary = _summarize_timings(fb_timings)

    backward_timings = [fb - f for fb, f in zip(fb_timings, forward_timings)]
    backward_summary = _summarize_timings(backward_timings)

    return {
        "forward": forward_summary,
        "backward": backward_summary,
        "total": fb_summary,
    }


def measure_peak_memory(fn: Callable) -> dict[str, float]:
    """Measure peak CUDA memory usage of a function.
    
    Args:
        fn: Function to measure.
        
    Returns:
        Dictionary with peak_memory_allocated_gb, peak_memory_reserved_gb.
    """
    if not torch.cuda.is_available():
        return {
            "peak_memory_allocated_gb": 0.0,
            "peak_memory_reserved_gb": 0.0,
            "note": "CUDA not available",
        }

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    fn()
    _cuda_sync()

    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

    return {
        "peak_memory_allocated_gb": peak_allocated,
        "peak_memory_reserved_gb": peak_reserved,
    }
