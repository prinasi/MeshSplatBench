"""Runtime statistics helpers for training and evaluation runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


def reset_cuda_peak_memory_stats() -> None:
    """Reset PyTorch CUDA peak memory counters when CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def cuda_peak_memory_stats() -> dict[str, float] | dict[str, str]:
    """Return PyTorch CUDA peak memory in MiB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"note": "CUDA not available"}
        torch.cuda.synchronize()
        return {
            "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_cuda_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
    except Exception as exc:
        return {"note": f"CUDA memory stats unavailable: {exc}"}


def _visible_gpu_id() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        first = visible.split(",", 1)[0].strip()
        if first:
            return first
    return None


class GpuMemoryMonitor:
    """Sample GPU memory.used with nvidia-smi in a background thread."""

    def __init__(self, gpu_id: str | None = None, interval_s: float = 1.0) -> None:
        self.gpu_id = gpu_id if gpu_id is not None else _visible_gpu_id()
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.samples: int = 0
        self.note: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.note = "nvidia-smi not found; memory monitoring skipped"
            return
        self._thread = threading.Thread(target=self._run, name="tribench-gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2.0, 2.0))
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "peak_gpu_memory_mib": self.peak_mib,
            "gpu_memory_samples": self.samples,
            "gpu_id": self.gpu_id,
            "memory_monitor_note": self.note,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_id:
            cmd.extend(["-i", self.gpu_id])
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
        except Exception as exc:
            if self.note is None:
                self.note = f"nvidia-smi sampling failed: {exc}"
            return

        values: list[int] = []
        for line in out.splitlines():
            text = line.strip()
            if text.isdigit():
                values.append(int(text))
        if not values:
            return
        current = values[0]
        self.samples += 1
        self.peak_mib = current if self.peak_mib is None else max(self.peak_mib, current)


def run_with_training_stats(
    fn: Callable[[], dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run a training callable, record timing/memory stats, and write train_stats.json."""
    output_path = Path(output_dir).expanduser()
    reset_cuda_peak_memory_stats()
    monitor = GpuMemoryMonitor()
    monitor.start()
    try:
        summary = fn()
    finally:
        memory_summary = monitor.stop()

    stats = dict(summary)
    stats.update(memory_summary)
    stats.update(cuda_peak_memory_stats())
    write_training_stats(stats, output_path)
    return stats


def write_training_stats(stats: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser() / "train_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **stats,
        "recorded_at_unix_s": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_training_stats_for_checkpoint(
    checkpoint: str | Path | None,
    metrics_output: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find train_stats.json near a checkpoint or metrics output path."""
    candidates: list[Path] = []
    if checkpoint is not None:
        ckpt = Path(checkpoint).expanduser()
        if ckpt.is_dir():
            candidates.extend([ckpt / "train_stats.json", ckpt.parent / "train_stats.json"])
            if ckpt.parent.name.startswith("iteration_"):
                candidates.append(ckpt.parent.parent.parent / "train_stats.json")
            elif ckpt.name.startswith("iteration_"):
                candidates.append(ckpt.parent.parent / "train_stats.json")
        else:
            candidates.extend([ckpt.parent / "train_stats.json", ckpt.parent.parent / "train_stats.json"])
            if ckpt.parent.name.startswith("iteration_"):
                candidates.append(ckpt.parent.parent.parent / "train_stats.json")
    if metrics_output is not None:
        candidates.append(Path(metrics_output).expanduser().parent / "train_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None
