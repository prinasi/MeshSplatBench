"""Checkpoint discovery helpers for training methods."""

from __future__ import annotations

import re
from pathlib import Path


_ITERATION_RE = re.compile(r"^iteration_(\d+)$")


def find_latest_point_cloud_checkpoint(output_dir: str | Path) -> tuple[int, Path] | None:
    """Return the latest complete point-cloud checkpoint under ``output_dir``."""
    point_cloud_dir = Path(output_dir).expanduser() / "point_cloud"
    if not point_cloud_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in point_cloud_dir.iterdir():
        if not path.is_dir():
            continue
        match = _ITERATION_RE.match(path.name)
        if match is None:
            continue
        if not (path / "point_cloud_state_dict.pt").is_file():
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])
