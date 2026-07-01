"""Conftest at project root — ensures tribench package is importable for tests."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
