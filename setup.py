from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
BUILD_HELPER = ROOT / "trianglebench" / "vendor" / "build.py"


def _should_build_cuda() -> bool:
    """Decide whether to compile CUDA extensions.

    Decision priority:
      1. ``TRIANGLEBENCH_SKIP_CUDA=1``  → always skip (fastest daily dev).
      2. ``TRIANGLEBENCH_BUILD_CUDA=1`` → always build (explicit opt-in).
      3. Auto-detect ``CUDA_HOME`` / ``CUDA_PATH`` / ``nvcc`` on PATH.
    """
    skip = os.environ.get("TRIANGLEBENCH_SKIP_CUDA", "").lower()
    if skip in ("1", "true", "yes"):
        return False

    force = os.environ.get("TRIANGLEBENCH_BUILD_CUDA", "").lower()
    if force in ("1", "true", "yes"):
        return True

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return True
    return shutil.which("nvcc") is not None


kwargs: dict = dict(
    packages=find_packages(include=["trianglebench", "trianglebench.*"]),
)

if _should_build_cuda():
    spec = importlib.util.spec_from_file_location("trianglebench_vendor_build", BUILD_HELPER)
    build = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build)
    kwargs["ext_modules"] = build.get_ext_modules()
    kwargs["cmdclass"] = {"build_ext": build.ParallelBuildExt}

    backends = os.environ.get("TRIANGLEBENCH_CUDA_BACKENDS", "all")
    ccache = "yes" if os.environ.get("TRIANGLEBENCH_CCACHE", "").lower() in ("1", "true", "yes") else "no"
    print(
        f"[TriangleBench] Building CUDA extensions (backends={backends}, ccache={ccache}).\n"
        "  Tip: set TRIANGLEBENCH_SKIP_CUDA=1 to skip CUDA compilation for faster\n"
        "       editable installs when you only changed Python code."
    )
else:
    print(
        "[TriangleBench] Skipping CUDA extension build.\n"
        "  Python-only toolkit installed. To build CUDA extensions:\n"
        "    export CUDA_HOME=/usr/local/cuda\n"
        "    python setup.py build_ext --inplace     # build & install in-place\n"
        "    # or:\n"
        "    TRIANGLEBENCH_BUILD_CUDA=1 pip install -e '.[cuda]'\n"
        "\n"
        "  Speed tips:\n"
        "    TRIANGLEBENCH_CCACHE=1           # cache NVCC output (needs ccache)\n"
        "    TRIANGLEBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn  # build subset\n"
        "    MAX_BUILD_JOBS=8                 # parallel compilation\n"
    )

setup(**kwargs)
