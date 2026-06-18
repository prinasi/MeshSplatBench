"""Build helpers for TriangleBench's bundled CUDA renderer modules."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CSRC_ROOT = PACKAGE_ROOT / "vendor" / "_csrc"
BUILD_JOBS = os.getenv("MAX_BUILD_JOBS")

# When set, only build the listed backends (comma-separated canonical names).
# e.g.  TRIANGLEBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn
_ONLY_BACKENDS = os.getenv("TRIANGLEBENCH_CUDA_BACKENDS")

# Enable ccache for NVCC.  Requires ``ccache`` on PATH.
_USE_CCACHE = os.getenv("TRIANGLEBENCH_CCACHE", "").lower() in ("1", "true", "yes")


class ParallelBuildExt(BuildExtension):
    """Build CUDA extensions in isolated temporary directories."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parallel = True if BUILD_JOBS is None else int(BUILD_JOBS)

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
            return

        self_copy = copy.copy(self)
        self_copy.build_temp = os.path.join(self.build_temp, ext.name)
        BuildExtension.build_extension(self_copy, ext)

    def build_cmake_extension(self, ext: "CMakeExtension") -> None:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError(
                "Building TriangleBench's bundled DiffSoup extension requires "
                "CMake. Install it with `pip install cmake` or use build "
                "isolation so pyproject.toml can install `cmake>=3.20`."
            )
        try:
            nanobind_dir = subprocess.check_output(
                [sys.executable, "-m", "nanobind", "--cmake_dir"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Building TriangleBench's bundled DiffSoup extension requires "
                "nanobind in the Python environment used for the build. Install "
                "it with `pip install nanobind`, or build without "
                "`--no-build-isolation` so pyproject.toml can install "
                "`nanobind>=2.11.0`."
            ) from exc

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp, ext.name).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir.mkdir(parents=True, exist_ok=True)

        cuda_archs = os.getenv("TRIANGLEBENCH_CUDA_ARCHS", "89")
        configure_args = [
            cmake,
            "-S",
            str(ext.sourcedir),
            "-B",
            str(build_temp),
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
            f"-Dnanobind_DIR={nanobind_dir}",
        ]
        build_args = [cmake, "--build", str(build_temp), "--config", cfg]
        if BUILD_JOBS is not None:
            build_args.extend(["--parallel", str(BUILD_JOBS)])

        subprocess.check_call(configure_args)
        subprocess.check_call(build_args)


class CMakeExtension(Extension):
    """A setuptools extension built by CMake instead of distutils."""

    def __init__(self, name: str, sourcedir: str | Path):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


def _rel_sources(root: Path, files: list[str]) -> list[str]:
    return [
        (root / file).resolve().relative_to(PROJECT_ROOT).as_posix()
        for file in files
    ]


def _ccache_nvcc_args() -> list[str]:
    """Return extra nvcc args that route compilation through ccache."""
    if not _USE_CCACHE:
        return []
    if not shutil.which("ccache"):
        print("[TriangleBench] TRIANGLEBENCH_CCACHE=1 but ccache not found on PATH — ignoring.")
        return []
    # The most portable way: tell nvcc to use ccache as its host compiler wrapper.
    return []  # ccache is activated via the CUDAHOSTCXX env var instead.


def _should_build(tag: str) -> bool:
    """Check whether a backend extension should be included in the build."""
    if _ONLY_BACKENDS is None:
        return True
    wanted = {s.strip().lower().replace("_", "-") for s in _ONLY_BACKENDS.split(",")}
    return tag.lower().replace("_", "-") in wanted


def get_ext_modules() -> list[Extension]:
    """Return all bundled CUDA/CMake extension modules.

    Environment variables that affect the build:

    ``TRIANGLEBENCH_CUDA_BACKENDS``
        Comma-separated list of backend names to build.  If set, only the
        listed backends are compiled (all others are skipped).  Names are
        matched case-insensitively with underscores and hyphens treated as
        equivalent.  Example::

            TRIANGLEBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn

    ``TRIANGLEBENCH_CCACHE``
        Set to ``1`` to route NVCC through *ccache*.  Requires *ccache* on
        ``PATH`` and activates by setting ``CUDAHOSTCXX=ccache`` before
        invoking the compiler.

    ``TRIANGLEBENCH_CUDA_ARCHS``
        Semicolon-separated CUDA architectures (default ``89``).
    """
    # Activate ccache for NVCC if requested (must happen before any build).
    if _USE_CCACHE and shutil.which("ccache"):
        os.environ.setdefault("CUDAHOSTCXX", "ccache")

    tri_root = CSRC_ROOT / "triangle_splatting_rasterization"
    mesh_root = CSRC_ROOT / "mesh_splatting_rasterization"
    d2ts_root = CSRC_ROOT / "d2ts_rasterization"
    knn_root = CSRC_ROOT / "simple_knn"
    diffsoup_root = CSRC_ROOT / "diffsoup_core"

    glm_include = str((tri_root / "third_party" / "glm").resolve())
    mesh_glm_include = str((mesh_root / "third_party" / "glm").resolve())

    extensions: list[Extension] = []

    # -- triangle-splatting rasterization -----------------------------
    if _should_build("triangle-splatting"):
        extensions.append(CUDAExtension(
            name="trianglebench.vendor._cmod.triangle_splatting_rasterization._C",
            sources=_rel_sources(
                tri_root,
                [
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "cuda_rasterizer/utils.cu",
                    "rasterize_points.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [f"-I{glm_include}", "--use_fast_math"]},
        ))

    # -- mesh-splatting rasterization ---------------------------------
    if _should_build("mesh-splatting"):
        extensions.append(CUDAExtension(
            name="trianglebench.vendor._cmod.mesh_splatting_rasterization._C",
            sources=_rel_sources(
                mesh_root,
                [
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "cuda_rasterizer/utils.cu",
                    "cuda_rasterizer/adam.cu",
                    "rasterize_points.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [f"-I{mesh_glm_include}", "--use_fast_math"]},
        ))

    # -- 2DTS rasterization ------------------------------------------
    if _should_build("2dts"):
        extensions.append(CUDAExtension(
            name="trianglebench.vendor._cmod.d2ts_rasterization._C",
            sources=_rel_sources(
                d2ts_root,
                [
                    "src/rasterizer.cu",
                    "src/forward.cu",
                    "src/backward.cu",
                    "src/extension_interface.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": []},
        ))

    # -- simple-knn ---------------------------------------------------
    if _should_build("simple-knn"):
        extensions.append(CUDAExtension(
            name="trianglebench.vendor._cmod.simple_knn._C",
            sources=_rel_sources(
                knn_root,
                [
                    "interface.cu",
                    "simple_knn.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [], "cxx": []},
        ))

    # -- DiffSoup (CMake) ---------------------------------------------
    if _should_build("diffsoup"):
        extensions.append(CMakeExtension(
            "trianglebench.vendor.diffsoup.diffsoup._core",
            diffsoup_root,
        ))

    if _ONLY_BACKENDS and not extensions:
        print(
            f"[TriangleBench] TRIANGLEBENCH_CUDA_BACKENDS='{_ONLY_BACKENDS}' "
            "did not match any known backend.  Available: "
            "triangle-splatting, mesh-splatting, 2dts, simple-knn, diffsoup"
        )

    return extensions
