#!/usr/bin/env python3
"""Validate a TriAsset package without CUDA, Unity, or PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DTYPES = {"float32": np.dtype("<f4"), "int32": np.dtype("<i4"), "uint32": np.dtype("<u4")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _buffers(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    buffers = manifest.get("buffers")
    if not isinstance(buffers, dict):
        raise ValueError("manifest.buffers must be an object")
    return buffers


def _memmap(root: Path, spec: dict[str, Any]) -> np.memmap:
    dtype_name = spec.get("dtype")
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported buffer dtype {dtype_name!r}")
    return np.memmap(root / spec["file"], dtype=DTYPES[dtype_name], mode="r").reshape(spec["shape"])


def validate_triasset(
    root: str | Path,
    *,
    max_faces: int = 1_000_000,
    verify_checkpoint_hash: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError(f"unsupported schema_version {manifest.get('schema_version')!r}")
    if int(manifest.get("export_contract_revision", 0)) < 2:
        raise ValueError("stale export contract revision; re-export with the current TriBench exporter")

    buffers = _buffers(manifest)
    buffer_list = manifest.get("buffer_list")
    if not isinstance(buffer_list, list) or {entry.get("name") for entry in buffer_list} != set(buffers):
        raise ValueError("manifest.buffer_list must contain exactly the declared buffer names")
    checked_bytes = 0
    for name, spec in buffers.items():
        dtype_name = spec.get("dtype")
        if dtype_name not in DTYPES:
            raise ValueError(f"buffer {name} has unsupported dtype {dtype_name!r}")
        expected = int(np.prod(spec["shape"], dtype=np.int64)) * DTYPES[dtype_name].itemsize
        actual = (root / spec["file"]).stat().st_size
        if actual != expected:
            raise ValueError(f"buffer {name} has {actual} bytes, expected {expected}")
        checked_bytes += actual

    result: dict[str, Any] = {
        "asset": str(root),
        "method": manifest.get("method"),
        "buffers": len(buffers),
        "checked_bytes": checked_bytes,
    }
    rendering = manifest.get("rendering") or {}
    if rendering.get("unity_profile_name") != "method-aware":
        raise ValueError("manifest must label the specialized Unity condition as method-aware")
    if rendering.get("cuda_equivalent") is not False:
        raise ValueError("manifest must explicitly set cuda_equivalent=false")
    if rendering.get("background_color") not in {"black", "white"}:
        raise ValueError("manifest must declare a deterministic black or white background_color")
    if not isinstance(rendering.get("unity_method_aware_requires_per_camera_depth_sort"), bool):
        raise ValueError("manifest must declare the Unity method-aware per-camera sorting requirement")
    if not isinstance(rendering.get("unity_method_aware_status"), str):
        raise ValueError("manifest must declare unity_method_aware_status")
    general = rendering.get("general_purpose")
    if not isinstance(general, dict) or not isinstance(general.get("supported"), bool):
        raise ValueError("manifest must declare general_purpose.supported")
    if general["supported"] and "sh_dc" not in buffers:
        raise ValueError("general-purpose SH-DC condition requires the sh_dc buffer")
    if manifest.get("method") == "diffsoup" and general["supported"]:
        raise ValueError("DiffSoup general-purpose rendering requires an explicit appearance bake")
    if manifest.get("method") == "mesh-splatting":
        for required in ("indices", "vertex_weight_logits", "triangle_opacity"):
            if required not in buffers:
                raise ValueError(
                    f"mesh-splatting package is stale or incomplete: missing {required!r}; re-export it"
                )
        faces = _memmap(root, buffers["indices"])
        logits = _memmap(root, buffers["vertex_weight_logits"]).reshape(-1)
        exported = _memmap(root, buffers["triangle_opacity"]).reshape(-1)
        count = len(faces) if max_faces <= 0 else min(len(faces), max_faces)
        face_sample = np.asarray(faces[:count], dtype=np.int64)
        if face_sample.size and (face_sample.min() < 0 or face_sample.max() >= len(logits)):
            raise ValueError("indices reference an invalid vertex opacity")
        floor = float(rendering.get("opacity_floor", 0.0))
        activated = floor + (1.0 - floor) / (1.0 + np.exp(-np.asarray(logits[face_sample])))
        expected = activated.min(axis=1)
        error = np.abs(np.asarray(exported[:count]) - expected)
        max_error = float(error.max()) if error.size else 0.0
        if max_error > 2e-7:
            raise ValueError(f"triangle_opacity does not match min-after-activation; max error={max_error}")
        result["mesh_opacity"] = {
            "faces_checked": count,
            "opacity_floor": floor,
            "max_abs_error": max_error,
            "reduction": rendering.get("triangle_opacity_reduction"),
        }

    if manifest.get("method") == "2dts" and rendering.get("gamma_rescale"):
        gamma = float(rendering["gamma"])
        beta = 1.0 / gamma
        expected = 1.0 / math.sqrt((2.0**beta) * beta * math.gamma(beta))
        actual = float(rendering.get("gamma_vertex_rescale", float("nan")))
        if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"gamma_vertex_rescale is {actual}, expected {expected}")
        result["d2ts_gamma_rescale"] = actual

    if verify_checkpoint_hash:
        source = manifest.get("source") or {}
        checkpoint = Path(source.get("checkpoint", ""))
        if not checkpoint.is_file():
            raise ValueError(f"source checkpoint is unavailable: {checkpoint}")
        actual_hash = _sha256(checkpoint)
        if actual_hash != source.get("checkpoint_sha256"):
            raise ValueError("source checkpoint SHA-256 does not match manifest")
        result["checkpoint_sha256_verified"] = True

    result["status"] = "ok"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path, help="Path to a .triasset directory")
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1_000_000,
        help="Maximum MeshSplatting faces to verify; use 0 for all",
    )
    parser.add_argument("--verify-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_triasset(
            args.asset,
            max_faces=args.max_faces,
            verify_checkpoint_hash=args.verify_checkpoint_hash,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
