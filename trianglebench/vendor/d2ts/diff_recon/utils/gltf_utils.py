import base64
import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

MATRIX_DIMS = {
    "MAT2": 2,
    "MAT3": 3,
    "MAT4": 4,
}


def load_glb_chunks(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        magic, version, length = struct.unpack("<III", handle.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a GLB file")
        if version != 2:
            raise ValueError(f"Unsupported GLB version: {version}")

        json_chunk = None
        bin_chunk = b""
        while handle.tell() < length:
            chunk_len, chunk_type = struct.unpack("<II", handle.read(8))
            chunk_data = handle.read(chunk_len)
            if chunk_type == JSON_CHUNK_TYPE:
                json_chunk = json.loads(chunk_data.decode("utf-8").rstrip(" \t\r\n\x00"))
            elif chunk_type == BIN_CHUNK_TYPE:
                bin_chunk = chunk_data

    if json_chunk is None:
        raise ValueError(f"{path} does not contain a JSON chunk")
    return json_chunk, bin_chunk


def _apply_normalization(values: np.ndarray, component_type: int) -> np.ndarray:
    if component_type == 5120:
        return np.maximum(values.astype(np.float32) / 127.0, -1.0)
    if component_type == 5121:
        return values.astype(np.float32) / 255.0
    if component_type == 5122:
        return np.maximum(values.astype(np.float32) / 32767.0, -1.0)
    if component_type == 5123:
        return values.astype(np.float32) / 65535.0
    if component_type == 5125:
        return values.astype(np.float32) / 4294967295.0
    return values.astype(np.float32)


def _read_strided_buffer(data: bytes, dtype: np.dtype, count: int, num_components: int, offset: int, stride_bytes: int) -> np.ndarray:
    values = np.empty((count, num_components), dtype=dtype)
    for row_idx in range(count):
        row_offset = offset + row_idx * stride_bytes
        values[row_idx] = np.frombuffer(data, dtype=dtype, count=num_components, offset=row_offset)
    return values


def _reshape_accessor_values(accessor_type: str, values: np.ndarray) -> np.ndarray:
    if accessor_type not in MATRIX_DIMS:
        return values

    dim = MATRIX_DIMS[accessor_type]
    return values.reshape(-1, dim, dim).transpose(0, 2, 1)


def read_accessor(gltf: dict, bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    component_type = accessor["componentType"]
    dtype = np.dtype(COMPONENT_DTYPE[component_type])
    accessor_type = accessor["type"]
    num_components = TYPE_COMPONENTS[accessor_type]
    count = accessor["count"]

    if "bufferView" in accessor:
        buffer_view = gltf["bufferViews"][accessor["bufferView"]]
        offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride_bytes = buffer_view.get("byteStride")
        tight_stride = dtype.itemsize * num_components
        if stride_bytes is None or stride_bytes == tight_stride:
            values = np.frombuffer(bin_chunk, dtype=dtype, count=count * num_components, offset=offset).reshape(count, num_components)
        else:
            values = _read_strided_buffer(bin_chunk, dtype, count, num_components, offset, stride_bytes)
    else:
        values = np.zeros((count, num_components), dtype=dtype)

    if accessor.get("sparse") is not None:
        sparse = accessor["sparse"]
        sparse_count = sparse["count"]

        indices_info = sparse["indices"]
        indices_view = gltf["bufferViews"][indices_info["bufferView"]]
        indices_offset = indices_view.get("byteOffset", 0) + indices_info.get("byteOffset", 0)
        indices_dtype = np.dtype(COMPONENT_DTYPE[indices_info["componentType"]])
        indices = np.frombuffer(bin_chunk, dtype=indices_dtype, count=sparse_count, offset=indices_offset)

        values_info = sparse["values"]
        values_view = gltf["bufferViews"][values_info["bufferView"]]
        values_offset = values_view.get("byteOffset", 0) + values_info.get("byteOffset", 0)
        sparse_values = np.frombuffer(bin_chunk, dtype=dtype, count=sparse_count * num_components, offset=values_offset).reshape(sparse_count, num_components)

        values = values.copy()
        values[indices] = sparse_values

    if accessor.get("normalized", False):
        values = _apply_normalization(values, component_type)

    return _reshape_accessor_values(accessor_type, values)


def load_image_from_gltf(image_def: dict, glb_path: Path, bin_chunk: bytes, gltf: dict) -> np.ndarray:
    if "uri" in image_def:
        uri = image_def["uri"]
        if uri.startswith("data:"):
            encoded = uri.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = (glb_path.parent / uri).read_bytes()
    else:
        buffer_view = gltf["bufferViews"][image_def["bufferView"]]
        offset = buffer_view.get("byteOffset", 0)
        image_bytes = bin_chunk[offset : offset + buffer_view["byteLength"]]

    with Image.open(BytesIO(image_bytes)) as image:
        # Match trimesh's texture convention so static and animated loaders
        # feed the renderer the same image orientation.
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        return np.ascontiguousarray(rgba[::-1])


def ensure_rgba(image: np.ndarray | None, fill: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> np.ndarray:
    if image is None:
        return np.asarray(fill, dtype=np.float32).reshape(1, 1, 4)

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {image.shape}")
    if image.shape[2] == 4:
        return np.clip(image, 0.0, 1.0)
    if image.shape[2] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        return np.clip(np.concatenate([image, alpha], axis=2), 0.0, 1.0)
    raise ValueError(f"Unsupported image channel count: {image.shape[2]}")


def build_texture_atlas(textures: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    if len(textures) == 0:
        atlas = np.ones((1, 1, 4), dtype=np.float32)
        return atlas, []

    textures_rgba = [ensure_rgba(texture) for texture in textures]
    atlas_height = max(texture.shape[0] for texture in textures_rgba)
    atlas_width = sum(texture.shape[1] for texture in textures_rgba)
    atlas = np.zeros((atlas_height, atlas_width, 4), dtype=np.float32)
    transforms = []
    atlas_width_denom = max(atlas_width - 1, 1)
    atlas_height_denom = max(atlas_height - 1, 1)

    offset_x = 0
    for texture in textures_rgba:
        height, width = texture.shape[:2]
        offset_y = atlas_height - height
        atlas[offset_y : offset_y + height, offset_x : offset_x + width] = texture

        # The rasterizers sample UVs as x = u * (W - 1), y = (1 - v) * (H - 1).
        # Atlas remapping therefore needs to preserve pixel-center coordinates,
        # not just normalized tile extents, or samples bleed into adjacent tiles.
        scale_u = 0.0 if width <= 1 else (width - 1) / atlas_width_denom
        scale_v = 0.0 if height <= 1 else (height - 1) / atlas_height_denom
        offset_u = offset_x / atlas_width_denom
        offset_v = 1.0 - (offset_y + height - 1) / atlas_height_denom if atlas_height > 1 else 0.0
        transforms.append(np.asarray([scale_u, scale_v, offset_u, offset_v], dtype=np.float32))
        offset_x += width

    return atlas, transforms