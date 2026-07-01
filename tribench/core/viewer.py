"""NerfBaselines-compatible interactive viewer for TriBench models.

Serves the nerfbaselines Three.js frontend with remote rendering backed by
any TriBench RendererAdapter. Supports:

- Free-viewpoint remote rendering (HTTP POST /render + WebSocket /render-websocket)
- Train/test camera frustums with click-to-navigate
- Dual point cloud PLY preview and geometry endpoints
- color/depth/alpha/normal output switching with colormap and split compare
- Auto-fallback to DummyRenderer when CUDA is unavailable (macOS)
"""

from __future__ import annotations

import errno
import io
import json
import math
import os
import shutil
import socket
import struct
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from tribench.core.cameras import CameraBatch
from tribench.core.datasets import DatasetBase, DatasetSample
from tribench.core.rendering import tensor_to_uint8
from tribench.renderers.base import RenderOutput, RendererAdapter

# ─── Constants ────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "viewer_static"
_PALETTES_JSON = Path(__file__).parent / "palettes.json"
_THUMBNAIL_SIZE = 96

_METHOD_OUTPUTS: dict[str, tuple[str, ...]] = {
    "triangle-splatting": ("color", "depth", "alpha", "normal"),
    "mesh-splatting": ("color", "depth", "alpha", "normal"),
    "2dts": ("color", "depth", "alpha", "normal"),
    "diffsoup": ("color", "depth", "alpha"),
    "dummy": ("color", "depth", "alpha", "normal"),
}


# ─── Errors ───────────────────────────────────────────────────────────────


class ViewerError(Exception):
    """Base error for viewer requests."""


class ViewerNotFound(ViewerError):
    """Requested viewer resource was not found."""


class ViewerBadRequest(ViewerError):
    """Viewer request payload is invalid."""


# ─── Backend ──────────────────────────────────────────────────────────────


class _ViewerBackend:
    """Holds adapter, datasets, geometry, and serves viewer data."""

    def __init__(
        self,
        adapter: RendererAdapter,
        *,
        train_dataset: DatasetBase | None = None,
        test_dataset: DatasetBase | None = None,
        primary_split: str = "test",
        device: str | torch.device | None = None,
        max_render_size: int = 1280,
        jpeg_quality: int = 90,
        geometry_path: str | Path | None = None,
        cg_geometry_path: str | Path | None = None,
        mesh_geometry_path: str | Path | None = None,
        glb_geometry_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        mesh_preview: bool = False,
    ) -> None:
        self.adapter = adapter
        self.datasets: dict[str, DatasetBase | None] = {
            "train": train_dataset,
            "test": test_dataset,
        }
        self.primary_split = primary_split
        self.device = device
        self.max_render_size = int(max_render_size)
        self.jpeg_quality = int(jpeg_quality)
        self._image_cache: dict[str, tuple[bytes, str]] = {}
        self._render_lock = threading.Lock()
        self._info: dict[str, Any] | None = None
        self._dataset_json: dict[str, Any] | None = None

        # Geometry paths
        self.geometry_path = Path(geometry_path) if geometry_path else None
        self.cg_geometry_path = Path(cg_geometry_path) if cg_geometry_path else None
        self.mesh_geometry_path = Path(mesh_geometry_path) if mesh_geometry_path else None
        self.glb_geometry_path = Path(glb_geometry_path) if glb_geometry_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.mesh_preview = mesh_preview

    def get_info(self) -> dict[str, Any]:
        if self._info is None:
            self._info = _build_viewer_params(
                self.adapter,
                train_dataset=self.datasets.get("train"),
                test_dataset=self.datasets.get("test"),
                primary_split=self.primary_split,
                max_render_size=self.max_render_size,
                has_pointcloud=self.geometry_path is not None,
                mesh_preview=self.mesh_preview and self.mesh_geometry_path is not None,
                mesh_url="./dataset/geometry/mesh.ply" if self.mesh_geometry_path else None,
            )
        return self._info

    def get_dataset_json(self) -> dict[str, Any]:
        if self._dataset_json is None:
            self._dataset_json = _build_dataset_json(self.datasets)
        return self._dataset_json

    def get_dataset_image(self, split: str, idx: int, *, thumb: bool = False) -> tuple[bytes, str]:
        dataset = self.datasets.get(split)
        if dataset is None:
            raise ViewerNotFound(f"Split '{split}' not found")
        idx = _clamp_index(idx, len(dataset))
        cache_key = f"{split}/{idx}/{thumb}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]

        sample = dataset.sample(idx)
        image = _sample_pil_image(sample)
        if thumb:
            image.thumbnail((_THUMBNAIL_SIZE, _THUMBNAIL_SIZE), _RESAMPLE_BILINEAR)

        with io.BytesIO() as buf:
            image.save(buf, format="JPEG", quality=86)
            payload = buf.getvalue()
        result = (payload, "image/jpeg")
        if thumb:
            self._image_cache[cache_key] = result
        return result

    def get_pointcloud(self) -> io.BytesIO:
        if self.geometry_path is None or not self.geometry_path.exists():
            raise ViewerNotFound("No point cloud available")
        data = self.geometry_path.read_bytes()
        buf = io.BytesIO(data)
        return buf

    def get_geometry_file(self, name: str) -> tuple[bytes, str]:
        """Serve extended geometry files."""
        path_map = {
            "viewer_points.ply": self.geometry_path,
            "cg_points.ply": self.cg_geometry_path,
            "mesh.ply": self.mesh_geometry_path,
            "mesh.glb": self.glb_geometry_path,
            "metadata.json": self.metadata_path,
        }
        p = path_map.get(name)
        if p is None or not p.exists():
            raise ViewerNotFound(f"Geometry file '{name}' not found")
        ct = "application/octet-stream"
        if name.endswith(".json"):
            ct = "application/json"
        return p.read_bytes(), ct

    def render(self, request: dict[str, Any]) -> tuple[bytes, str]:
        with self._render_lock:
            return _render_viewer_frame(
                self.adapter,
                request,
                datasets=self.datasets,
                device=self.device,
                max_render_size=self.max_render_size,
                jpeg_quality=self.jpeg_quality,
            )

    def handle_websocket_message(self, data: str) -> bytes:
        """Process a WebSocket render message and return binary response."""
        thread = None
        try:
            req = json.loads(data)
            thread = req.pop("thread", None)
            payload, mimetype = self.render(req)
            header = json.dumps({
                "status": "ok",
                "thread": thread,
                "mimetype": mimetype,
            }).encode("utf-8")
            return struct.pack("!I", len(header)) + header + payload
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "thread": thread,
            }).encode("utf-8")


# ─── Viewer params ────────────────────────────────────────────────────────


def _build_viewer_params(
    adapter: RendererAdapter,
    *,
    train_dataset: DatasetBase | None,
    test_dataset: DatasetBase | None,
    primary_split: str,
    max_render_size: int,
    has_pointcloud: bool,
    mesh_preview: bool = False,
    mesh_url: str | None = None,
) -> dict[str, Any]:
    """Build the JSON blob injected into index.html as viewer params."""

    method = str(getattr(adapter, "name", adapter.__class__.__name__))
    output_types = list(_METHOD_OUTPUTS.get(method, ("color", "depth", "alpha", "normal")))

    # Scene info from whichever dataset is available
    ref_dataset = test_dataset or train_dataset
    scene = _scene_from_dataset(ref_dataset) if ref_dataset is not None else {
        "center": [0, 0, 0], "radius": 1.0, "up": [0, -1, 0]
    }

    default_res = 768 if max_render_size <= 0 else min(max(int(max_render_size), 1), 1280)

    # Build outputs_configuration
    outputs_cfg: dict[str, Any] = {}
    for ot in output_types:
        if ot == "depth":
            outputs_cfg[ot] = {"palette_enabled": True}
        elif ot in ("color", "normal"):
            outputs_cfg[ot] = {}
        elif ot in ("alpha", "accumulation"):
            outputs_cfg[ot] = {"palette_enabled": True, "range_min": 0, "range_max": 1}
        else:
            outputs_cfg[ot] = {"palette_enabled": True}

    # Renderer: mesh preview mode or remote rendering
    if mesh_preview and mesh_url:
        renderer: dict[str, Any] = {
            "type": "mesh",
            "mesh_url": mesh_url,
        }
    else:
        renderer = {
            "type": "remote",
            "websocket_url": "./render-websocket",
            "http_url": "./render",
            "output_types": output_types,
        }

    info: dict[str, Any] = {
        "renderer": renderer,
        "dataset": {
            "url": "./dataset.json",
        },
        "state": {
            "render_resolution": default_res,
            "prerender_enabled": False,
            "camera_control_mode": "orbit",
            "output_types": output_types,
            "output_type": output_types[0] if output_types else "color",
            "split_output_type": output_types[0] if output_types else "color",
            "camera_path_render_output_type": output_types[0] if output_types else "color",
            "outputs_configuration": outputs_cfg,
            "method_info": _safe_model_stats(adapter),
            "dataset_info": {
                "primary_split": primary_split,
            },
        },
    }

    if has_pointcloud:
        info["dataset"]["pointcloud_url"] = "./dataset/pointcloud.ply"

    # Scene metadata
    if ref_dataset is not None:
        info["state"]["dataset_info"].update({
            "type": "object-centric",
            "scene_center": scene["center"],
            "scene_radius": scene["radius"],
            "up": scene["up"],
        })
        info["viewer_initial_target"] = scene["center"]
        info["viewer_up"] = scene["up"]

    # Viewer transform / initial pose from scene
    if ref_dataset is not None and len(ref_dataset) > 0:
        cam = ref_dataset.sample(0).camera
        c2w = cam.camtoworlds[0].detach().cpu().numpy()
        info["viewer_initial_pose"] = c2w[:3, :4].flatten().tolist()

    return info


def _build_dataset_json(datasets: dict[str, DatasetBase | None]) -> dict[str, Any]:
    """Build nerfbaselines-compatible dataset.json with train/test splits."""

    result: dict[str, Any] = {}
    for split, ds in datasets.items():
        if ds is None:
            result[split] = None
            continue

        cameras = []
        for idx in range(len(ds)):
            sample = ds.sample(idx)
            camera = sample.camera
            metadata = sample.metadata or camera.metadata or {}
            image_name = str(metadata.get("image_name") or sample.name)

            # Intrinsics as [fx, fy, cx, cy] (nerfbaselines frontend convention)
            K = camera.Ks[0].detach().cpu().numpy()
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            cameras.append({
                "pose": _matrix_to_list(camera.camtoworlds[0][:3, :4]),
                "intrinsics": [fx, fy, cx, cy],
                "image_size": [int(camera.width), int(camera.height)],
                "near": float(camera.near),
                "far": float(camera.far),
                "image_name": image_name,
                "image_url": f"./dataset/images/{split}/{idx}.jpg",
                "thumbnail_url": f"./dataset/images/{split}/{idx}.jpg?thumb=1",
            })
        result[split] = {"cameras": cameras}

    # Scene metadata
    ref_ds = datasets.get("test") or datasets.get("train")
    if ref_ds is not None and len(ref_ds) > 0:
        scene = _scene_from_dataset(ref_ds)
        result["metadata"] = {
            "scene_center": scene["center"],
            "scene_radius": scene["radius"],
            "up": scene["up"],
        }

    return result


# ─── Serve ────────────────────────────────────────────────────────────────


def serve_viewer(
    adapter: RendererAdapter,
    dataset: DatasetBase | None = None,
    *,
    train_dataset: DatasetBase | None = None,
    test_dataset: DatasetBase | None = None,
    host: str = "127.0.0.1",
    port: int = 7007,
    device: str | torch.device | None = None,
    max_render_size: int = 1280,
    jpeg_quality: int = 90,
    geometry_path: str | Path | None = None,
    cg_geometry_path: str | Path | None = None,
    mesh_geometry_path: str | Path | None = None,
    glb_geometry_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    mesh_preview: bool = False,
) -> str:
    """Serve an interactive nerfbaselines-compatible browser viewer.

    Args:
        adapter: RendererAdapter for remote rendering.
        dataset: Primary dataset (backward compat — becomes test or train).
        train_dataset: Explicit train split (overrides dataset).
        test_dataset: Explicit test split (overrides dataset).
        geometry_path: Path to viewer point cloud PLY.
        cg_geometry_path: Path to high-fidelity CG point cloud PLY.
        mesh_geometry_path: Path to optional mesh PLY.
        glb_geometry_path: Path to optional mesh GLB.
        metadata_path: Path to geometry_metadata.json.
        mesh_preview: If True and mesh_geometry_path is set, use mesh local renderer.
    """
    # Resolve datasets
    if train_dataset is None and test_dataset is None and dataset is not None:
        # Backward compat: single dataset goes to primary split
        test_dataset = dataset

    primary_split = "test" if test_dataset is not None else "train"

    backend = _ViewerBackend(
        adapter,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        primary_split=primary_split,
        device=device,
        max_render_size=max_render_size,
        jpeg_quality=jpeg_quality,
        geometry_path=geometry_path,
        cg_geometry_path=cg_geometry_path,
        mesh_geometry_path=mesh_geometry_path,
        glb_geometry_path=glb_geometry_path,
        metadata_path=metadata_path,
        mesh_preview=mesh_preview,
    )
    handler = partial(_ViewerRequestHandler, backend=backend)
    server = _bind_server(host, int(port), handler)
    actual_port = server.server_address[1]
    url = f"http://{_display_host(host, actual_port)}"
    print(f"TriBench viewer running at {url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return url


# ─── HTTP handler ─────────────────────────────────────────────────────────


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving nerfbaselines-compatible routes + static files."""

    def __init__(self, *args: Any, backend: _ViewerBackend, **kwargs: Any) -> None:
        self.backend = backend
        # Serve static files from viewer_static/
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def list_directory(self, path: str) -> Any:  # noqa: A002
        # Disable directory listing
        self.send_error(404, "Not Found")
        return None

    def send_header(self, keyword: str, value: str | int) -> None:  # type: ignore[override]
        if keyword == "Content-type" and str(value) == "text/javascript":
            super().send_header("Cross-Origin-Opener-Policy", "same-origin")
            super().send_header("Cross-Origin-Embedder-Policy", "require-corp")
        return super().send_header(keyword, value)

    def end_headers(self) -> None:  # noqa: N802
        self.send_header("Cache-Control", "no-store")
        return super().end_headers()

    # ── GET routes ──

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/":
                return self._handle_index()
            if path == "/dataset.json":
                return self._send_json(self.backend.get_dataset_json())
            if path == "/palettes.js":
                return self._handle_palettes_js()
            if path == "/3dgs.js":
                raise ViewerNotFound("TriBench viewer does not serve the 3DGS local renderer")
            if path == "/dataset/pointcloud.ply":
                return self._handle_pointcloud()
            if path.startswith("/dataset/geometry/"):
                name = path[len("/dataset/geometry/"):]
                return self._handle_geometry_file(name)
            if path.startswith("/dataset/images/") and path.endswith(".jpg"):
                split, idx = _parse_image_path(path)
                thumb = query.get("thumb", ["0"])[0] == "1"
                payload, mimetype = self.backend.get_dataset_image(split, idx, thumb=thumb)
                return self._send(payload, mimetype)
            if path == "/render-websocket":
                return self._handle_websocket()

            # Fall through to static file serving (viewer.js, controls.js, etc.)
            return super().do_GET()

        except Exception as exc:
            self._send_error(exc)

    # ── POST routes ──

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/render":
                raise ViewerNotFound("Not found")
            try:
                content_length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(content_length)
                request = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception as exc:
                raise ViewerBadRequest("Invalid JSON request") from exc
            if not isinstance(request, dict):
                raise ViewerBadRequest("Render request must be a JSON object")
            payload, mimetype = self.backend.render(request)
            self._send(payload, mimetype)
        except Exception as exc:
            self._send_error(exc)

    # ── Route handlers ──

    def _handle_index(self) -> None:
        index_path = _STATIC_DIR / "index.html"
        html = index_path.read_text("utf-8")
        html = html.replace("{{ data|safe }}", json.dumps(self.backend.get_info()))
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        self.wfile.write(content)

    def _handle_palettes_js(self) -> None:
        js = _generate_palettes_js()
        data = js.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_pointcloud(self) -> None:
        with self.backend.get_pointcloud() as buf:
            data = buf.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _handle_geometry_file(self, name: str) -> None:
        data, ct = self.backend.get_geometry_file(name)
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_websocket(self) -> None:
        """WebSocket /render-websocket handler using nerfbaselines protocol."""
        from tribench.core._websocket import httpserver_websocket_handler, ConnectionClosed

        @httpserver_websocket_handler
        def _ws_handler(self_inner, ws):
            try:
                while True:
                    reqdata = ws.receive()
                    if reqdata is None:
                        break
                    out = self.backend.handle_websocket_message(reqdata)
                    ws.send(out)
            except ConnectionClosed:
                pass

        _ws_handler(self)

    # ── Helpers ──

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send(data, "application/json")

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, exc: Exception) -> None:
        status = 500
        if isinstance(exc, ViewerBadRequest):
            status = 400
        elif isinstance(exc, ViewerNotFound):
            status = 404
        payload = {"status": "error", "message": str(exc)}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ─── Palettes ─────────────────────────────────────────────────────────────


def _generate_palettes_js() -> str:
    """Generate palettes.js module from palettes.json."""
    with open(_PALETTES_JSON) as f:
        palettes = json.load(f)
    out = "const palettes = {\n"
    for name, colors in palettes.items():
        arr = np.array(colors, dtype=np.uint8).reshape(-1).tolist()
        out += f"{json.dumps(name)}:[{','.join(map(str, arr))}],\n"
    out += "};\nexport default palettes;\n"
    return out


# ─── Render helpers ───────────────────────────────────────────────────────


def _render_viewer_frame(
    adapter: RendererAdapter,
    request: dict[str, Any],
    *,
    datasets: dict[str, DatasetBase | None],
    device: str | torch.device | None,
    max_render_size: int,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    """Render one viewer request and return encoded image bytes."""

    output_type = _normalize_output_type(request.get("output_type", "color"))
    ref_dataset = datasets.get("test") or datasets.get("train")
    camera = _camera_from_request(request, dataset=ref_dataset, max_render_size=max_render_size)
    render_device = torch.device(device) if device is not None else adapter.device
    camera = camera.to(render_device)
    render_mode = _render_mode(adapter, output_type)

    with torch.no_grad():
        output = adapter.render(camera, mode=render_mode)

    # Format primary output
    arr = _format_output(output, output_type, output_range=request.get("output_range"))

    # Split compare
    split_type = request.get("split_output_type")
    if split_type:
        split_type = _normalize_output_type(split_type)
        split_arr = _format_output(output, split_type, output_range=request.get("split_range"))
        arr = _combine_outputs(arr, split_arr,
                               split_percentage=float(request.get("split_percentage", 0.5)),
                               split_tilt=float(request.get("split_tilt", 0)))

    lossless = bool(request.get("lossless", output_type != "color"))
    image_format = "PNG" if lossless else "JPEG"
    mimetype = "image/png" if image_format == "PNG" else "image/jpeg"
    with io.BytesIO() as buf:
        if image_format == "JPEG":
            Image.fromarray(arr).save(buf, format=image_format, quality=int(jpeg_quality))
        else:
            Image.fromarray(arr).save(buf, format=image_format)
        return buf.getvalue(), mimetype


def _combine_outputs(
    o1: np.ndarray, o2: np.ndarray, *, split_percentage: float = 0.5, split_tilt: float = 0
) -> np.ndarray:
    """Combine two outputs with a tilted splitting line (nerfbaselines-style)."""
    assert o1.shape == o2.shape, f"Shape mismatch: {o1.shape} vs {o2.shape}"
    h, w = o1.shape[:2]
    tilt_rad = split_tilt * math.pi / 180
    split_dir = [math.cos(tilt_rad), math.sin(tilt_rad)]
    y, x = np.indices((h, w))
    proj = (x - w / 2) * split_dir[0] + (y - h / 2) * split_dir[1]
    dir_len = w / 2 * abs(split_dir[0]) + h / 2 * abs(split_dir[1])
    mask = proj >= (split_percentage * 2 - 1) * dir_len
    result = o1.copy()
    result[mask] = o2[mask]
    return result


def _camera_from_request(
    request: dict[str, Any],
    *,
    dataset: DatasetBase | None,
    max_render_size: int,
) -> CameraBatch:
    if "pose" not in request:
        if dataset is None:
            raise ViewerBadRequest("Render request missing pose")
        idx = _clamp_index(int(request.get("idx", 0)), len(dataset))
        camera = dataset.sample(idx).camera
        image_size = request.get("image_size")
        if image_size is None:
            return camera
        width, height = _image_size_from_request(image_size, max_render_size)
        return _resize_camera(camera, width, height)

    pose = _pose_matrix(request["pose"])
    width, height = _image_size_from_request(request.get("image_size"), max_render_size)
    K = _intrinsics_matrix(request.get("intrinsics"), width=width, height=height, dataset=dataset)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = pose
    try:
        w2c = np.linalg.inv(c2w).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ViewerBadRequest("Camera pose is not invertible") from exc

    metadata = {"split": dataset.split} if dataset is not None else {}
    return CameraBatch(
        viewmats=torch.from_numpy(w2c).unsqueeze(0),
        camtoworlds=torch.from_numpy(c2w).unsqueeze(0),
        Ks=torch.from_numpy(K).unsqueeze(0),
        width=width,
        height=height,
        near=float(request.get("near", _dataset_near(dataset))),
        far=float(request.get("far", _dataset_far(dataset))),
        metadata=metadata,
    )


def _image_size_from_request(value: Any, max_render_size: int) -> tuple[int, int]:
    if value is None:
        width, height = 768, 512
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ViewerBadRequest("image_size must be [width, height]")
        width, height = int(round(float(arr[0]))), int(round(float(arr[1])))
    width = max(1, width)
    height = max(1, height)
    if max_render_size > 0 and max(width, height) > max_render_size:
        scale = float(max_render_size) / float(max(width, height))
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return width, height


def _pose_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 4):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 12:
        return flat.reshape(3, 4)
    if flat.size == 16:
        return flat.reshape(4, 4)[:3, :4]
    raise ViewerBadRequest("pose must contain a 3x4 or 4x4 camera-to-world matrix")


def _intrinsics_matrix(
    value: Any,
    *,
    width: int,
    height: int,
    dataset: DatasetBase | None,
) -> np.ndarray:
    if value is None:
        if dataset is not None and len(dataset) > 0:
            camera = dataset.sample(0).camera
            scale_x = width / float(camera.width)
            scale_y = height / float(camera.height)
            K = camera.Ks[0].detach().cpu().numpy().astype(np.float32).copy()
            K[0, :] *= scale_x
            K[1, :] *= scale_y
            return K
        focal = 0.5 * width / math.tan(math.radians(30.0))
        return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size == 4:
        fx, fy, cx, cy = [float(x) for x in flat]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    raise ViewerBadRequest("intrinsics must be a 3x3 matrix or [fx, fy, cx, cy]")


def _resize_camera(camera: CameraBatch, width: int, height: int) -> CameraBatch:
    scale_x = width / float(camera.width)
    scale_y = height / float(camera.height)
    Ks = camera.Ks.clone()
    Ks[:, 0, :] *= scale_x
    Ks[:, 1, :] *= scale_y
    return CameraBatch(
        viewmats=camera.viewmats.clone(),
        camtoworlds=camera.camtoworlds.clone(),
        Ks=Ks,
        width=width,
        height=height,
        near=camera.near,
        far=camera.far,
        metadata=camera.metadata,
    )


def _render_mode(adapter: RendererAdapter, output_type: str) -> str:
    method = str(getattr(adapter, "name", "")).replace("_", "-").lower()
    if method == "diffsoup" and output_type in {"depth", "alpha", "accumulation"}:
        return "eval_aux"
    return "eval"


def _to_hwc(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to [H, W, C] format, handling NCHW, NHW, and HWC inputs."""
    t = tensor.detach().float().cpu()
    if t.dim() == 2:
        return t.unsqueeze(-1)  # [H, W] → [H, W, 1]
    if t.dim() == 3:
        if t.shape[0] <= 4 and t.shape[-1] > 4:
            # NCHW with N=1 → [C, H, W] → [H, W, C]
            return t.squeeze(0).permute(1, 2, 0)
        return t  # Already [H, W, C]
    if t.dim() == 4:
        # [B, C, H, W] → [H, W, C]
        return t.squeeze(0).permute(1, 2, 0)
    return t


def _format_output(
    output: RenderOutput,
    output_type: str,
    *,
    output_range: Any = None,
) -> np.ndarray:
    tensor = _select_output(output, output_type)
    if output_type == "normal":
        arr = _to_hwc(tensor)
        if arr.shape[-1] == 3:
            arr = arr * 0.5 + 0.5
        return tensor_to_uint8(arr)
    if output_type in {"depth", "alpha", "accumulation"}:
        return _colorize_scalar(tensor, output_range=output_range)
    return tensor_to_uint8(_to_hwc(tensor))


def _select_output(output: RenderOutput, output_type: str) -> torch.Tensor:
    if output_type == "color":
        return output.rgb
    if output_type in {"alpha", "accumulation"}:
        if output.alpha is None:
            raise ViewerBadRequest("Renderer did not return alpha output")
        return output.alpha
    if output_type == "depth":
        if output.depth is None:
            raise ViewerBadRequest("Renderer did not return depth output")
        return output.depth
    if output_type == "normal":
        if output.normal is None:
            raise ViewerBadRequest("Renderer did not return normal output")
        return output.normal
    raise ViewerBadRequest(f"Unknown output type: {output_type}")


def _normalize_output_type(value: Any) -> str:
    output_type = str(value or "color").replace("rgb", "color").lower()
    if output_type == "opacity":
        return "alpha"
    return output_type


def _colorize_scalar(tensor: torch.Tensor, *, output_range: Any = None) -> np.ndarray:
    arr = _to_hwc(tensor)
    if arr.dim() == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    values = arr.numpy()
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values, dtype=np.float32)
        finite = np.ones_like(values, dtype=bool)

    lo: float
    hi: float
    if output_range is not None:
        rng = np.asarray(output_range, dtype=np.float32).reshape(-1)
        if rng.size == 2 and np.isfinite(rng).all() and rng[0] != rng[1]:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            lo, hi = _robust_range(values, finite)
    else:
        lo, hi = _robust_range(values, finite)
    scaled = np.zeros_like(values, dtype=np.float32)
    scaled[finite] = np.clip((values[finite] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    scaled[~finite] = 0.0
    red = np.clip(1.7 * scaled - 0.15, 0.0, 1.0)
    green = np.clip(1.7 * (1.0 - np.abs(scaled - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.25 * (1.0 - scaled), 0.0, 1.0)
    return ((np.stack([red, green, blue], axis=-1) * 255.0) + 0.5).astype(np.uint8)


def _robust_range(values: np.ndarray, finite: np.ndarray) -> tuple[float, float]:
    valid = values[finite]
    lo = float(np.percentile(valid, 2.0))
    hi = float(np.percentile(valid, 98.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


# ─── Utility helpers ──────────────────────────────────────────────────────


_RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)


def _parse_image_path(path: str) -> tuple[str, int]:
    """Parse /dataset/images/{split}/{idx}.jpg → (split, idx)."""
    prefix = "/dataset/images/"
    suffix = ".jpg"
    rest = path[len(prefix):-len(suffix)]
    if "/" not in rest:
        raise ViewerBadRequest("Invalid image path")
    split, idx_str = rest.rsplit("/", 1)
    try:
        return split, int(idx_str)
    except ValueError as exc:
        raise ViewerBadRequest("Invalid dataset image index") from exc


def _sample_pil_image(sample: DatasetSample) -> Image.Image:
    if sample.image is not None:
        return Image.fromarray(tensor_to_uint8(sample.image)).convert("RGB")
    if sample.image_path is not None and Path(sample.image_path).exists():
        return Image.open(sample.image_path).convert("RGB")
    camera = sample.camera
    return Image.new("RGB", (int(camera.width), int(camera.height)), (20, 22, 24))


def _scene_from_dataset(dataset: DatasetBase) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"center": [0.0, 0.0, 0.0], "radius": 1.0, "up": [0.0, -1.0, 0.0]}

    centers = []
    ups = []
    for idx in range(len(dataset)):
        c2w = dataset.sample(idx).camera.camtoworlds[0].detach().cpu().numpy()
        centers.append(c2w[:3, 3])
        ups.append(-c2w[:3, 1])
    centers_np = np.stack(centers, axis=0).astype(np.float32)
    center = np.mean(centers_np, axis=0)
    distances = np.linalg.norm(centers_np - center[None, :], axis=1)
    radius = float(np.percentile(distances, 90.0)) if len(distances) > 1 else 1.0
    if not math.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    up = np.mean(np.stack(ups, axis=0).astype(np.float32), axis=0)
    up_norm = float(np.linalg.norm(up))
    if not math.isfinite(up_norm) or up_norm <= 1e-6:
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        up = up / up_norm
    return {
        "center": [float(x) for x in center],
        "radius": radius,
        "up": [float(x) for x in up],
    }


def _safe_model_stats(adapter: RendererAdapter) -> dict[str, Any]:
    try:
        stats = adapter.model_stats()
    except Exception:
        return {"method": getattr(adapter, "name", adapter.__class__.__name__)}
    return _jsonable(stats)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matrix_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _clamp_index(idx: int, length: int) -> int:
    if length <= 0:
        raise ViewerNotFound("Dataset has no cameras")
    return max(0, min(length - 1, int(idx)))


def _dataset_near(dataset: DatasetBase | None) -> float:
    if dataset is not None and len(dataset) > 0:
        return float(dataset.sample(0).camera.near)
    return 0.01


def _dataset_far(dataset: DatasetBase | None) -> float:
    if dataset is not None and len(dataset) > 0:
        return float(dataset.sample(0).camera.far)
    return 100.0


def _bind_server(host: str, port: int, handler: Any) -> ThreadingHTTPServer:
    bind_host = "" if host in {"", "localhost"} else host
    address_family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET

    class ViewerHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    ViewerHTTPServer.address_family = address_family

    for _ in range(100):
        try:
            return ViewerHTTPServer((bind_host, port), handler)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE and port > 0:
                port += 1
                continue
            raise
    raise RuntimeError("Could not find a free port")


def _display_host(host: str, port: int) -> str:
    if ":" in host:
        return f"[{host}]:{port}"
    if host in {"", "0.0.0.0"}:
        return f"localhost:{port}"
    return f"{host}:{port}"


# ─── Public API ───────────────────────────────────────────────────────────


def get_viewer_info(
    adapter: RendererAdapter,
    dataset: DatasetBase | None,
    *,
    train_dataset: DatasetBase | None = None,
    max_render_size: int = 1280,
) -> dict[str, Any]:
    """Build the viewer params JSON for embedding in index.html.

    This is the public API wrapper around ``_build_viewer_params``.
    Returns a dict with ``renderer``, ``dataset``, ``state`` keys.
    """
    test_ds = dataset
    train_ds = train_dataset
    has_pc = False  # point cloud path not known at this level
    return _build_viewer_params(
        adapter,
        train_dataset=train_ds,
        test_dataset=test_ds,
        primary_split="test" if test_ds is not None else "train",
        max_render_size=max_render_size,
        has_pointcloud=has_pc,
    )


def get_viewer_dataset(dataset: DatasetBase) -> dict[str, Any]:
    """Build the nerfbaselines-compatible dataset.json.

    This is the public API wrapper around ``_build_dataset_json``.
    Returns a dict with ``test`` (or ``train``) key containing ``cameras`` list.
    """
    return _build_dataset_json({"test": dataset})


def render_viewer_frame(
    adapter: RendererAdapter,
    request: dict[str, Any],
    *,
    dataset: DatasetBase | None = None,
    device: str | torch.device | None = None,
    max_render_size: int = 1280,
    jpeg_quality: int = 90,
) -> tuple[bytes, str]:
    """Render one viewer frame request and return (payload_bytes, mimetype).

    This is the public API wrapper around ``_render_viewer_frame``.
    """
    datasets = {"test": dataset} if dataset is not None else {}
    return _render_viewer_frame(
        adapter,
        request,
        datasets=datasets,
        device=device,
        max_render_size=max_render_size,
        jpeg_quality=jpeg_quality,
    )


def get_viewer_html(
    adapter: RendererAdapter,
    dataset: DatasetBase | None = None,
    *,
    train_dataset: DatasetBase | None = None,
    geometry_path: str | Path | None = None,
    mesh_geometry_path: str | Path | None = None,
    mesh_preview: bool = False,
    max_render_size: int = 1280,
) -> str:
    """Build the complete HTML page for the viewer.

    Returns the index.html with ``{{ data|safe }}`` replaced by the viewer
    params JSON. This is useful for testing that the HTML is correctly
    generated without starting a server.
    """
    info = _build_viewer_params(
        adapter,
        train_dataset=train_dataset,
        test_dataset=dataset,
        primary_split="test" if dataset is not None else "train",
        max_render_size=max_render_size,
        has_pointcloud=geometry_path is not None,
        mesh_preview=mesh_preview and mesh_geometry_path is not None,
        mesh_url="./dataset/geometry/mesh.ply" if mesh_geometry_path else None,
    )
    index_path = _STATIC_DIR / "index.html"
    html = index_path.read_text("utf-8")
    return html.replace("{{ data|safe }}", json.dumps(info))
