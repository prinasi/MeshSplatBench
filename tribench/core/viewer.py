"""Minimal online render viewer."""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

from tribench.core.datasets import DatasetBase
from tribench.core.rendering import render_sample, tensor_to_uint8
from tribench.renderers.base import RendererAdapter


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TriBench Viewer</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    header { height: 48px; display: flex; align-items: center; gap: 12px; padding: 0 16px; background: #1c1c1c; }
    button, input { height: 30px; }
    main { height: calc(100vh - 48px); display: grid; place-items: center; }
    img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .spacer { flex: 1; }
  </style>
</head>
<body>
  <header>
    <strong>TriBench Viewer</strong>
    <button onclick="step(-1)">Prev</button>
    <input id="idx" type="number" min="0" value="0" onchange="loadFrame()">
    <button onclick="step(1)">Next</button>
    <span id="status"></span>
    <span class="spacer"></span>
    <button onclick="loadFrame()">Refresh</button>
  </header>
  <main><img id="render" alt="render"></main>
  <script>
    let count = 1;
    async function loadMeta() {
      const meta = await fetch('/meta').then(r => r.json());
      count = meta.num_views;
      document.getElementById('status').textContent = `${meta.split}: ${count} views`;
    }
    function step(delta) {
      const el = document.getElementById('idx');
      let value = Number(el.value || 0) + delta;
      value = Math.max(0, Math.min(count - 1, value));
      el.value = value;
      loadFrame();
    }
    function loadFrame() {
      const idx = document.getElementById('idx').value || 0;
      document.getElementById('render').src = `/render?idx=${idx}&t=${Date.now()}`;
    }
    loadMeta().then(loadFrame);
  </script>
</body>
</html>
"""


def serve_viewer(
    adapter: RendererAdapter,
    dataset: DatasetBase,
    *,
    host: str = "127.0.0.1",
    port: int = 7007,
    device: str | torch.device = "cuda",
) -> str:
    """Serve a tiny browser viewer that renders dataset cameras on demand."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/meta":
                payload = {
                    "num_views": len(dataset),
                    "split": dataset.split,
                }
                self._send(json.dumps(payload).encode("utf-8"), "application/json")
                return
            if parsed.path == "/render":
                idx = int(parse_qs(parsed.query).get("idx", ["0"])[0])
                idx = max(0, min(len(dataset) - 1, idx))
                sample = dataset.sample(idx)
                output = render_sample(adapter, sample, device=device)
                arr = tensor_to_uint8(output.rgb)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                self._send(buf.getvalue(), "image/png")
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002
            return

        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"TriBench viewer running at {url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return url

