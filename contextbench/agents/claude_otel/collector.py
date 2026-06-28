# SPDX-License-Identifier: Apache-2.0

"""Local OTLP/HTTP capture helpers for Claude Code telemetry."""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class OtelHttpCapture:
    """Capture OTLP/HTTP JSON posts from one Claude invocation."""

    def __init__(
        self,
        output_path: Path,
        *,
        bind_host: str = "127.0.0.1",
        endpoint_host: str | None = None,
    ) -> None:
        self.output_path = output_path
        self.bind_host = bind_host
        self.endpoint_host = endpoint_host or bind_host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("OTEL capture server is not started")
        return int(self._server.server_address[1])

    @property
    def endpoint(self) -> str:
        return f"http://{self.endpoint_host}:{self.port}"

    @property
    def logs_endpoint(self) -> str:
        return f"{self.endpoint}/v1/logs"

    @property
    def metrics_endpoint(self) -> str:
        return f"{self.endpoint}/v1/metrics"

    @property
    def traces_endpoint(self) -> str:
        return f"{self.endpoint}/v1/traces"

    def __enter__(self) -> "OtelHttpCapture":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

        capture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
                capture._handle_post(self)

            def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer((self.bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # Give async OTEL exporters a short window to finish any in-flight POST.
        time.sleep(0.2)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("content-length") or "0"))
        entry: dict[str, object] = {
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "path": handler.path,
            "content_type": handler.headers.get("content-type"),
        }
        try:
            entry["body"] = json.loads(body.decode("utf-8"))
            entry["body_encoding"] = "json"
        except Exception:
            entry["body"] = base64.b64encode(body).decode("ascii")
            entry["body_encoding"] = "base64"

        with self._lock:
            with open(self.output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")

        handler.send_response(200)
        handler.send_header("content-type", "application/json")
        handler.end_headers()
        handler.wfile.write(b"{}\n")
