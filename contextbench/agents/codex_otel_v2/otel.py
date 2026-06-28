# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry capture helpers for Codex OTEL v2 runs."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_EMAIL_RE = re.compile(r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "api-key",
    "password",
    "secret",
    "token",
    "user.account_id",
    "user.email",
}
_SENSITIVE_KEY_FRAGMENTS = ("authorization", "api_key", "api-key", "password", "secret")


def _is_sensitive_key(key: str) -> bool:
    if key in _SENSITIVE_KEYS or any(fragment in key for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return True
    if "token_count" in key:
        return False
    return key == "token" or key.endswith(".token") or key.endswith("_token") or "auth.token" in key


def sanitize_otel_value(value: object, *, parent_key: str = "") -> object:
    """Redact stable user identifiers and secrets before persisting OTEL payloads."""

    key = parent_key.lower()
    if _is_sensitive_key(key):
        return "[redacted]"
    if isinstance(value, str):
        return _EMAIL_RE.sub("[redacted-email]", value)
    if isinstance(value, list):
        return [sanitize_otel_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        otlp_key = value.get("key")
        if isinstance(otlp_key, str) and "value" in value:
            return {
                str(raw_key): (
                    sanitize_otel_value(item, parent_key=otlp_key)
                    if str(raw_key) == "value"
                    else sanitize_otel_value(item, parent_key=str(raw_key))
                )
                for raw_key, item in value.items()
            }
        sanitized: dict[str, object] = {}
        for raw_key, item in value.items():
            child_key = str(raw_key)
            sanitized[child_key] = sanitize_otel_value(item, parent_key=child_key)
        return sanitized
    return value


def otlp_value_to_python(value: object) -> object:
    """Convert OTLP JSON value envelopes into ordinary Python values."""

    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)  # OTLP JSON often encodes integers as strings.
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value["doubleValue"]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        values = value.get("arrayValue")
        if isinstance(values, dict):
            return [otlp_value_to_python(item) for item in values.get("values", [])]
        return []
    if "kvlistValue" in value:
        values = value.get("kvlistValue")
        if isinstance(values, dict):
            return attributes_to_dict(values.get("values"))
        return {}
    return value


def attributes_to_dict(attributes: object) -> dict[str, object]:
    result: dict[str, object] = {}
    if not isinstance(attributes, list):
        return result
    for item in attributes:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        result[key] = sanitize_otel_value(otlp_value_to_python(item.get("value")), parent_key=key)
    return result


def _body_to_python(value: object) -> object:
    if isinstance(value, dict):
        return otlp_value_to_python(value)
    return value


def flatten_otlp_logs(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    events: list[dict[str, object]] = []
    for resource_log in payload.get("resourceLogs") or []:
        if not isinstance(resource_log, dict):
            continue
        resource = resource_log.get("resource")
        resource_attrs = attributes_to_dict(resource.get("attributes") if isinstance(resource, dict) else None)
        for scope_log in resource_log.get("scopeLogs") or []:
            if not isinstance(scope_log, dict):
                continue
            scope = scope_log.get("scope")
            scope_attrs = attributes_to_dict(scope.get("attributes") if isinstance(scope, dict) else None)
            for record in scope_log.get("logRecords") or []:
                if not isinstance(record, dict):
                    continue
                attrs = attributes_to_dict(record.get("attributes"))
                body = _body_to_python(record.get("body"))
                event_name = _event_name_from(attrs=attrs, body=body, fallback=record.get("name"))
                events.append(
                    {
                        "signal": "log",
                        "name": event_name,
                        "body": body,
                        "attributes": attrs,
                        "resource": resource_attrs,
                        "scope": scope_attrs,
                        "time_unix_nano": record.get("timeUnixNano"),
                        "observed_time_unix_nano": record.get("observedTimeUnixNano"),
                        "severity_text": record.get("severityText"),
                    }
                )
    return events


def flatten_otlp_traces(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    spans: list[dict[str, object]] = []
    for resource_span in payload.get("resourceSpans") or []:
        if not isinstance(resource_span, dict):
            continue
        resource = resource_span.get("resource")
        resource_attrs = attributes_to_dict(resource.get("attributes") if isinstance(resource, dict) else None)
        for scope_span in resource_span.get("scopeSpans") or []:
            if not isinstance(scope_span, dict):
                continue
            scope = scope_span.get("scope")
            scope_attrs = attributes_to_dict(scope.get("attributes") if isinstance(scope, dict) else None)
            for span in scope_span.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                attrs = attributes_to_dict(span.get("attributes"))
                spans.append(
                    {
                        "signal": "span",
                        "name": str(span.get("name") or "").strip(),
                        "attributes": attrs,
                        "resource": resource_attrs,
                        "scope": scope_attrs,
                        "trace_id": span.get("traceId"),
                        "span_id": span.get("spanId"),
                        "parent_span_id": span.get("parentSpanId"),
                        "start_time_unix_nano": span.get("startTimeUnixNano"),
                        "end_time_unix_nano": span.get("endTimeUnixNano"),
                        "status": span.get("status"),
                    }
                )
    return spans


def flatten_otlp_metrics(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    metrics: list[dict[str, object]] = []
    for resource_metric in payload.get("resourceMetrics") or []:
        if not isinstance(resource_metric, dict):
            continue
        resource = resource_metric.get("resource")
        resource_attrs = attributes_to_dict(resource.get("attributes") if isinstance(resource, dict) else None)
        for scope_metric in resource_metric.get("scopeMetrics") or []:
            if not isinstance(scope_metric, dict):
                continue
            scope = scope_metric.get("scope")
            scope_attrs = attributes_to_dict(scope.get("attributes") if isinstance(scope, dict) else None)
            for metric in scope_metric.get("metrics") or []:
                if not isinstance(metric, dict):
                    continue
                metrics.append(
                    {
                        "signal": "metric",
                        "name": str(metric.get("name") or "").strip(),
                        "description": metric.get("description"),
                        "unit": metric.get("unit"),
                        "resource": resource_attrs,
                        "scope": scope_attrs,
                        "raw": metric,
                    }
                )
    return metrics


def _event_name_from(*, attrs: dict[str, object], body: object, fallback: object = None) -> str:
    for key in ("event.name", "event_name", "name", "otel.name"):
        value = str(attrs.get(key) or "").strip()
        if value:
            return value
    if isinstance(body, str) and body.strip().startswith("codex."):
        return body.strip()
    if isinstance(body, dict):
        for key in ("event.name", "event_name", "name"):
            value = str(body.get(key) or "").strip()
            if value:
                return value
    return str(fallback or "").strip()


class OtlpJsonCaptureServer:
    """Small OTLP/HTTP JSON collector used during one Codex invocation."""

    def __init__(self, *, bind_host: str = "127.0.0.1") -> None:
        self._requests: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((bind_host, 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, name="codex-otel-capture", daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def export(self, *, include_requests: bool = False) -> dict[str, object]:
        with self._lock:
            requests = [dict(item) for item in self._requests]
        logs: list[dict[str, object]] = []
        traces: list[dict[str, object]] = []
        metrics: list[dict[str, object]] = []
        for request in requests:
            payload = request.get("json")
            if request.get("path") == "/v1/traces":
                traces.extend(flatten_otlp_traces(payload))
            elif request.get("path") == "/v1/metrics":
                metrics.extend(flatten_otlp_metrics(payload))
            else:
                logs.extend(flatten_otlp_logs(payload))
                traces.extend(flatten_otlp_traces(payload))
                metrics.extend(flatten_otlp_metrics(payload))
        capture: dict[str, object] = {
            "request_count": len(requests),
            "logs": logs,
            "traces": traces,
            "metrics": metrics,
        }
        if include_requests:
            capture["requests"] = requests
        return capture

    def write_json(self, path: Path, *, include_requests: bool = False) -> None:
        path.write_text(
            json.dumps(self.export(include_requests=include_requests), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type") or ""
                decoded: object | None = None
                parse_error: str | None = None
                try:
                    decoded = sanitize_otel_value(json.loads(body.decode("utf-8")))
                except Exception as exc:
                    parse_error = str(exc)
                entry: dict[str, object] = {
                    "received_at": time.time(),
                    "path": self.path,
                    "content_type": content_type,
                    "size_bytes": len(body),
                    "json": decoded,
                }
                if parse_error:
                    entry["parse_error"] = parse_error
                    entry["body_base64"] = base64.b64encode(body).decode("ascii")
                with outer._lock:
                    outer._requests.append(entry)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        return Handler
