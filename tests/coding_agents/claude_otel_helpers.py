# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def otel_attr(key: str, value: object) -> dict[str, object]:
    if isinstance(value, bool):
        otel_value = {"boolValue": value}
    elif isinstance(value, int):
        otel_value = {"intValue": str(value)}
    elif isinstance(value, float):
        otel_value = {"doubleValue": value}
    else:
        otel_value = {"stringValue": str(value)}
    return {"key": key, "value": otel_value}


def logs_payload(*records: dict[str, object]) -> dict[str, object]:
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": [otel_attr("service.name", "contextbench-claude-otel")]},
                "scopeLogs": [{"logRecords": list(records)}],
            }
        ]
    }


def traces_payload(*spans: dict[str, object]) -> dict[str, object]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [otel_attr("service.name", "contextbench-claude-otel")]},
                "scopeSpans": [{"spans": list(spans)}],
            }
        ]
    }


def log_record(name: str, attrs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "timeUnixNano": "1",
        "body": {"stringValue": name},
        "attributes": [otel_attr("event.name", name.removeprefix("claude_code.")), *attrs],
    }


def span(name: str, attrs: list[dict[str, object]], events: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "traceId": "0" * 32,
        "spanId": "1" * 16,
        "name": name,
        "attributes": attrs,
        "events": events or [],
    }
