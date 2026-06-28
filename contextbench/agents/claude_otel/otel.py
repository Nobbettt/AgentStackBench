# SPDX-License-Identifier: Apache-2.0

"""Parsing utilities for captured Claude Code OpenTelemetry payloads."""

from __future__ import annotations

import json
from pathlib import Path

from ...coding_agents.files import read_jsonl_values

_MINIMAL_STRUCTURED_OUTPUT_KEYS = ("status", "final_answer", "notes")


def _minimal_structured_output(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if not all(key in value for key in _MINIMAL_STRUCTURED_OUTPUT_KEYS):
        return None
    return {key: value.get(key) for key in _MINIMAL_STRUCTURED_OUTPUT_KEYS}


def extract_claude_otel_structured_output_from_value(value: object) -> dict[str, object] | None:
    return _minimal_structured_output(value)


def any_value_to_python(value: object) -> object:
    """Convert OTLP JSON AnyValue shapes to plain Python values."""

    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value.get("stringValue")
    if "boolValue" in value:
        return bool(value.get("boolValue"))
    if "intValue" in value:
        raw = value.get("intValue")
        try:
            return int(raw)  # OTLP JSON often encodes int64 as strings.
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value.get("doubleValue")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "bytesValue" in value:
        return value.get("bytesValue")
    if "arrayValue" in value:
        array_value = value.get("arrayValue")
        values = array_value.get("values") if isinstance(array_value, dict) else []
        return [any_value_to_python(item) for item in values or []]
    if "kvlistValue" in value:
        kvlist = value.get("kvlistValue")
        values = kvlist.get("values") if isinstance(kvlist, dict) else []
        return attributes_to_dict(values)
    return {str(key): any_value_to_python(item) for key, item in value.items()}


def attributes_to_dict(attributes: object) -> dict[str, object]:
    if isinstance(attributes, dict):
        return {str(key): any_value_to_python(value) for key, value in attributes.items()}
    result: dict[str, object] = {}
    if not isinstance(attributes, list):
        return result
    for item in attributes:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        result[key] = any_value_to_python(item.get("value"))
    return result


def _body_value(body: object) -> object:
    return any_value_to_python(body)


def _event_name(attrs: dict[str, object], body: object) -> str:
    raw = attrs.get("event.name") or attrs.get("event_name") or attrs.get("name")
    if raw:
        text = str(raw).strip()
        return text if text.startswith("claude_code.") else f"claude_code.{text}"
    if isinstance(body, dict):
        raw = body.get("event.name") or body.get("event_name") or body.get("name")
        if raw:
            text = str(raw).strip()
            return text if text.startswith("claude_code.") else f"claude_code.{text}"
    if isinstance(body, str) and body.startswith("claude_code."):
        return body
    return ""


def _requests_from_raw_response(raw_response: dict[str, object]) -> list[dict[str, object]]:
    otel = raw_response.get("otel")
    if not isinstance(otel, dict):
        return []
    requests = otel.get("requests")
    return [item for item in requests if isinstance(item, dict)] if isinstance(requests, list) else []


def _stored_otel_dicts(raw_response: dict[str, object], key: str) -> list[dict[str, object]]:
    otel = raw_response.get("otel")
    if not isinstance(otel, dict):
        return []
    values = otel.get(key)
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _otel_key_present(raw_response: dict[str, object], key: str) -> bool:
    otel = raw_response.get("otel")
    return isinstance(otel, dict) and key in otel


def read_capture_requests(path: Path) -> list[dict[str, object]]:
    return [item for item in read_jsonl_values(path) if isinstance(item, dict)] if path.exists() else []


def extract_log_records_from_requests(raw_response: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for request in _requests_from_raw_response(raw_response):
        body = request.get("body")
        if not isinstance(body, dict):
            continue
        for resource_log in body.get("resourceLogs") or []:
            if not isinstance(resource_log, dict):
                continue
            resource_attrs = attributes_to_dict((resource_log.get("resource") or {}).get("attributes"))
            for scope_log in resource_log.get("scopeLogs") or []:
                if not isinstance(scope_log, dict):
                    continue
                scope = scope_log.get("scope") if isinstance(scope_log.get("scope"), dict) else {}
                scope_attrs = attributes_to_dict(scope.get("attributes")) if isinstance(scope, dict) else {}
                for record in scope_log.get("logRecords") or []:
                    if not isinstance(record, dict):
                        continue
                    attrs = {
                        **resource_attrs,
                        **scope_attrs,
                        **attributes_to_dict(record.get("attributes")),
                    }
                    body_value = _body_value(record.get("body"))
                    records.append(
                        {
                            "name": _event_name(attrs, body_value),
                            "attributes": attrs,
                            "body": body_value,
                            "time_unix_nano": record.get("timeUnixNano"),
                            "severity_text": record.get("severityText"),
                        }
                    )
    return records


def extract_log_records(raw_response: dict[str, object]) -> list[dict[str, object]]:
    if _otel_key_present(raw_response, "logs"):
        return _stored_otel_dicts(raw_response, "logs")
    return []


def extract_spans_from_requests(raw_response: dict[str, object]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for request in _requests_from_raw_response(raw_response):
        body = request.get("body")
        if not isinstance(body, dict):
            continue
        for resource_span in body.get("resourceSpans") or []:
            if not isinstance(resource_span, dict):
                continue
            resource_attrs = attributes_to_dict((resource_span.get("resource") or {}).get("attributes"))
            for scope_span in resource_span.get("scopeSpans") or []:
                if not isinstance(scope_span, dict):
                    continue
                for span in scope_span.get("spans") or []:
                    if not isinstance(span, dict):
                        continue
                    events = []
                    for event in span.get("events") or []:
                        if not isinstance(event, dict):
                            continue
                        events.append(
                            {
                                "name": str(event.get("name") or ""),
                                "attributes": attributes_to_dict(event.get("attributes")),
                                "time_unix_nano": event.get("timeUnixNano"),
                            }
                        )
                    spans.append(
                        {
                            "name": str(span.get("name") or ""),
                            "span_id": span.get("spanId"),
                            "parent_span_id": span.get("parentSpanId"),
                            "trace_id": span.get("traceId"),
                            "attributes": {
                                **resource_attrs,
                                **attributes_to_dict(span.get("attributes")),
                            },
                            "events": events,
                            "status": span.get("status") if isinstance(span.get("status"), dict) else {},
                            "start_time_unix_nano": span.get("startTimeUnixNano"),
                            "end_time_unix_nano": span.get("endTimeUnixNano"),
                        }
                    )
    return spans


def extract_spans(raw_response: dict[str, object]) -> list[dict[str, object]]:
    stored = _stored_otel_dicts(raw_response, "spans")
    if _otel_key_present(raw_response, "spans"):
        return stored
    return []


def extract_metrics_from_requests(raw_response: dict[str, object]) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for request in _requests_from_raw_response(raw_response):
        body = request.get("body")
        if not isinstance(body, dict):
            continue
        for resource_metric in body.get("resourceMetrics") or []:
            if not isinstance(resource_metric, dict):
                continue
            resource_attrs = attributes_to_dict((resource_metric.get("resource") or {}).get("attributes"))
            for scope_metric in resource_metric.get("scopeMetrics") or []:
                if not isinstance(scope_metric, dict):
                    continue
                for metric in scope_metric.get("metrics") or []:
                    if not isinstance(metric, dict):
                        continue
                    points = []
                    data_kind = ""
                    for candidate in ("sum", "gauge", "histogram"):
                        data = metric.get(candidate)
                        if isinstance(data, dict):
                            data_kind = candidate
                            for point in data.get("dataPoints") or []:
                                if not isinstance(point, dict):
                                    continue
                                points.append(
                                    {
                                        "attributes": {
                                            **resource_attrs,
                                            **attributes_to_dict(point.get("attributes")),
                                        },
                                        "as_int": point.get("asInt"),
                                        "as_double": point.get("asDouble"),
                                        "count": point.get("count"),
                                        "sum": point.get("sum"),
                                    }
                                )
                            break
                    metrics.append(
                        {
                            "name": str(metric.get("name") or ""),
                            "description": metric.get("description"),
                            "unit": metric.get("unit"),
                            "data_kind": data_kind,
                            "points": points,
                        }
                    )
    return metrics


def extract_metrics(raw_response: dict[str, object]) -> list[dict[str, object]]:
    stored = _stored_otel_dicts(raw_response, "metrics")
    if _otel_key_present(raw_response, "metrics"):
        return stored
    return []


def _api_body_logs_from_records(
    records: list[dict[str, object]],
    event_name: str,
    *,
    body_ref_root: Path,
) -> list[tuple[dict[str, object], object | None]]:
    items: list[tuple[dict[str, object], object | None]] = []
    for log in records:
        if log.get("name") != event_name:
            continue
        attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
        body = None
        if attrs.get("body_ref"):
            body = _read_body_ref(attrs.get("body_ref"), body_ref_root=body_ref_root)
        items.append(({"attributes": attrs, "time_unix_nano": log.get("time_unix_nano")}, body))
    return items


def _body_ref_summary(event_name: str, attrs: dict[str, object], time_unix_nano: object) -> dict[str, object]:
    return {
        "name": event_name,
        "body_ref": attrs.get("body_ref"),
        "body_length": attrs.get("body_length"),
        "body_truncated": attrs.get("body_truncated"),
        "model": attrs.get("model"),
        "request_id": attrs.get("request_id"),
        "time_unix_nano": time_unix_nano,
    }


def extract_api_body_refs_from_requests(raw_response: dict[str, object]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for log in extract_log_records_from_requests(raw_response):
        name = str(log.get("name") or "")
        if name not in {"claude_code.api_request_body", "claude_code.api_response_body"}:
            continue
        attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
        refs.append(_body_ref_summary(name, attrs, log.get("time_unix_nano")))
    return refs


def extract_api_body_refs(raw_response: dict[str, object]) -> list[dict[str, object]]:
    if _otel_key_present(raw_response, "api_body_refs"):
        return _stored_otel_dicts(raw_response, "api_body_refs")
    return []


def _parse_json_payload(value: object) -> object | None:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _resolve_body_ref_path(value: object, *, body_ref_root: Path) -> tuple[Path | None, str | None]:
    raw_path = str(value or "").strip()
    if not raw_path:
        return None, "missing_body_ref"
    try:
        root = body_ref_root.resolve(strict=True)
    except FileNotFoundError:
        return None, "body_ref_root_missing"
    except OSError as exc:
        return None, f"body_ref_root_unreadable: {exc}"
    try:
        path = Path(raw_path).resolve(strict=True)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"unreadable_path: {exc}"
    try:
        path.relative_to(root)
    except ValueError:
        return None, "outside_body_ref_root"
    if not path.is_file():
        return None, "not_file"
    return path, None


def body_ref_read_error(value: object, *, body_ref_root: Path) -> str | None:
    path, error = _resolve_body_ref_path(value, body_ref_root=body_ref_root)
    if error:
        return error
    assert path is not None
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid_json: {exc}"
    return None


def _read_body_ref(value: object, *, body_ref_root: Path) -> object | None:
    path, error = _resolve_body_ref_path(value, body_ref_root=body_ref_root)
    if error or path is None:
        return None
    try:
        return _parse_json_payload(path.read_text(encoding="utf-8"))
    except Exception:
        return None



def _tool_name_from_value(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("name") or value.get("tool_name") or "").strip()


def _tool_uses_from_response_body(body: object) -> list[dict[str, object]]:
    if not isinstance(body, dict):
        return []
    content = body.get("content")
    if not isinstance(content, list):
        return []
    tool_uses: list[dict[str, object]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        tool_name = _tool_name_from_value(item) or "unknown"
        tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
        tool_uses.append(
            {
                "id": str(item.get("id") or "").strip(),
                "name": tool_name,
                "input": dict(tool_input),
            }
        )
    return tool_uses


def _tool_results_from_request_body(body: object) -> list[dict[str, object]]:
    if not isinstance(body, dict):
        return []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            tool_use_id = str(item.get("tool_use_id") or "").strip()
            if not tool_use_id or tool_use_id in seen:
                continue
            seen.add(tool_use_id)
            results.append(
                {
                    "tool_use_id": tool_use_id,
                    "content": item.get("content"),
                    "is_error": item.get("is_error"),
                }
            )
    return results


def _api_request_body_summaries_from_records(
    records: list[dict[str, object]], *, body_ref_root: Path
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for log, body in _api_body_logs_from_records(
        records, "claude_code.api_request_body", body_ref_root=body_ref_root
    ):
        attrs = log["attributes"] if isinstance(log.get("attributes"), dict) else {}
        tools: list[str] = []
        if isinstance(body, dict):
            for item in body.get("tools") or []:
                name = _tool_name_from_value(item)
                if name:
                    tools.append(name)
        summaries.append(
            {
                "source": "claude.otel.api_request_body",
                "request_id": attrs.get("request_id"),
                "time_unix_nano": log.get("time_unix_nano"),
                "model": attrs.get("model") or (body.get("model") if isinstance(body, dict) else None),
                "tool_names": tools,
                "tool_count": len(tools),
                "tool_results": _tool_results_from_request_body(body),
            }
        )
    return summaries


def extract_api_request_body_summaries_from_requests(
    raw_response: dict[str, object], *, body_ref_root: Path
) -> list[dict[str, object]]:
    return _api_request_body_summaries_from_records(
        extract_log_records_from_requests(raw_response), body_ref_root=body_ref_root
    )


def extract_api_request_body_summaries(raw_response: dict[str, object]) -> list[dict[str, object]]:
    if _otel_key_present(raw_response, "api_request_bodies"):
        return _stored_otel_dicts(raw_response, "api_request_bodies")
    return []


def _api_response_body_summaries_from_records(
    records: list[dict[str, object]], *, body_ref_root: Path
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for log, body in _api_body_logs_from_records(
        records, "claude_code.api_response_body", body_ref_root=body_ref_root
    ):
        attrs = log["attributes"] if isinstance(log.get("attributes"), dict) else {}
        usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else None
        summaries.append(
            {
                "source": "claude.otel.api_response_body",
                "request_id": attrs.get("request_id"),
                "time_unix_nano": log.get("time_unix_nano"),
                "model": attrs.get("model") or (body.get("model") if isinstance(body, dict) else None),
                "message_id": body.get("id") if isinstance(body, dict) else None,
                "stop_reason": body.get("stop_reason") if isinstance(body, dict) else None,
                "stop_details": body.get("stop_details") if isinstance(body, dict) else None,
                "usage": dict(usage) if usage is not None else None,
                "tool_uses": _tool_uses_from_response_body(body),
            }
        )
    return summaries


def extract_api_response_body_summaries_from_requests(
    raw_response: dict[str, object], *, body_ref_root: Path
) -> list[dict[str, object]]:
    return _api_response_body_summaries_from_records(
        extract_log_records_from_requests(raw_response), body_ref_root=body_ref_root
    )


def extract_api_response_body_summaries(raw_response: dict[str, object]) -> list[dict[str, object]]:
    if _otel_key_present(raw_response, "api_response_bodies"):
        return _stored_otel_dicts(raw_response, "api_response_bodies")
    return []


def extract_available_tools_from_api_bodies(raw_response: dict[str, object]) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for summary in extract_api_request_body_summaries(raw_response):
        for item in summary.get("tool_names") or []:
            name = str(item or "").strip()
            if name and name not in seen:
                tools.append(name)
                seen.add(name)
    return tools


def extract_tool_uses_from_api_bodies(raw_response: dict[str, object]) -> list[dict[str, object]]:
    tool_uses: list[dict[str, object]] = []
    for response in extract_api_response_body_summaries(raw_response):
        for item in response.get("tool_uses") or []:
            if not isinstance(item, dict):
                continue
            tool_uses.append(
                {
                    **item,
                    "request_id": response.get("request_id"),
                    "message_id": response.get("message_id"),
                    "time_unix_nano": response.get("time_unix_nano"),
                }
            )
    return tool_uses


def extract_tool_results_from_api_bodies(raw_response: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for request in extract_api_request_body_summaries(raw_response):
        for item in request.get("tool_results") or []:
            if not isinstance(item, dict):
                continue
            tool_use_id = str(item.get("tool_use_id") or "").strip()
            if not tool_use_id or tool_use_id in seen:
                continue
            seen.add(tool_use_id)
            results.append(
                {
                    **item,
                    "request_id": request.get("request_id"),
                    "time_unix_nano": request.get("time_unix_nano"),
                }
            )
    return results
