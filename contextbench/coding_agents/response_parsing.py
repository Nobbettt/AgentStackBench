
"""Response parsing helpers for coding-agent integrations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .constants import SEMANTIC_OUTPUT_KEYS
from .files import read_json, read_json_or_text, read_jsonl_values
from .types import ClaudeRawResponse, CodexRawResponse, StructuredOutput


def parse_json_from_text(text: object) -> StructuredOutput | dict[str, object] | None:
    value = str(text or "").strip()
    if not value:
        return None
    candidates = [value]
    if value.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", value, re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())
    first_brace = value.find("{")
    last_brace = value.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(value[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def collect_nested_values(value: object, depth: int = 0) -> list[object]:
    if depth > 8 or value is None:
        return []
    collected = [value]
    if isinstance(value, list):
        for item in value:
            collected.extend(collect_nested_values(item, depth + 1))
        return collected
    if isinstance(value, dict):
        for item in value.values():
            collected.extend(collect_nested_values(item, depth + 1))
    return collected


def is_structured_output_candidate(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in SEMANTIC_OUTPUT_KEYS)


def extract_structured_output_from_value(value: object) -> StructuredOutput | None:
    for candidate in collect_nested_values(value):
        if is_structured_output_candidate(candidate):
            return candidate
        if isinstance(candidate, str):
            parsed = parse_json_from_text(candidate)
            if is_structured_output_candidate(parsed):
                return parsed
    return None


def extract_structured_output_from_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        parsed = read_json(path)
    except Exception:
        parsed = parse_json_from_text(path.read_text(encoding="utf-8"))
    return extract_structured_output_from_value(parsed)


def extract_structured_output_from_jsonl_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    latest = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                parsed = parse_json_from_text(line)
            structured = extract_structured_output_from_value(parsed)
            if structured:
                latest = structured
    return latest


@lru_cache(maxsize=32)
def _read_schema(path: str) -> dict[str, object]:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON schema is not an object: {path}")
    return payload


def _schema_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _schema_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _validate_json_schema_subset(value: object, schema: object, *, path: str = "$") -> list[str]:
    """Validate the JSON-schema subset used by coding-agent output schemas."""

    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected_type = schema.get("type")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: expected one of {enum_values!r}, got {value!r}")

    if expected_type == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {_schema_type_name(value)}")
            return errors
        required = [str(item) for item in schema.get("required") or []]
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties")
        property_schemas = properties if isinstance(properties, dict) else {}
        if schema.get("additionalProperties") is False:
            extra_keys = sorted(str(key) for key in value if str(key) not in property_schemas)
            for key in extra_keys:
                errors.append(f"{path}: unexpected property {key!r}")
        for key, child_schema in property_schemas.items():
            if key in value:
                errors.extend(_validate_json_schema_subset(value[key], child_schema, path=_schema_path(path, str(key))))
        return errors

    if expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {_schema_type_name(value)}")
            return errors
        item_schema = schema.get("items")
        for index, item in enumerate(value):
            errors.extend(_validate_json_schema_subset(item, item_schema, path=f"{path}[{index}]"))
        return errors

    if expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {_schema_type_name(value)}")
        return errors

    if expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: expected integer, got {_schema_type_name(value)}")
        return errors

    return errors


def structured_output_schema_error(value: object, schema_path: Path | None) -> str | None:
    if schema_path is None:
        return None
    try:
        schema = _read_schema(str(schema_path.resolve()))
    except Exception as exc:
        return f"Unable to load structured output schema: {exc}"
    errors = _validate_json_schema_subset(value, schema)
    if not errors:
        return None
    preview = "; ".join(errors[:5])
    if len(errors) > 5:
        preview += f"; ... and {len(errors) - 5} more"
    return f"Structured output failed schema validation: {preview}"


def build_codex_raw_response(events_path: Path, final_output_path: Path | None) -> CodexRawResponse:
    raw_response: CodexRawResponse = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": read_jsonl_values(events_path) if events_path.exists() else [],
    }
    if final_output_path and final_output_path.exists():
        raw_response["final_message"] = read_json_or_text(final_output_path)
    return raw_response


def _read_claude_stream_json(path: Path) -> list[object]:
    events: list[object] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append(line)
    return events


def build_claude_raw_response(raw_output_path: Path) -> ClaudeRawResponse:
    response_format = "stream-json" if raw_output_path.suffix.lower() == ".jsonl" else "json"
    response: object | None = None
    if raw_output_path.exists():
        try:
            parsed = read_json(raw_output_path)
        except Exception:
            response = _read_claude_stream_json(raw_output_path)
            response_format = "stream-json"
        else:
            if response_format == "stream-json":
                response = parsed if isinstance(parsed, list) else [parsed]
            else:
                response = parsed
    return {
        "agent": "claude",
        "response_format": response_format,
        "response": response,
    }
