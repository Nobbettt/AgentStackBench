
"""Response parsing helpers for coding-agent integrations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .constants import RICH_OUTPUT_KEYS, SEMANTIC_OUTPUT_KEYS
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
    return isinstance(value, dict) and (
        all(key in value for key in RICH_OUTPUT_KEYS)
        or all(key in value for key in SEMANTIC_OUTPUT_KEYS)
    )


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


def build_codex_raw_response(events_path: Path, final_output_path: Path | None) -> CodexRawResponse:
    raw_response: CodexRawResponse = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": read_jsonl_values(events_path) if events_path.exists() else [],
    }
    if final_output_path and final_output_path.exists():
        raw_response["final_message"] = read_json_or_text(final_output_path)
    return raw_response


def build_claude_raw_response(raw_output_path: Path) -> ClaudeRawResponse:
    return {
        "agent": "claude",
        "response_format": "json",
        "response": read_json_or_text(raw_output_path) if raw_output_path.exists() else None,
    }
