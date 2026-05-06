# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: add setup-run record helpers for unscored bootstrap prompts.

"""Record normalization helpers for coding-agent integrations."""

from __future__ import annotations

import re
import time
from pathlib import Path

from .types import (
    CommandResult,
    LineSpan,
    RetrievalStep,
    SetupRunRecord,
    SpanMap,
    StructuredOutput,
    SymbolMap,
    TaskRecord,
    TokenUsage,
    ToolCall,
)


def _maybe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def normalize_span_entry(value: dict[str, object]) -> LineSpan | None:
    if not isinstance(value, dict):
        return None
    start = _maybe_int(value.get("start", value.get("start_line")))
    end = _maybe_int(value.get("end", value.get("end_line")))
    if start is None or end is None:
        return None
    start = max(1, start)
    end = max(start, end)
    return {"start": start, "end": end}


def normalize_span_map(raw_value: object) -> SpanMap:
    if isinstance(raw_value, list):
        normalized: SpanMap = {}
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file") or "").strip()
            span = normalize_span_entry(item)
            if not file_path or not span:
                continue
            normalized.setdefault(file_path, []).append(span)
        return normalized

    if not isinstance(raw_value, dict):
        return {}

    normalized: SpanMap = {}
    for file_path, spans in raw_value.items():
        if not isinstance(spans, list):
            continue
        next_spans = [span for span in (normalize_span_entry(span) for span in spans) if span]
        if next_spans:
            normalized[str(file_path)] = next_spans
    return normalized


def normalize_symbol_map(raw_value: object) -> SymbolMap:
    if isinstance(raw_value, list):
        normalized: SymbolMap = {}
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file") or "").strip()
            name = str(item.get("name") or item.get("symbol") or "").strip()
            if not file_path or not name:
                continue
            normalized.setdefault(file_path, []).append(name)
        return {file_path: sorted(set(values)) for file_path, values in normalized.items() if values}

    if not isinstance(raw_value, dict):
        return {}

    normalized: SymbolMap = {}
    for file_path, symbols in raw_value.items():
        if not isinstance(symbols, list):
            continue
        next_symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
        if next_symbols:
            normalized[str(file_path)] = sorted(set(next_symbols))
    return normalized


def merge_span_maps(*maps: object) -> SpanMap:
    merged: SpanMap = {}
    for raw_map in maps:
        for file_path, spans in normalize_span_map(raw_map).items():
            merged.setdefault(file_path, []).extend(spans)
    return merged


def parse_unified_diff(diff_text: str) -> SpanMap:
    spans_by_file: SpanMap = {}
    current_file: str | None = None
    for line in str(diff_text or "").splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if not match or not current_file:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        end = start + count - 1 if count > 0 else start
        spans_by_file.setdefault(current_file, []).append({"start": start, "end": end})
    return spans_by_file


def normalize_retrieval_steps(raw_steps: object) -> list[RetrievalStep]:
    if not isinstance(raw_steps, list):
        return []
    normalized_steps = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        files = sorted({str(item).strip() for item in (step.get("files") or []) if str(item).strip()})
        spans = normalize_span_map(step.get("spans"))
        symbols = normalize_symbol_map(step.get("symbols"))
        if files or spans or symbols:
            normalized_steps.append({"files": files, "spans": spans, "symbols": symbols})
    return normalized_steps


def build_task_record(
    *,
    task: dict[str, object],
    agent: str,
    workspace_path: Path,
    task_dir: Path,
    prompt_path: Path,
    command_result: CommandResult,
    structured_output: StructuredOutput | None,
    token_usage: TokenUsage | None,
    tool_calls: list[ToolCall] | None,
    raw_response_path: Path | None,
    diff_path: Path | None,
    model_patch: str,
    started_at: float,
    completed_at: float,
    setup_run: SetupRunRecord | None = None,
) -> TaskRecord:
    if command_result["timeout"]:
        status = "timeout"
    elif not command_result["ok"]:
        status = "failed"
    else:
        status = (structured_output or {}).get("status") or "completed"

    record: TaskRecord = {
        "agent": agent,
        "bench": task.get("bench"),
        "instance_id": task.get("instance_id"),
        "original_inst_id": task.get("original_inst_id") or None,
        "repo": task.get("repo") or None,
        "repo_url": task.get("repo_url") or None,
        "commit": task.get("commit") or None,
        "language": task.get("language") or None,
        "workspace_path": str(workspace_path),
        "task_dir": str(task_dir),
        "prompt_path": str(prompt_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
        "duration_ms": int((completed_at - started_at) * 1000),
        "timeout": bool(command_result["timeout"]),
        "exit_code": command_result["exit_code"],
        "signal": command_result["signal"],
        "ok": bool(command_result["ok"]),
        "status": status,
        "final_output": structured_output,
        "token_usage": token_usage,
        "tool_calls": tool_calls or [],
        "raw_response_path": str(raw_response_path) if raw_response_path else None,
        "diff_path": str(diff_path) if diff_path else None,
        "model_patch": model_patch,
    }
    if setup_run is not None:
        record["setup_run"] = setup_run
    return record


def build_setup_run_record(
    *,
    prompt_path: Path,
    stderr_path: Path,
    command_result: CommandResult,
    raw_response_path: Path | None,
    token_usage: TokenUsage | None,
    tool_calls: list[ToolCall] | None,
    started_at: float,
    completed_at: float,
) -> SetupRunRecord:
    return {
        "prompt_path": str(prompt_path),
        "stderr_path": str(stderr_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
        "duration_ms": int((completed_at - started_at) * 1000),
        "timeout": bool(command_result["timeout"]),
        "exit_code": command_result["exit_code"],
        "signal": command_result["signal"],
        "ok": bool(command_result["ok"]),
        "status": "timeout" if command_result["timeout"] else ("completed" if command_result["ok"] else "failed"),
        "raw_response_path": str(raw_response_path) if raw_response_path else None,
        "token_usage": token_usage,
        "tool_calls": tool_calls or [],
    }
