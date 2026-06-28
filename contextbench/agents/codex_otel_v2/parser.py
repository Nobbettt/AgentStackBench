# SPDX-License-Identifier: Apache-2.0

"""Parser for Codex OTEL v2 raw responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..base import BaseOtelAgentParser
from ...coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from ...coding_agents.trace_inference import (
    infer_read_span_from_text,
    infer_retrieval_step_from_command,
    infer_retrieval_step_from_tool_result,
    tool_result_text_from_value,
    trajectory_from_steps,
)
from ...coding_agents.types import (
    RetrievalStep,
    StructuredOutput,
    TokenUsage,
    ToolCall,
    TraceInferenceMeta,
    TrajectoryData,
)


_TOKEN_KEYS = {
    "input_tokens": ("input_tokens", "input_token_count", "input.token_count", "codex.input_tokens", "llm.input_tokens"),
    "output_tokens": ("output_tokens", "output_token_count", "output.token_count", "codex.output_tokens", "llm.output_tokens"),
    "cached_input_tokens": ("cached_input_tokens", "cached_token_count", "cached_input.token_count", "codex.cached_input_tokens"),
    "reasoning_tokens": ("reasoning_output_tokens", "reasoning_token_count", "reasoning_tokens", "codex.reasoning_output_tokens"),
    "total_tokens": ("total_tokens", "tool_token_count", "codex.total_tokens"),
}
_TOOL_NAME_KEYS = (
    "tool.name",
    "tool_name",
    "codex.tool.name",
    "codex.tool_name",
    "mcp.tool.name",
    "mcp_tool",
    "name",
)
_MCP_SERVER_KEYS = ("mcp_server", "mcp.server", "mcp.server.name", "server_name", "serverName", "server")
_MCP_TOOL_KEYS = ("mcp_tool", "mcp.tool", "mcp.tool.name", "mcp_tool_name", "mcpTool")
_V2_REQUIRED_FINAL_OUTPUT_KEYS = ("status", "final_answer", "notes")


def _events(raw_response: dict[str, object], key: str) -> list[dict[str, object]]:
    otel = raw_response.get("otel")
    if not isinstance(otel, dict):
        return []
    values = otel.get(key)
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _all_otel_events(raw_response: dict[str, object]) -> list[dict[str, object]]:
    return [*_events(raw_response, "logs"), *_events(raw_response, "traces"), *_events(raw_response, "metrics")]


def _attributes(event: dict[str, object]) -> dict[str, object]:
    attrs = event.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _body(event: dict[str, object]) -> object:
    return event.get("body")


def _event_name(event: dict[str, object]) -> str:
    name = str(event.get("name") or "").strip()
    if name:
        return name
    body = _body(event)
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, dict):
        for key in ("event.name", "event_name", "name"):
            value = str(body.get(key) or "").strip()
            if value:
                return value
    return ""


def _is_tool_result_event_name(name: str) -> bool:
    return name in {"codex.tool_result", "tool_result", "tool.result"} or name.endswith(".tool_result")


def _is_tool_decision_event_name(name: str) -> bool:
    return name in {"codex.tool_decision", "tool_decision", "tool.decision"} or name.endswith(".tool_decision")


def _v2_structured_output_candidate(value: object) -> StructuredOutput | None:
    if not isinstance(value, dict):
        return None
    if not all(key in value for key in _V2_REQUIRED_FINAL_OUTPUT_KEYS):
        return None
    structured: StructuredOutput = {
        "status": value["status"],
        "final_answer": value["final_answer"],
        "notes": value["notes"],
    }
    return structured


def _extract_v2_structured_output_from_final_message(value: object) -> StructuredOutput | None:
    return _v2_structured_output_candidate(value)


def _lookup_int(attrs: dict[str, object], keys: Iterable[str]) -> int:
    lower_attrs = {str(key).lower(): value for key, value in attrs.items()}
    for key in keys:
        value = lower_attrs.get(key.lower())
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _string_attr(attrs: dict[str, object], *keys: str) -> str:
    lower_attrs = {str(key).lower(): value for key, value in attrs.items()}
    for key in keys:
        value = lower_attrs.get(key.lower())
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _success_from_attrs(attrs: dict[str, object]) -> bool:
    for key in ("success", "ok", "tool.success", "codex.tool.success"):
        if key in attrs:
            value = attrs[key]
            if isinstance(value, str):
                return value.strip().lower() not in {"0", "false", "no", "off"}
            return bool(value)
    status = _string_attr(attrs, "status", "tool.status", "codex.tool.status").lower()
    if status in {"completed", "complete", "success", "succeeded", "ok"}:
        return True
    if status in {"cancelled", "canceled", "denied", "error", "failed", "failure", "rejected", "timeout"}:
        return False
    return False


def _json_object_from_attr(attrs: dict[str, object], key: str) -> dict[str, object]:
    value = attrs.get(key)
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_qualified_mcp_tool_name(value: str) -> tuple[str, str] | None:
    if not value.startswith("mcp__"):
        return None
    parts = value.split("__", 2)
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def _mcp_tool_identity(attrs: dict[str, object]) -> tuple[str, str, str] | None:
    server = _string_attr(attrs, *_MCP_SERVER_KEYS)
    explicit_tool = _string_attr(attrs, *_MCP_TOOL_KEYS)
    raw_tool = _string_attr(attrs, *_TOOL_NAME_KEYS)
    for candidate in (explicit_tool, raw_tool):
        qualified = _parse_qualified_mcp_tool_name(candidate)
        if qualified:
            qualified_server, qualified_tool = qualified
            return candidate, qualified_server, qualified_tool
    tool = explicit_tool or raw_tool
    if server and tool:
        return f"mcp__{server}__{tool}", server, tool
    return None


def _tool_name(event: dict[str, object]) -> str:
    attrs = _attributes(event)
    mcp_identity = _mcp_tool_identity(attrs)
    if mcp_identity:
        return mcp_identity[0]
    value = _string_attr(attrs, *_TOOL_NAME_KEYS)
    if value:
        return value
    event_name = _event_name(event)
    if event_name.startswith("codex."):
        return event_name
    return "unknown"


def _tool_payload(event: dict[str, object]) -> dict[str, object]:
    attrs = dict(_attributes(event))
    mcp_identity = _mcp_tool_identity(attrs)
    arguments = _json_object_from_attr(attrs, "arguments")
    if arguments:
        attrs["arguments_json"] = arguments
        command = str(arguments.get("cmd") or "").strip()
        if command:
            attrs.setdefault("command", command)
        workdir = str(arguments.get("workdir") or "").strip()
        if workdir:
            attrs.setdefault("workdir", workdir)
    body = _body(event)
    if isinstance(body, dict):
        attrs.setdefault("body", body)
    elif isinstance(body, str) and body:
        attrs.setdefault("body", body)
    attrs.setdefault("otel_event_name", _event_name(event))
    attrs.setdefault("otel_signal", event.get("signal"))
    attrs.setdefault("ok", _success_from_attrs(attrs))
    if mcp_identity:
        _, server, tool = mcp_identity
        if not str(attrs.get("mcp_server") or "").strip():
            attrs["mcp_server"] = server
        existing_mcp_tool = str(attrs.get("mcp_tool") or "").strip()
        if not existing_mcp_tool or existing_mcp_tool.startswith("mcp__"):
            attrs["mcp_tool"] = tool
    return attrs


def _tool_stdout_text(output_text: str) -> str:
    marker = "\nOutput:\n"
    if marker in output_text:
        return output_text.rsplit(marker, 1)[1]
    if output_text.startswith("Output:\n"):
        return output_text[len("Output:\n") :]
    return output_text


def _counted_output_span(stdout_text: str, start: int, end: int) -> tuple[int, int] | None:
    if not stdout_text:
        return None
    line_count = len(stdout_text.rstrip("\n").splitlines())
    if line_count <= 0:
        return None
    counted_end = start + line_count - 1
    if counted_end > end:
        return None
    return start, counted_end


def _refine_command_step_from_output(step: RetrievalStep, output_text: str) -> RetrievalStep:
    stdout_text = _tool_stdout_text(output_text)
    output_span = infer_read_span_from_text(stdout_text)
    files = list(step.get("files", []))
    if len(files) != 1:
        return step
    file_path = files[0]
    spans = step.get("spans", {})
    existing_spans = spans.get(file_path, [])
    if len(existing_spans) != 1:
        return step
    existing = existing_spans[0]
    try:
        existing_start = int(existing.get("start") or 0)
        existing_end = int(existing.get("end") or existing_start)
    except (TypeError, ValueError):
        return step
    if output_span is None:
        output_span = _counted_output_span(stdout_text, existing_start, existing_end)
        if output_span is None:
            return step
    if not (existing_start <= output_span[0] <= output_span[1] <= existing_end):
        return step
    refined_spans = {key: [dict(span) for span in value] for key, value in spans.items() if key != file_path}
    refined_spans[file_path] = [{"start": output_span[0], "end": output_span[1]}]
    return {
        "files": files,
        "spans": refined_spans,
        "symbols": {key: list(value) for key, value in step.get("symbols", {}).items()},
    }


class CodexOtelV2AgentParser(BaseOtelAgentParser):
    def extract_structured_output(self, raw_response: dict[str, object]) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        final_message = raw_response.get("final_message")
        if final_message is not None:
            return _extract_v2_structured_output_from_final_message(final_message)
        return None

    def extract_token_usage(self, raw_response: dict[str, object]) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        completion_events = []
        for event in _all_otel_events(raw_response):
            name = _event_name(event)
            attrs = _attributes(event)
            kind = _string_attr(attrs, "event.kind", "event_kind", "sse.event", "sse_event", "kind", "type")
            if name == "codex.sse_event" and "completed" in kind:
                completion_events.append(event)
            elif name in {"response.completed", "codex.response.completed"}:
                completion_events.append(event)
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        reasoning_tokens = 0
        total_tokens = 0
        matched = False
        for event in completion_events:
            attrs = _attributes(event)
            next_input_tokens = _lookup_int(attrs, _TOKEN_KEYS["input_tokens"])
            next_output_tokens = _lookup_int(attrs, _TOKEN_KEYS["output_tokens"])
            next_cached_input_tokens = _lookup_int(attrs, _TOKEN_KEYS["cached_input_tokens"])
            next_reasoning_tokens = _lookup_int(attrs, _TOKEN_KEYS["reasoning_tokens"])
            next_total_tokens = _lookup_int(attrs, _TOKEN_KEYS["total_tokens"]) or next_input_tokens + next_output_tokens
            if not any(
                (
                    next_input_tokens,
                    next_output_tokens,
                    next_cached_input_tokens,
                    next_reasoning_tokens,
                    next_total_tokens,
                )
            ):
                continue
            matched = True
            input_tokens += next_input_tokens
            output_tokens += next_output_tokens
            cached_input_tokens += next_cached_input_tokens
            reasoning_tokens += next_reasoning_tokens
            total_tokens += next_total_tokens
        if not matched:
            return None
        usage: TokenUsage = {
            "source": "codex-otel-v2.sse_event",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "total_tokens": total_tokens or input_tokens + output_tokens,
        }
        if cached_input_tokens:
            usage["cache_read_input_tokens"] = cached_input_tokens
        if reasoning_tokens:
            usage["reasoning_tokens"] = reasoning_tokens
        return usage

    def extract_tool_calls(self, raw_response: dict[str, object]) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        calls_by_key: dict[str, tuple[bool, ToolCall]] = {}
        generic_calls: list[ToolCall] = []
        for index, event in enumerate(_all_otel_events(raw_response)):
            name = _event_name(event)
            is_result = _is_tool_result_event_name(name)
            is_decision = _is_tool_decision_event_name(name)
            if not (is_result or is_decision or name in {"codex.tool", "tool"}):
                continue
            tool_name = _tool_name(event)
            payload = _tool_payload(event)
            call: ToolCall = {
                "source": f"codex-otel-v2.{event.get('signal') or 'event'}",
                "tool_name": tool_name,
                "payload": payload,
            }
            if not (is_result or is_decision):
                generic_calls.append(call)
                continue
            call_id = str(payload.get("call_id") or payload.get("tool_call_id") or "").strip()
            command_or_args = str(payload.get("command") or payload.get("arguments") or "").strip()
            key = call_id or (f"{tool_name}:{command_or_args}" if command_or_args else f"{index}:{name}:{tool_name}")
            existing = calls_by_key.get(key)
            if is_result or existing is None:
                calls_by_key[key] = (is_result, call)
        return [call for _, call in calls_by_key.values()] + generic_calls

    def has_successful_tool_result_event(self, raw_response: dict[str, object]) -> bool:
        if not isinstance(raw_response, dict):
            return False
        for event in _all_otel_events(raw_response):
            name = _event_name(event)
            if _is_tool_result_event_name(name) and _success_from_attrs(_attributes(event)):
                return True
        return False

    def infer_trajectory_data(
        self,
        raw_response: dict[str, object],
        *,
        record: dict[str, object],
    ) -> TrajectoryData | None:
        if not isinstance(raw_response, dict):
            return None
        workspace_path_value = str(record.get("workspace_path") or "").strip()
        if not workspace_path_value:
            return None
        workspace_path = Path(workspace_path_value)
        steps = []
        meta: TraceInferenceMeta = {}
        for event in _all_otel_events(raw_response):
            name = _event_name(event)
            is_result = _is_tool_result_event_name(name)
            if not is_result:
                continue
            attrs = _attributes(event)
            if not _success_from_attrs(attrs):
                continue
            command = _string_attr(attrs, "command", "tool.command", "codex.tool.command", "shell.command")
            if not command:
                command = str(_json_object_from_attr(attrs, "arguments").get("cmd") or "").strip()
            output_text = _string_attr(
                attrs,
                "output",
                "tool.output",
                "codex.tool.output",
            )
            if len(output_text) > MAX_COMMAND_OUTPUT_CHARS:
                meta["dropped_large_command_outputs"] = int(meta.get("dropped_large_command_outputs", 0) or 0) + 1
                output_text = ""
            step = None
            if command:
                step = infer_retrieval_step_from_command(
                    command,
                    output_text=output_text,
                    workspace_path=workspace_path,
                    meta=meta,
                )
                if step:
                    step = _refine_command_step_from_output(step, output_text)
            if step is None:
                result_value: object = attrs
                body = _body(event)
                if isinstance(body, (dict, str)):
                    result_value = body
                step = infer_retrieval_step_from_tool_result(
                    result_value,
                    output_text=output_text or tool_result_text_from_value(result_value),
                    workspace_path=workspace_path,
                    meta=meta,
                )
            if step:
                steps.append(step)
        traj = trajectory_from_steps(steps)
        if traj is None:
            if not meta:
                return None
            return {
                "pred_steps": [],
                "pred_files": [],
                "pred_spans": {},
                "pred_symbols": {},
                "trace_inference_meta": meta,
            }
        if meta:
            traj["trace_inference_meta"] = meta
        return traj
