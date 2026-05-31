
"""Codex-specific parsing for wrapper-produced records and raw responses."""

from __future__ import annotations

from ..base import BaseCodingAgentParser
from ...coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from ...coding_agents.response_parsing import extract_structured_output_from_value
from ...coding_agents.trace_inference import (
    infer_retrieval_step_from_command,
    infer_retrieval_step_from_tool_result,
    tool_result_text_from_value,
    trajectory_from_steps,
)
from ...coding_agents.types import CodexRawResponse, StructuredOutput, TokenUsage, ToolCall, TraceInferenceMeta, TrajectoryData


def _codex_tool_name(event: dict[str, object], item: dict[str, object] | None = None) -> str:
    sources = [source for source in (item, event) if isinstance(source, dict)]
    for source in sources:
        for key in ("tool_name", "toolName", "name"):
            value = str(source.get(key) or "").strip()
            if value.startswith("mcp__"):
                return value

    for source in sources:
        server = str(
            source.get("mcp_server")
            or source.get("mcpServer")
            or source.get("server_name")
            or source.get("serverName")
            or source.get("server")
            or ""
        ).strip()
        tool = str(
            source.get("mcp_tool")
            or source.get("mcpTool")
            or source.get("tool_name")
            or source.get("toolName")
            or source.get("name")
            or ""
        ).strip()
        if tool.startswith("mcp__"):
            return tool
        if server and tool:
            return f"mcp__{server}__{tool}"

    for source in sources:
        for key in ("tool_name", "toolName", "name", "type"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return "unknown"


def _codex_tool_payload(event: dict[str, object], item: dict[str, object]) -> dict[str, object]:
    payload = dict(item)
    payload.setdefault("event_type", event.get("type"))
    for key in ("status", "error", "is_error", "tool_name", "name"):
        if key not in payload and key in event:
            payload[key] = event[key]
    tool_name = _codex_tool_name(event, item)
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        if len(parts) >= 3:
            payload.setdefault("mcp_server", parts[1])
            payload.setdefault("mcp_tool", parts[2])
    return payload


class CodexAgentParser(BaseCodingAgentParser):
    def extract_structured_output(self, raw_response: CodexRawResponse) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        final_message = raw_response.get("final_message")
        if final_message is not None:
            structured = extract_structured_output_from_value(final_message)
            if structured:
                return structured
        return extract_structured_output_from_value(raw_response)

    def extract_token_usage(self, raw_response: CodexRawResponse) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        events = raw_response.get("events")
        if not isinstance(events, list):
            return None
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
            if not cached_input_tokens:
                details = usage.get("input_tokens_details")
                if isinstance(details, dict):
                    cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
            reasoning_tokens = 0
            output_details = usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)
            result: TokenUsage = {
                "source": "codex.turn.completed",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "total_tokens": int(usage.get("total_tokens", 0) or (input_tokens + output_tokens)),
            }
            if cached_input_tokens:
                result["cache_read_input_tokens"] = cached_input_tokens
            if reasoning_tokens:
                result["reasoning_tokens"] = reasoning_tokens
            return result
        return None

    def extract_tool_calls(self, raw_response: CodexRawResponse) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        events = raw_response.get("events")
        if not isinstance(events, list):
            return []
        calls: list[ToolCall] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            event_type_lower = event_type.lower()
            item = event.get("item")
            if isinstance(item, dict):
                item_type_lower = str(item.get("type") or "").lower()
                if (
                    "tool" in item_type_lower
                    or "mcp" in item_type_lower
                    or "tool" in event_type_lower
                    or "mcp" in event_type_lower
                ):
                    calls.append(
                        {
                            "source": "codex.item",
                            "tool_name": _codex_tool_name(event, item),
                            "payload": _codex_tool_payload(event, item),
                        }
                    )
                    continue
            if "tool" not in event_type_lower and "mcp" not in event_type_lower:
                continue
            calls.append(
                {
                    "source": "codex.event",
                    "tool_name": _codex_tool_name(event),
                    "payload": dict(event),
                }
            )
        return calls

    def infer_trajectory_data(
        self,
        raw_response: CodexRawResponse,
        *,
        record: dict[str, object],
    ) -> TrajectoryData | None:
        if not isinstance(raw_response, dict):
            return None
        events = raw_response.get("events")
        if not isinstance(events, list):
            return None
        workspace_path_value = str(record.get("workspace_path") or "").strip()
        if not workspace_path_value:
            return None
        from pathlib import Path

        workspace_path = Path(workspace_path_value)
        steps = []
        meta: TraceInferenceMeta = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "command_execution":
                if str(item.get("status") or "completed") != "completed":
                    continue
                exit_code = item.get("exit_code")
                if exit_code not in (0, "0", None):
                    continue
                command = str(item.get("command") or "")
                output_text = str(item.get("aggregated_output") or "")
                if len(output_text) > MAX_COMMAND_OUTPUT_CHARS:
                    meta["dropped_large_command_outputs"] = int(meta.get("dropped_large_command_outputs", 0) or 0) + 1
                    output_text = ""
                step = infer_retrieval_step_from_command(
                    command,
                    output_text=output_text,
                    workspace_path=workspace_path,
                    meta=meta,
                )
                if step:
                    steps.append(step)
                continue
            event_type = str(event.get("type") or "").lower()
            if (
                "tool" in item_type.lower()
                or "mcp" in item_type.lower()
                or "tool" in event_type
                or "mcp" in event_type
            ):
                if str(item.get("status") or "completed") not in {"completed", "success", "succeeded", ""}:
                    continue
                if bool(item.get("is_error") or item.get("error")):
                    continue
                result_value = item.get("result")
                if result_value is None:
                    result_value = item.get("output")
                if result_value is None:
                    result_value = item.get("content")
                if result_value is None:
                    result_value = item
                output_text = tool_result_text_from_value(result_value)
                step = infer_retrieval_step_from_tool_result(
                    result_value,
                    output_text=output_text,
                    workspace_path=workspace_path,
                    meta=meta,
                )
                if step:
                    steps.append(step)
                continue
            # File-change events are solution artifacts, not retrieval context.
            # They must not be scored as files the agent inspected.

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
