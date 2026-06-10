# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-09 from upstream ContextBench.
# Summary of changes: keep Claude trace inference conservative for grep-derived context.

"""Claude-specific parsing for wrapper-produced records and raw responses."""

from __future__ import annotations

from ..base import BaseCodingAgentParser
from ...coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from ...coding_agents.response_parsing import extract_structured_output_from_value
from ...coding_agents.trace_inference import (
    infer_read_step,
    infer_retrieval_step_from_command,
    infer_retrieval_step_from_tool_result,
    infer_search_file_step_from_path,
    normalize_workspace_path,
    tool_result_text_from_value,
    trajectory_from_steps,
)
from ...coding_agents.types import ClaudeRawResponse, StructuredOutput, TokenUsage, ToolCall, TraceInferenceMeta, TrajectoryData


_NON_RETRIEVAL_TOOLS = frozenset(
    {
        "Edit",
        "MultiEdit",
        "Write",
        "NotebookEdit",
        "TodoWrite",
    }
)


def _bounded_tool_output_text(text: str, *, meta: TraceInferenceMeta) -> str:
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text
    meta["dropped_large_command_outputs"] = int(meta.get("dropped_large_command_outputs", 0) or 0) + 1
    return ""


class ClaudeAgentParser(BaseCodingAgentParser):
    def extract_structured_output(self, raw_response: ClaudeRawResponse) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if isinstance(response, dict):
            result = response.get("result")
            structured = extract_structured_output_from_value(result)
            if structured:
                return structured
            structured = extract_structured_output_from_value(response)
            if structured:
                return structured
        elif isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "result":
                    result = item.get("result")
                    structured = extract_structured_output_from_value(result)
                    if structured:
                        return structured
                structured = extract_structured_output_from_value(item)
                if structured:
                    return structured
        return extract_structured_output_from_value(raw_response)

    def extract_token_usage(self, raw_response: ClaudeRawResponse) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
                    return self._build_usage(usage)
            return None
        if not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        return self._build_usage(usage)

    def _build_usage(self, usage: dict[str, object]) -> TokenUsage:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read_input_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or (input_tokens + output_tokens))
        result: TokenUsage = {
            "source": "claude.response.usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
        reasoning_tokens = int(usage.get("reasoning_tokens", 0) or usage.get("thinking_tokens", 0) or 0)
        output_details = usage.get("output_tokens_details")
        if not reasoning_tokens and isinstance(output_details, dict):
            reasoning_tokens = int(
                output_details.get("reasoning_tokens", 0)
                or output_details.get("thinking_tokens", 0)
                or 0
            )
        if reasoning_tokens:
            result["reasoning_tokens"] = reasoning_tokens
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            result["server_tool_use"] = server_tool_use
        return result

    def extract_tool_calls(self, raw_response: ClaudeRawResponse) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        response = raw_response.get("response")
        if isinstance(response, list):
            calls: list[ToolCall] = []
            pending_indices: dict[str, int] = {}
            for item in response:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "assistant":
                    message = item.get("message")
                    if not isinstance(message, dict):
                        continue
                    for content in message.get("content", []):
                        if not isinstance(content, dict) or content.get("type") != "tool_use":
                            continue
                        tool_id = str(content.get("id") or "").strip()
                        tool_name = str(content.get("name") or "unknown").strip() or "unknown"
                        payload: dict[str, object] = {
                            "id": tool_id,
                            "input": dict(content.get("input") or {}) if isinstance(content.get("input"), dict) else {},
                        }
                        if tool_name.startswith("mcp__"):
                            parts = tool_name.split("__", 2)
                            if len(parts) >= 3:
                                payload["mcp_server"] = parts[1]
                                payload["mcp_tool"] = parts[2]
                        pending_indices[tool_id] = len(calls)
                        calls.append(
                            {
                                "source": "claude.tool_use",
                                "tool_name": tool_name,
                                "payload": payload,
                            }
                        )
                    continue

                if item_type == "user":
                    message = item.get("message")
                    if not isinstance(message, dict):
                        continue
                    for content in message.get("content", []):
                        if not isinstance(content, dict) or content.get("type") != "tool_result":
                            continue
                        tool_use_id = str(content.get("tool_use_id") or "").strip()
                        call_index = pending_indices.get(tool_use_id)
                        if call_index is None:
                            continue
                        call_payload = calls[call_index]["payload"]
                        result_content = content.get("content")
                        call_payload["result"] = {
                            "is_error": bool(content.get("is_error")),
                            "content_chars": len(str(result_content or "")),
                        }

            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
                    server_tool_use = usage.get("server_tool_use")
                    if isinstance(server_tool_use, dict):
                        calls.append(
                            {
                                "source": "claude.server_tool_use",
                                "tool_name": "server_tool_use",
                                "payload": dict(server_tool_use),
                            }
                        )
                        break
            return calls
        if not isinstance(response, dict):
            return []
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return []
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            return [
                {
                    "source": "claude.server_tool_use",
                    "tool_name": "server_tool_use",
                    "payload": dict(server_tool_use),
                }
            ]
        return []

    def extract_available_tools(self, raw_response: ClaudeRawResponse) -> list[str]:
        if not isinstance(raw_response, dict):
            return []
        response = raw_response.get("response")
        if not isinstance(response, list):
            return []
        tools: list[str] = []
        seen: set[str] = set()
        for item in response:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "system" or item.get("subtype") != "init":
                continue
            for tool in item.get("tools") or []:
                name = str(tool or "").strip()
                if name and name not in seen:
                    tools.append(name)
                    seen.add(name)
        return tools

    def infer_trajectory_data(
        self,
        raw_response: ClaudeRawResponse,
        *,
        record: dict[str, object],
    ) -> TrajectoryData | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if not isinstance(response, list):
            return None
        workspace_path_value = str(record.get("workspace_path") or "").strip()
        if not workspace_path_value:
            return None
        from pathlib import Path

        workspace_path = Path(workspace_path_value)
        steps = []
        pending_tools: dict[str, tuple[str, dict[str, object]]] = {}
        meta: TraceInferenceMeta = {}

        for item in response:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "assistant":
                message = item.get("message")
                if not isinstance(message, dict):
                    continue
                for content in message.get("content", []):
                    if not isinstance(content, dict) or content.get("type") != "tool_use":
                        continue
                    tool_id = str(content.get("id") or "").strip()
                    tool_name = str(content.get("name") or "").strip()
                    tool_input = content.get("input")
                    if tool_id and isinstance(tool_input, dict):
                        pending_tools[tool_id] = (tool_name, dict(tool_input))
                        # Edit/Write targets are solution artifacts, not retrieval context.
                continue

            if item_type != "user":
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                continue
            for content in message.get("content", []):
                if not isinstance(content, dict) or content.get("type") != "tool_result":
                    continue
                tool_use_id = str(content.get("tool_use_id") or "").strip()
                tool_payload = pending_tools.get(tool_use_id)
                if not tool_payload:
                    continue
                tool_name, tool_input = tool_payload
                result_content = content.get("content")
                output_text = tool_result_text_from_value(result_content)
                if tool_name in _NON_RETRIEVAL_TOOLS:
                    continue
                if tool_name == "Read":
                    output_text = _bounded_tool_output_text(output_text, meta=meta)
                    file_path = str(tool_input.get("file_path") or "").strip()
                    if file_path:
                        steps.append(infer_read_step(file_path, output_text=output_text, workspace_path=workspace_path))
                    continue
                if tool_name == "Grep":
                    output_text = _bounded_tool_output_text(output_text, meta=meta)
                    step = infer_search_file_step_from_path(
                        str(tool_input.get("path") or "").strip(),
                        output_text=output_text,
                        workspace_path=workspace_path,
                    )
                    if step:
                        steps.append(step)
                    continue
                if tool_name == "Bash":
                    output_text = _bounded_tool_output_text(output_text, meta=meta)
                    command = str(tool_input.get("command") or "")
                    step = infer_retrieval_step_from_command(
                        command,
                        output_text=output_text,
                        workspace_path=workspace_path,
                        meta=meta,
                    )
                    if step:
                        steps.append(step)
                    continue
                step = infer_retrieval_step_from_tool_result(
                    result_content,
                    output_text=output_text,
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
