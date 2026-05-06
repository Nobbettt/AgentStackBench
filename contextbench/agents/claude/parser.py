
"""Claude-specific parsing for wrapper-produced records and raw responses."""

from __future__ import annotations

from ..base import BaseCodingAgentParser
from ...coding_agents.response_parsing import extract_structured_output_from_value, parse_json_from_text
from ...coding_agents.trace_inference import (
    infer_read_step,
    infer_retrieval_step_from_command,
    infer_grep_spans_from_text,
    infer_file_list_from_text,
    normalize_workspace_path,
    trajectory_from_steps,
)
from ...coding_agents.types import ClaudeRawResponse, StructuredOutput, TokenUsage, ToolCall, TraceInferenceMeta, TrajectoryData


class ClaudeAgentParser(BaseCodingAgentParser):
    def extract_structured_output(self, raw_response: ClaudeRawResponse) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if isinstance(response, dict):
            result = response.get("result")
            if isinstance(result, str):
                parsed = parse_json_from_text(result)
                if parsed:
                    return parsed
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
                    if isinstance(result, str):
                        parsed = parse_json_from_text(result)
                        if parsed:
                            return parsed
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
        result: TokenUsage = {
            "source": "claude.response.usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            result["server_tool_use"] = server_tool_use
        return result

    def extract_tool_calls(self, raw_response: ClaudeRawResponse) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        response = raw_response.get("response")
        if isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
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
                output_text = str(content.get("content") or "")
                if tool_name == "Read":
                    file_path = str(tool_input.get("file_path") or "").strip()
                    if file_path:
                        steps.append(infer_read_step(file_path, output_text=output_text, workspace_path=workspace_path))
                    continue
                if tool_name == "Grep":
                    spans = infer_grep_spans_from_text(output_text, workspace_path, meta=meta)
                    files = sorted(spans)
                    if not files:
                        path_value = str(tool_input.get("path") or "").strip()
                        if path_value and "." in Path(path_value).name:
                            files = [normalize_workspace_path(path_value, workspace_path)]
                            if files[0] not in spans:
                                spans = infer_grep_spans_from_text(output_text, workspace_path, meta=meta)
                        else:
                            files = infer_file_list_from_text(output_text, workspace_path, meta=meta)
                    if files or spans:
                        steps.append({"files": files, "spans": spans, "symbols": {}})
                    continue
                if tool_name == "Bash":
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
