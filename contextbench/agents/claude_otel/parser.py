# SPDX-License-Identifier: Apache-2.0

"""Claude v2 parser backed by OpenTelemetry capture."""

from __future__ import annotations

from pathlib import Path

from ..base import BaseOtelAgentParser
from ...coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from .otel import (
    extract_claude_otel_structured_output_from_value,
    extract_api_response_body_summaries,
    extract_available_tools_from_api_bodies,
    extract_log_records,
    extract_spans,
    extract_tool_results_from_api_bodies,
    extract_tool_uses_from_api_bodies,
)
from ...coding_agents.trace_inference import (
    infer_read_step,
    infer_retrieval_step_from_command,
    infer_retrieval_step_from_tool_result,
    infer_search_file_step_from_path,
    tool_result_text_from_value,
    trajectory_from_steps,
)
from ...coding_agents.types import StructuredOutput, TokenUsage, ToolCall, TraceInferenceMeta, TrajectoryData


_NON_RETRIEVAL_TOOLS = frozenset(
    {
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "StructuredOutput",
        "TodoWrite",
        "Write",
    }
)
_REJECTED_DECISIONS = frozenset({"block", "blocked", "deny", "denied", "reject", "rejected"})


def _int_value(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cost_usd_from_attrs(attrs: dict[str, object]) -> float:
    cost_usd = _float_value(attrs.get("cost_usd"))
    if cost_usd:
        return cost_usd
    cost_micros = _float_value(attrs.get("cost_usd_micros"))
    return cost_micros / 1_000_000 if cost_micros else 0.0


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _contains_nested_placeholder(value: object) -> bool:
    if value == "<nested>":
        return True
    if isinstance(value, list):
        return any(_contains_nested_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nested_placeholder(item) for item in value.values())
    return False


def _bounded_tool_output_text(text: str, *, meta: TraceInferenceMeta) -> str:
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text
    meta["dropped_large_command_outputs"] = int(meta.get("dropped_large_command_outputs", 0) or 0) + 1
    return ""


def _merge_server_tool_use(target: dict[str, int], value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        count = _int_value(item)
        if count or key not in target:
            target[str(key)] = int(target.get(str(key), 0)) + count


def _merge_model_usage_value(
    target: dict[str, dict[str, int | float]],
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    if not model:
        return
    bucket = target.setdefault(
        model,
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    )
    bucket["input_tokens"] = int(bucket.get("input_tokens", 0)) + input_tokens
    bucket["output_tokens"] = int(bucket.get("output_tokens", 0)) + output_tokens
    bucket["cache_read_input_tokens"] = int(bucket.get("cache_read_input_tokens", 0)) + cache_read_input_tokens
    bucket["cache_creation_input_tokens"] = (
        int(bucket.get("cache_creation_input_tokens", 0)) + cache_creation_input_tokens
    )
    if cost_usd:
        bucket["cost_usd"] = float(bucket.get("cost_usd", 0.0)) + cost_usd


class ClaudeOtelAgentParser(BaseOtelAgentParser):
    """Parse Claude Code OTEL artifacts."""

    def extract_structured_output(self, raw_response: object) -> StructuredOutput | None:
        if isinstance(raw_response, dict):
            structured = self._structured_output_from_otel(raw_response)
            if structured is not None:
                return structured
        # Avoid recursively scanning the OTEL record after the OTEL-specific
        # paths rejected placeholder-truncated values such as "<nested>".
        return None

    def _structured_output_from_otel(self, raw_response: dict[str, object]) -> StructuredOutput | None:
        for item in reversed(extract_tool_uses_from_api_bodies(raw_response)):
            if item.get("name") != "StructuredOutput":
                continue
            structured = extract_claude_otel_structured_output_from_value(item.get("input"))
            if structured is not None and not _contains_nested_placeholder(structured):
                return structured
        return None

    def extract_token_usage(self, raw_response: object) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        return self._usage_from_api_response_bodies(raw_response)

    def _usage_from_api_response_bodies(self, raw_response: dict[str, object]) -> TokenUsage | None:
        responses = [
            item
            for item in extract_api_response_body_summaries(raw_response)
            if isinstance(item.get("usage"), dict)
        ]
        if not responses:
            return None

        input_tokens = output_tokens = cache_read = cache_creation = reasoning_tokens = 0
        server_tool_use: dict[str, int] = {}
        final_usage: dict[str, object] | None = None
        model_usage: dict[str, dict[str, int | float]] = {}

        for response in responses:
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            input_value = _int_value(usage.get("input_tokens"))
            output_value = _int_value(usage.get("output_tokens"))
            cache_read_value = _int_value(usage.get("cache_read_input_tokens"))
            cache_creation_value = _int_value(usage.get("cache_creation_input_tokens"))
            input_tokens += input_value
            output_tokens += output_value
            cache_read += cache_read_value
            cache_creation += cache_creation_value
            output_details = usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                reasoning_tokens += _int_value(
                    output_details.get("reasoning_tokens") or output_details.get("thinking_tokens")
                )
            reasoning_tokens += _int_value(usage.get("reasoning_tokens") or usage.get("thinking_tokens"))
            _merge_server_tool_use(server_tool_use, usage.get("server_tool_use"))
            final_usage = usage

            model = str(response.get("model") or "").strip()
            if model:
                _merge_model_usage_value(
                    model_usage,
                    model,
                    input_tokens=input_value,
                    output_tokens=output_value,
                    cache_read_input_tokens=cache_read_value,
                    cache_creation_input_tokens=cache_creation_value,
                )

        cost_usd = self._cost_from_api_request_logs(raw_response, model_usage=model_usage)
        result: TokenUsage = {
            "source": "claude.otel.api_response_body+api_request_cost" if cost_usd else "claude.otel.api_response_body",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }
        if cost_usd:
            result["cost_usd"] = cost_usd
        if reasoning_tokens:
            result["reasoning_tokens"] = reasoning_tokens
        if server_tool_use:
            result["server_tool_use"] = server_tool_use  # type: ignore[typeddict-item]
        if final_usage is not None:
            last_input_tokens = _int_value(final_usage.get("input_tokens"))
            last_output_tokens = _int_value(final_usage.get("output_tokens"))
            result["last_api_response_input_tokens"] = last_input_tokens
            result["last_api_response_output_tokens"] = last_output_tokens
            result["last_api_response_total_tokens"] = last_input_tokens + last_output_tokens
            result["last_api_response_cache_read_input_tokens"] = _int_value(
                final_usage.get("cache_read_input_tokens")
            )
            result["last_api_response_cache_creation_input_tokens"] = _int_value(
                final_usage.get("cache_creation_input_tokens")
            )
        result["api_response_count"] = len(responses)
        if model_usage:
            result["model_usage"] = model_usage
        return result

    def _cost_from_api_request_logs(
        self,
        raw_response: dict[str, object],
        *,
        model_usage: dict[str, dict[str, int | float]] | None = None,
    ) -> float:
        cost_usd = 0.0
        for log in extract_log_records(raw_response):
            if log.get("name") != "claude_code.api_request":
                continue
            attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
            item_cost = _cost_usd_from_attrs(attrs)
            cost_usd += item_cost
            if model_usage is not None:
                model = str(attrs.get("model") or "").strip()
                _merge_model_usage_value(model_usage, model, cost_usd=item_cost)
        return cost_usd

    def extract_tool_calls(self, raw_response: object) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        return self._tool_calls_from_api_bodies(raw_response)

    def _tool_calls_from_api_bodies(self, raw_response: dict[str, object]) -> list[ToolCall]:
        tool_uses = extract_tool_uses_from_api_bodies(raw_response)
        if not tool_uses:
            return []

        logs_by_id = self._tool_result_log_attrs_by_id(raw_response)
        decisions_by_id = self._tool_decision_log_attrs_by_id(raw_response)
        spans_by_id = self._tool_span_attrs_by_id(raw_response)
        result_by_id = self._tool_result_info_by_id(raw_response)
        calls: list[ToolCall] = []

        for item in tool_uses:
            tool_use_id = str(item.get("id") or "").strip()
            tool_name = str(item.get("name") or "unknown").strip() or "unknown"
            tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
            log_attrs = logs_by_id.get(tool_use_id, {})
            decision_attrs = decisions_by_id.get(tool_use_id, {})
            span_attrs = spans_by_id.get(tool_use_id, {})
            payload: dict[str, object] = {
                "id": tool_use_id,
                "input": dict(tool_input),
                "request_id": item.get("request_id"),
                "message_id": item.get("message_id"),
            }

            self._apply_tool_decision(payload, decision_attrs)
            result_info = result_by_id.get(tool_use_id)
            result_is_error = None
            if result_info is None:
                payload["ok"] = False
                payload["status"] = "missing_result"
            else:
                result_is_error = _bool_value(result_info.get("is_error"))
                if result_is_error is not None:
                    payload["ok"] = not result_is_error
            duration_ms = _int_value(log_attrs.get("duration_ms") or span_attrs.get("duration_ms"))
            if duration_ms:
                payload["duration_ms"] = duration_ms
            for key in ("tool_input_size_bytes", "tool_result_size_bytes"):
                value = _int_value(log_attrs.get(key))
                if value:
                    payload[key] = value
            result_tokens = _int_value(span_attrs.get("result_tokens"))
            if result_tokens:
                payload["result_tokens"] = result_tokens
            if tool_name.startswith("mcp__"):
                parts = tool_name.split("__", 2)
                if len(parts) >= 3:
                    payload["mcp_server"] = parts[1]
                    payload["mcp_tool"] = parts[2]
            if result_info is not None:
                output_text = str(result_info.get("text") or "")
                payload["result"] = {
                    "content_chars": len(output_text),
                    "is_error": bool(result_is_error),
                }

            calls.append({"source": "claude.otel.api_body_tool_use", "tool_name": tool_name, "payload": payload})

        server_tool_use, saw_server_tool_use = self._server_tool_use_from_api_bodies(raw_response)
        if saw_server_tool_use:
            calls.append(
                {
                    "source": "claude.otel.server_tool_use",
                    "tool_name": "server_tool_use",
                    "payload": dict(server_tool_use),
                }
            )
        return calls

    def _tool_result_log_attrs_by_id(self, raw_response: dict[str, object]) -> dict[str, dict[str, object]]:
        by_id: dict[str, dict[str, object]] = {}
        for log in extract_log_records(raw_response):
            if log.get("name") != "claude_code.tool_result":
                continue
            attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
            tool_use_id = str(attrs.get("tool_use_id") or "").strip()
            if tool_use_id:
                by_id[tool_use_id] = attrs
        return by_id

    def _tool_decision_log_attrs_by_id(self, raw_response: dict[str, object]) -> dict[str, dict[str, object]]:
        by_id: dict[str, dict[str, object]] = {}
        for log in extract_log_records(raw_response):
            if log.get("name") != "claude_code.tool_decision":
                continue
            attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
            tool_use_id = str(attrs.get("tool_use_id") or "").strip()
            if tool_use_id:
                by_id[tool_use_id] = attrs
        return by_id

    def _tool_span_attrs_by_id(self, raw_response: dict[str, object]) -> dict[str, dict[str, object]]:
        by_id: dict[str, dict[str, object]] = {}
        for span in extract_spans(raw_response):
            if span.get("name") != "claude_code.tool":
                continue
            attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
            tool_use_id = str(attrs.get("tool_use_id") or attrs.get("gen_ai.tool.call.id") or "").strip()
            if tool_use_id:
                by_id[tool_use_id] = attrs
        return by_id

    def _server_tool_use_from_api_bodies(self, raw_response: dict[str, object]) -> tuple[dict[str, int], bool]:
        server_tool_use: dict[str, int] = {}
        for response in extract_api_response_body_summaries(raw_response):
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            if isinstance(usage.get("server_tool_use"), dict):
                _merge_server_tool_use(server_tool_use, usage.get("server_tool_use"))
        return server_tool_use, any(count > 0 for count in server_tool_use.values())

    def _apply_tool_decision(self, payload: dict[str, object], attrs: dict[str, object]) -> None:
        decision = str(attrs.get("decision") or "").strip()
        if decision:
            payload["decision"] = decision
            if decision.lower() in _REJECTED_DECISIONS:
                payload["ok"] = False
                payload["status"] = "rejected"
        source = str(attrs.get("source") or "").strip()
        if source:
            payload["decision_source"] = source

    def extract_available_tools(self, raw_response: object) -> list[str]:
        if isinstance(raw_response, dict):
            return extract_available_tools_from_api_bodies(raw_response)
        return []

    def has_successful_tool_result_event(self, raw_response: object) -> bool:
        if not isinstance(raw_response, dict):
            return False
        return bool(self._successful_tool_result_info_by_id(raw_response))

    def infer_trajectory_data(self, raw_response: object, *, record: dict[str, object]) -> TrajectoryData | None:
        if not isinstance(raw_response, dict):
            return None
        workspace_path_value = str(record.get("workspace_path") or "").strip()
        if not workspace_path_value:
            return None
        workspace_path = Path(workspace_path_value)
        steps = []
        meta: TraceInferenceMeta = {}
        tool_result_info_by_id = self._successful_tool_result_info_by_id(raw_response)
        tool_result_by_id = {
            tool_use_id: str(info.get("text") or "")
            for tool_use_id, info in tool_result_info_by_id.items()
        }

        for call in self.extract_tool_calls(raw_response):
            if call.get("source") == "claude.otel.server_tool_use":
                continue
            tool_name = str(call.get("tool_name") or "")
            if tool_name in _NON_RETRIEVAL_TOOLS:
                continue
            payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
            tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
            tool_use_id = str(payload.get("id") or "")
            if tool_use_id not in tool_result_info_by_id:
                continue
            output_text = tool_result_by_id.get(tool_use_id, "")
            if tool_name == "Read":
                output_text = _bounded_tool_output_text(output_text, meta=meta)
                file_path = str(tool_input.get("file_path") or "").strip()
                if file_path:
                    steps.append(infer_read_step(file_path, output_text=output_text, workspace_path=workspace_path))
                continue
            if tool_name == "Bash":
                output_text = _bounded_tool_output_text(output_text, meta=meta)
                command = str(tool_input.get("command") or "").strip()
                step = infer_retrieval_step_from_command(
                    command,
                    output_text=output_text,
                    workspace_path=workspace_path,
                    meta=meta,
                )
                if step:
                    steps.append(step)
                continue
            if tool_name == "Grep":
                output_text = _bounded_tool_output_text(output_text, meta=meta)
                step = infer_search_file_step_from_path(
                    str(tool_input.get("path") or ""),
                    output_text=output_text,
                    workspace_path=workspace_path,
                )
                if step:
                    steps.append(step)
                continue
            step = infer_retrieval_step_from_tool_result(
                output_text,
                output_text=output_text,
                workspace_path=workspace_path,
                meta=meta,
            )
            if step:
                steps.append(step)

        traj = trajectory_from_steps(steps)
        if traj is not None:
            if meta:
                traj["trace_inference_meta"] = meta
            return traj

        return None

    def _successful_tool_result_info_by_id(self, raw_response: dict[str, object]) -> dict[str, dict[str, object]]:
        return {
            tool_use_id: info
            for tool_use_id, info in self._tool_result_info_by_id(raw_response).items()
            if _bool_value(info.get("is_error")) is not True
        }

    def _tool_result_info_by_id(self, raw_response: dict[str, object]) -> dict[str, dict[str, object]]:
        result_by_id: dict[str, dict[str, object]] = {}
        for result in extract_tool_results_from_api_bodies(raw_response):
            tool_use_id = str(result.get("tool_use_id") or "").strip()
            if not tool_use_id:
                continue
            text = tool_result_text_from_value(result.get("content"))
            result_by_id[tool_use_id] = {
                "text": text,
                "is_error": result.get("is_error"),
            }
        return result_by_id
