# SPDX-License-Identifier: Apache-2.0

"""Claude v2 runtime that captures OpenTelemetry output."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Callable, Sequence

from ..adapter_base import CodingAgentInvocationResult
from ..claude import runtime as claude_runtime
from .collector import OtelHttpCapture
from .otel import (
    body_ref_read_error,
    extract_api_body_refs_from_requests,
    extract_api_request_body_summaries_from_requests,
    extract_api_response_body_summaries,
    extract_api_response_body_summaries_from_requests,
    extract_log_records,
    extract_log_records_from_requests,
    extract_metrics,
    extract_metrics_from_requests,
    extract_spans,
    extract_spans_from_requests,
    extract_tool_results_from_api_bodies,
    read_capture_requests,
)
from .parser import ClaudeOtelAgentParser
from ...coding_agents.files import safe_path_component, write_json
from ...coding_agents.response_parsing import build_claude_raw_response, structured_output_schema_error
from ...coding_agents.runtime_backends import BaseTaskRuntime
from ...coding_agents.runtime_common import (
    run_command,
    write_prompt_file,
)
from ...coding_agents.trace_inference import tool_result_text_from_value
from ...coding_agents.types import ClaudeRawResponse, CommandResult
from ..otel_common import (
    OtelScoredRunPolicy,
    append_diagnostic_note,
    collector_endpoint_hosts,
    force_command_failure,
    one_attempt_retry_metadata,
)

_OTEL_ERROR_LOG_NAMES = frozenset(
    {
        "claude_code.api_error",
        "claude_code.api_refusal",
        "claude_code.api_retries_exhausted",
        "claude_code.internal_error",
    }
)
_PERSISTED_OUTPUT_RE = re.compile(
    r"<persisted-output>\s*Output too large \((?P<label>[^)]+)\)\. "
    r"Full output saved to:\s*(?P<path>[^\r\n]+)",
    re.MULTILINE,
)
_SCORED_RUN_POLICY = OtelScoredRunPolicy(
    missing_successful_tool_result_note=(
        "Claude OTEL completed without successful OTEL tool-result telemetry; "
        "refusing empty-context fallback output."
    ),
    missing_evaluable_context_note=(
        "Claude OTEL completed without OTEL-derived evaluable context; "
        "refusing empty-context fallback output."
    ),
)


def runtime_root(task_dir: Path) -> Path:
    return claude_runtime.runtime_root(task_dir)


def prepare_runtime_env(*args, **kwargs) -> dict[str, str]:
    return claude_runtime.prepare_runtime_env(*args, **kwargs)


def prepare_runtime_files(*args, **kwargs):
    return claude_runtime.prepare_runtime_files(*args, **kwargs)


def validate_auth(*args, **kwargs):
    return claude_runtime.validate_auth(*args, **kwargs)


def validate_auth_in_runtime(*args, **kwargs):
    return claude_runtime.validate_auth_in_runtime(*args, **kwargs)


def _string_fragments(value: object, *, depth: int = 0) -> list[str]:
    if depth > 6 or value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_string_fragments(item, depth=depth + 1))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for item in value.values():
            fragments.extend(_string_fragments(item, depth=depth + 1))
        return fragments
    return []


def _compact_diagnostic_text(fragments: Sequence[object], *, limit: int = 300) -> str:
    text = " ".join(fragment for value in fragments for fragment in _string_fragments(value)).strip()
    text = " ".join(text.split())
    return text[:limit].rstrip()


def _otel_error_observations(raw_response: dict[str, object]) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for log in extract_log_records(raw_response):
        name = str(log.get("name") or "")
        if name not in _OTEL_ERROR_LOG_NAMES:
            continue
        attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
        observations.append({"kind": "log", "name": name, "attributes": attrs, "body": log.get("body")})

    for span in extract_spans(raw_response):
        if span.get("name") != "claude_code.llm_request":
            continue
        attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
        success = str(attrs.get("success") or "").strip().lower()
        if success not in {"false", "0", "no"} and not attrs.get("error"):
            continue
        observations.append({"kind": "span", "name": "claude_code.llm_request", "attributes": attrs})
    return observations


def _structured_retry_error_from_otel(raw_response: dict[str, object]) -> bool:
    fragments: list[object] = []
    for log in extract_log_records(raw_response):
        if log.get("name") not in _OTEL_ERROR_LOG_NAMES:
            continue
        attrs = log.get("attributes") if isinstance(log.get("attributes"), dict) else {}
        fragments.append(attrs)
        fragments.append(log.get("body"))
    for response in extract_api_response_body_summaries(raw_response):
        fragments.append(response.get("stop_reason"))
        fragments.append(response.get("stop_details"))
    return "error_max_structured_output_retries" in _compact_diagnostic_text(fragments, limit=4000).lower()


def _classify_diagnostic_note_from_otel(
    *,
    command_result: CommandResult,
    raw_response: dict[str, object],
    structured_output: object | None,
    schema_path: Path | None,
) -> str | None:
    if _structured_retry_error_from_otel(raw_response):
        return (
            "Claude Code hit error_max_structured_output_retries: "
            "structured output could not be produced within Claude Code's retry limit."
        )

    observations = _otel_error_observations(raw_response)
    if observations:
        first = observations[-1]
        attrs = first.get("attributes") if isinstance(first.get("attributes"), dict) else {}
        detail = _compact_diagnostic_text(
            [
                attrs.get("error"),
                attrs.get("message"),
                attrs.get("status"),
                attrs.get("status_code"),
                first.get("body"),
            ],
            limit=240,
        )
        name = str(first.get("name") or "Claude OTEL error")
        return f"{name}: {detail}" if detail else name

    if schema_path is not None and command_result["ok"] and structured_output is None:
        return "Claude Code completed without producing valid structured output."
    if not command_result["ok"] and not command_result["timeout"]:
        return "Claude Code failed without OTEL error telemetry."
    return None


def enable_otel_env(env: dict[str, str] | None) -> dict[str, str]:
    result = dict(env or {})
    result.pop("OTEL_SDK_DISABLED", None)
    result.update(
        {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_TRACES_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/json",
            "OTEL_LOGS_EXPORT_INTERVAL": "1000",
            "OTEL_METRIC_EXPORT_INTERVAL": "1000",
            "OTEL_TRACES_EXPORT_INTERVAL": "1000",
            "OTEL_LOG_TOOL_DETAILS": "1",
            "OTEL_LOG_TOOL_CONTENT": "1",
            "OTEL_SERVICE_NAME": "contextbench-claude-otel",
        }
    )
    resource_attributes = result.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    contextbench_attrs = "service.name=contextbench-claude-otel,contextbench.agent=claude-otel"
    result["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{resource_attributes},{contextbench_attrs}" if resource_attributes else contextbench_attrs
    )
    return result


def _env_with_capture_endpoint(
    env: dict[str, str] | None,
    capture: OtelHttpCapture,
    *,
    api_bodies_dir: Path,
) -> dict[str, str]:
    result = enable_otel_env(env)
    result["OTEL_EXPORTER_OTLP_ENDPOINT"] = capture.endpoint
    result["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = capture.logs_endpoint
    result["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] = capture.metrics_endpoint
    result["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = capture.traces_endpoint
    result["OTEL_LOG_RAW_API_BODIES"] = f"file:{api_bodies_dir.resolve()}"
    return result


def build_otel_raw_response(
    *,
    otel_capture_path: Path,
    command_result: CommandResult,
    api_bodies_dir: Path,
) -> dict[str, object]:
    captured_requests = read_capture_requests(otel_capture_path)
    captured_raw_response: dict[str, object] = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "capture_path": str(otel_capture_path),
            "requests": captured_requests,
        },
    }
    raw_response: dict[str, object] = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "capture_path": str(otel_capture_path),
            "artifact_retention": "compact",
            "capture_retained": False,
            "api_body_artifacts_retained": False,
        },
        "command_result": dict(command_result),
    }
    raw_response["otel"]["logs"] = extract_log_records_from_requests(captured_raw_response)  # type: ignore[index]
    raw_response["otel"]["spans"] = extract_spans_from_requests(captured_raw_response)  # type: ignore[index]
    raw_response["otel"]["metrics"] = extract_metrics_from_requests(captured_raw_response)  # type: ignore[index]
    raw_response["otel"]["api_body_refs"] = extract_api_body_refs_from_requests(  # type: ignore[index]
        captured_raw_response
    )
    raw_response["otel"]["api_body_ref_errors"] = _api_body_ref_errors(  # type: ignore[index]
        raw_response["otel"]["api_body_refs"],  # type: ignore[index]
        body_ref_root=api_bodies_dir,
    )
    raw_response["otel"]["api_request_bodies"] = extract_api_request_body_summaries_from_requests(  # type: ignore[index]
        captured_raw_response,
        body_ref_root=api_bodies_dir,
    )
    raw_response["otel"]["api_response_bodies"] = extract_api_response_body_summaries_from_requests(  # type: ignore[index]
        captured_raw_response,
        body_ref_root=api_bodies_dir,
    )
    raw_response["otel"]["summary"] = {
        "request_count": len(captured_requests),
        "log_count": len(raw_response["otel"]["logs"]),  # type: ignore[index]
        "span_count": len(raw_response["otel"]["spans"]),  # type: ignore[index]
        "metric_count": len(raw_response["otel"]["metrics"]),  # type: ignore[index]
        "api_body_ref_count": len(raw_response["otel"]["api_body_refs"]),  # type: ignore[index]
        "api_body_ref_error_count": len(raw_response["otel"]["api_body_ref_errors"]),  # type: ignore[index]
        "api_request_body_count": len(raw_response["otel"]["api_request_bodies"]),  # type: ignore[index]
        "api_response_body_count": len(raw_response["otel"]["api_response_bodies"]),  # type: ignore[index]
    }
    return raw_response


def _api_body_ref_errors(api_body_refs: object, *, body_ref_root: Path) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    if not isinstance(api_body_refs, list):
        return errors
    for ref in api_body_refs:
        if not isinstance(ref, dict):
            continue
        event_name = str(ref.get("name") or "").strip()
        body_ref = str(ref.get("body_ref") or "").strip()
        if not body_ref:
            if event_name in {"claude_code.api_request_body", "claude_code.api_response_body"}:
                errors.append({"body_ref": None, "event": event_name, "error": "missing_body_ref"})
            continue
        error = body_ref_read_error(body_ref, body_ref_root=body_ref_root)
        if error:
            errors.append({"body_ref": body_ref, "error": error})
    return errors


def _api_body_ref_diagnostic(api_body_ref_errors: object) -> str | None:
    if not isinstance(api_body_ref_errors, list) or not api_body_ref_errors:
        return None
    first = api_body_ref_errors[0] if isinstance(api_body_ref_errors[0], dict) else {}
    detail = _compact_diagnostic_text(
        [
            first.get("body_ref"),
            first.get("error"),
        ],
        limit=240,
    )
    count = len(api_body_ref_errors)
    suffix = f" First error: {detail}" if detail else ""
    return f"Claude OTEL API body capture had {count} unreadable body_ref artifact(s).{suffix}"


def _otel_capture_diagnostic(otel: object) -> str | None:
    if not isinstance(otel, dict):
        return "Claude OTEL capture is missing from the raw response."
    summary = otel.get("summary") if isinstance(otel.get("summary"), dict) else {}
    if _int_summary_value(summary.get("request_count")) <= 0:
        return "Claude OTEL capture did not receive any OTLP requests."
    if _int_summary_value(summary.get("api_request_body_count")) <= 0:
        return "Claude OTEL capture did not include any API request body artifacts."
    if _int_summary_value(summary.get("api_response_body_count")) <= 0:
        return "Claude OTEL capture did not include any API response body artifacts."
    return None


def _int_summary_value(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _archive_persisted_tool_results_from_otel(
    raw_response: dict[str, object],
    *,
    task_dir: Path,
) -> list[dict[str, object]]:
    texts: list[str] = []
    for result in extract_tool_results_from_api_bodies(raw_response):
        text = tool_result_text_from_value(result.get("content")).strip()
        if text:
            texts.append(text)
    if not texts:
        return []
    matches = list(_PERSISTED_OUTPUT_RE.finditer("\n".join(texts)))
    if not matches:
        return []
    output_dir = task_dir / "claude-persisted-tool-results"
    allowed_roots = (task_dir, runtime_root(task_dir))
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches, start=1):
        raw_path = match.group("path").strip()
        if raw_path in seen:
            continue
        seen.add(raw_path)
        source = Path(raw_path)
        label = match.group("label").strip() or None
        entry: dict[str, object] = {
            "source_path": raw_path,
            "artifact_path": None,
            "status": "missing",
            "size_bytes": None,
            "label": label,
        }
        if source.is_file():
            try:
                source_resolved = source.resolve()
                allowed = any(source_resolved.is_relative_to(root.resolve()) for root in allowed_roots)
            except RuntimeError:
                allowed = False
            if not allowed:
                entry["status"] = "rejected_unsafe_source"
                results.append(entry)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = source.suffix if source.suffix else ".txt"
            name = safe_path_component(source.stem or "tool-result")
            target = output_dir / f"{index:03d}-{name}{suffix}"
            shutil.copy2(source, target)
            entry.update(
                {
                    "artifact_path": str(target),
                    "status": "archived",
                    "size_bytes": source.stat().st_size,
                }
            )
        results.append(entry)

    if results:
        write_json(task_dir / "claude-persisted-tool-results.json", results)
    return results


def _api_bodies_dir_for(task_dir: Path, raw_output_filename: str) -> Path:
    name = raw_output_filename.replace("claude-output.jsonl", "claude-otel-api-bodies")
    if name == raw_output_filename:
        name = raw_output_filename.replace(".jsonl", "-api-bodies").replace(".json", "-api-bodies")
    return task_dir / name


def run_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    raw_output_filename: str,
    timeout: int,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    env: dict[str, str] | None,
    schema_path: Path | None,
    settings_path: Path,
    mcp_config_path: Path,
    validate_runtime_isolation: bool,
    execution_backend: BaseTaskRuntime | None = None,
    retry_dirty_check: Callable[[], bool] | None = None,
) -> CodingAgentInvocationResult:
    del retry_dirty_check
    bind_host, endpoint_host = collector_endpoint_hosts(execution_backend)
    prompt_path = write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    otel_capture_path = task_dir / raw_output_filename.replace("claude-output.jsonl", "claude-otel-requests.jsonl")
    raw_response_path = task_dir / raw_response_filename
    stream_output_path = task_dir / raw_output_filename
    api_bodies_dir = _api_bodies_dir_for(task_dir, raw_output_filename)
    permission_mode = "acceptEdits"
    if execution_backend is not None and getattr(execution_backend.config, "backend", None) == "docker":
        permission_mode = "bypassPermissions"
    command, _ = claude_runtime.build_command(
        schema_path=schema_path,
        prompt=prompt,
        model=model,
        reasoning_effort=claude_runtime.normalize_reasoning_effort(reasoning_effort),
        extra_args=extra_args,
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
        permission_mode=permission_mode,
    )

    started_at = time.time()
    completed_at = started_at
    stream_raw_response: ClaudeRawResponse = {"agent": "claude", "response_format": "stream-json", "response": None}
    command_result: CommandResult = {"ok": False, "exit_code": None, "signal": None, "timeout": False}
    retry_events: list[dict[str, object]] = []

    for path in (stream_output_path, stderr_path, raw_response_path, otel_capture_path):
        path.unlink(missing_ok=True)
    shutil.rmtree(api_bodies_dir, ignore_errors=True)
    api_bodies_dir.mkdir(parents=True, exist_ok=True)

    try:
        with OtelHttpCapture(otel_capture_path, bind_host=bind_host, endpoint_host=endpoint_host) as capture:
            invocation_env = _env_with_capture_endpoint(env, capture, api_bodies_dir=api_bodies_dir)
            if execution_backend is None:
                command_result = run_command(
                    command,
                    cwd=workspace_path,
                    stdin_text=prompt,
                    stdout_path=stream_output_path,
                    stderr_path=stderr_path,
                    timeout=timeout,
                    env=invocation_env,
                )
            else:
                command_result = execution_backend.run_command(
                    command,
                    cwd=workspace_path,
                    stdin_text=prompt,
                    stdout_path=stream_output_path,
                    stderr_path=stderr_path,
                    timeout=timeout,
                    env=invocation_env,
                    host_runner=run_command,
                )

        completed_at = time.time()

        stream_raw_response = build_claude_raw_response(stream_output_path)
        isolation_diagnostic_note: str | None = None
        if command_result["ok"] and validate_runtime_isolation:
            try:
                claude_runtime.validate_isolation(
                    stream_raw_response,
                    allowed_mcp_servers=claude_runtime.configured_mcp_server_names(mcp_config_path),
                )
            except Exception as exc:
                command_result = force_command_failure(command_result)
                isolation_diagnostic_note = str(exc) or "Claude runtime isolation validation failed."

        raw_response = build_otel_raw_response(
            otel_capture_path=otel_capture_path,
            command_result=command_result,
            api_bodies_dir=api_bodies_dir,
        )
        otel = raw_response.get("otel") if isinstance(raw_response.get("otel"), dict) else {}
        api_body_ref_diagnostic_note = _api_body_ref_diagnostic(
            otel.get("api_body_ref_errors") if isinstance(otel, dict) else None
        )
        otel_capture_diagnostic_note = _otel_capture_diagnostic(otel)
        if api_body_ref_diagnostic_note or otel_capture_diagnostic_note:
            command_result = force_command_failure(command_result)
            raw_response["command_result"] = dict(command_result)
        write_json(raw_response_path, raw_response)
    finally:
        stream_output_path.unlink(missing_ok=True)
        otel_capture_path.unlink(missing_ok=True)
        shutil.rmtree(api_bodies_dir, ignore_errors=True)

    parser = ClaudeOtelAgentParser()
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    schema_validation_note = (
        structured_output_schema_error(structured_output, schema_path)
        if structured_output is not None and schema_path is not None
        else None
    )
    if schema_validation_note:
        structured_output = None
    diagnostic_note = _classify_diagnostic_note_from_otel(
        command_result=command_result,
        raw_response=raw_response,
        structured_output=structured_output,
        schema_path=schema_path,
    )
    if structured_output is not None and diagnostic_note == "Claude Code completed without producing valid structured output.":
        diagnostic_note = None
    if schema_validation_note:
        diagnostic_note = append_diagnostic_note(diagnostic_note, schema_validation_note)
    if isolation_diagnostic_note:
        diagnostic_note = append_diagnostic_note(diagnostic_note, isolation_diagnostic_note)
    if api_body_ref_diagnostic_note:
        diagnostic_note = append_diagnostic_note(diagnostic_note, api_body_ref_diagnostic_note)
    if otel_capture_diagnostic_note:
        diagnostic_note = append_diagnostic_note(diagnostic_note, otel_capture_diagnostic_note)
    validation = _SCORED_RUN_POLICY.validate(
        parser=parser,
        raw_response=raw_response,
        command_result=command_result,
        diagnostic_note=diagnostic_note,
        workspace_path=workspace_path,
        scored=schema_path is not None,
    )
    command_result = validation.command_result
    diagnostic_note = validation.diagnostic_note
    if validation.failed:
        raw_response["command_result"] = dict(command_result)
        write_json(raw_response_path, raw_response)
    return CodingAgentInvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        command_executions=parser.extract_command_executions(raw_response),
        available_tools=parser.extract_available_tools(raw_response),
        persisted_tool_results=_archive_persisted_tool_results_from_otel(raw_response, task_dir=task_dir),
        diagnostic_note=diagnostic_note,
        retry=one_attempt_retry_metadata(events=retry_events),
        started_at=started_at,
        completed_at=completed_at,
    )
