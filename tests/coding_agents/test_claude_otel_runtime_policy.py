# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from contextbench.coding_agents.constants import CLAUDE_OTEL_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from contextbench.coding_agents.runtime import run_coding_agent_task
from contextbench.agents.claude_otel import runtime as claude_otel_runtime
from contextbench.agents.otel_common import force_command_failure, one_attempt_retry_metadata

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def _patch_minimal_claude_otel_invocation(monkeypatch, raw_response: dict[str, object]) -> None:
    class FakeOtelHttpCapture:
        endpoint = "http://127.0.0.1:4318"
        logs_endpoint = "http://127.0.0.1:4318/v1/logs"
        metrics_endpoint = "http://127.0.0.1:4318/v1/metrics"
        traces_endpoint = "http://127.0.0.1:4318/v1/traces"

        def __init__(self, output_path, *, bind_host="127.0.0.1", endpoint_host=None):
            del output_path, bind_host, endpoint_host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.OtelHttpCapture", FakeOtelHttpCapture)
    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.run_command", fake_run_command)
    monkeypatch.setattr(
        "contextbench.agents.claude_otel.runtime.claude_runtime.build_command",
        lambda **kwargs: (["claude", "--fake"], "claude-output.jsonl"),
    )
    monkeypatch.setattr(
        "contextbench.agents.claude_otel.runtime.build_otel_raw_response",
        lambda *, otel_capture_path, command_result, api_bodies_dir: {
            **raw_response,
            "command_result": dict(command_result),
        },
    )


def _run_minimal_claude_otel_invocation(tmp_path, *, schema_path, final_raw_response):
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    settings_path = tmp_path / "settings.json"
    mcp_config_path = tmp_path / "mcp.json"
    workspace.mkdir()
    task_dir.mkdir()
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text("{}", encoding="utf-8")
    return claude_otel_runtime.run_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt="Inspect files.",
        prompt_filename="prompt.txt",
        stderr_filename="stderr.log",
        raw_response_filename="raw-response.json",
        raw_output_filename="claude-output.jsonl",
        timeout=30,
        model=None,
        reasoning_effort=None,
        extra_args=(),
        env={},
        schema_path=schema_path,
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
        validate_runtime_isolation=False,
    )


def test_claude_otel_run_cleans_raw_artifacts_when_compaction_fails(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    settings_path = tmp_path / "settings.json"
    mcp_config_path = tmp_path / "mcp.json"
    workspace.mkdir()
    task_dir.mkdir()
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text("{}", encoding="utf-8")
    api_bodies_dir = task_dir / "claude-otel-api-bodies"
    capture_path = task_dir / "claude-otel-requests.jsonl"
    stream_output_path = task_dir / "claude-output.jsonl"

    class FakeOtelHttpCapture:
        endpoint = "http://127.0.0.1:4318"
        logs_endpoint = "http://127.0.0.1:4318/v1/logs"
        metrics_endpoint = "http://127.0.0.1:4318/v1/metrics"
        traces_endpoint = "http://127.0.0.1:4318/v1/traces"

        def __init__(self, output_path, *, bind_host="127.0.0.1", endpoint_host=None):
            del output_path, bind_host, endpoint_host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout
        stdout_path.write_text('{"type":"result"}\n', encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        capture_path.write_text('{"body":{}}\n', encoding="utf-8")
        raw_body_dir = Path(str((env or {}).get("OTEL_LOG_RAW_API_BODIES") or "").removeprefix("file:"))
        raw_body_dir.mkdir(parents=True, exist_ok=True)
        (raw_body_dir / "request.json").write_text('{"secret":"raw"}', encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    def raise_compaction_error(*, otel_capture_path, command_result, api_bodies_dir):
        del otel_capture_path, command_result, api_bodies_dir
        raise RuntimeError("compaction failed")

    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.OtelHttpCapture", FakeOtelHttpCapture)
    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.run_command", fake_run_command)
    monkeypatch.setattr(
        "contextbench.agents.claude_otel.runtime.claude_runtime.build_command",
        lambda **kwargs: (["claude", "--fake"], "claude-output.jsonl"),
    )
    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.build_otel_raw_response", raise_compaction_error)

    with pytest.raises(RuntimeError, match="compaction failed"):
        claude_otel_runtime.run_invocation(
            task_dir=task_dir,
            workspace_path=workspace,
            prompt="Inspect files.",
            prompt_filename="prompt.txt",
            stderr_filename="stderr.log",
            raw_response_filename="raw-response.json",
            raw_output_filename="claude-output.jsonl",
            timeout=30,
            model=None,
            reasoning_effort=None,
            extra_args=(),
            env={},
            schema_path=CLAUDE_OTEL_OUTPUT_SCHEMA_PATH,
            settings_path=settings_path,
            mcp_config_path=mcp_config_path,
            validate_runtime_isolation=False,
        )

    assert not stream_output_path.exists()
    assert not capture_path.exists()
    assert not api_bodies_dir.exists()


def _claude_otel_raw_response(*, api_request_bodies, api_response_bodies) -> dict[str, object]:
    return {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "summary": {
                "request_count": 1,
                "api_request_body_count": 1,
                "api_response_body_count": 1,
            },
            "logs": [],
            "spans": [],
            "metrics": [],
            "api_body_refs": [],
            "api_body_ref_errors": [],
            "api_request_bodies": api_request_bodies,
            "api_response_bodies": api_response_bodies,
        },
    }


def test_scored_claude_otel_run_fails_without_successful_tool_result(tmp_path, monkeypatch) -> None:
    raw_response = _claude_otel_raw_response(
        api_request_bodies=[{"source": "claude.otel.api_request_body", "tool_results": []}],
        api_response_bodies=[
            {
                "source": "claude.otel.api_response_body",
                "tool_uses": [
                    {"id": "toolu_read", "name": "Read", "input": {"file_path": "app.py"}},
                    {
                        "id": "toolu_structured",
                        "name": "StructuredOutput",
                        "input": {"status": "completed", "final_answer": "done", "notes": ""},
                    },
                ],
            }
        ],
    )
    _patch_minimal_claude_otel_invocation(monkeypatch, raw_response)

    result = _run_minimal_claude_otel_invocation(
        tmp_path,
        schema_path=CLAUDE_OTEL_OUTPUT_SCHEMA_PATH,
        final_raw_response=raw_response,
    )

    assert result.command_result["ok"] is False
    assert "without successful OTEL tool-result telemetry" in str(result.diagnostic_note)


def test_scored_claude_otel_run_fails_without_otel_derived_context(tmp_path, monkeypatch) -> None:
    raw_response = _claude_otel_raw_response(
        api_request_bodies=[
            {
                "source": "claude.otel.api_request_body",
                "tool_results": [
                    {"tool_use_id": "toolu_bash", "content": "Fri Jun 26 12:00:00 CEST 2026\n"}
                ],
            }
        ],
        api_response_bodies=[
            {
                "source": "claude.otel.api_response_body",
                "tool_uses": [
                    {"id": "toolu_bash", "name": "Bash", "input": {"command": "date"}},
                    {
                        "id": "toolu_structured",
                        "name": "StructuredOutput",
                        "input": {"status": "completed", "final_answer": "done", "notes": ""},
                    },
                ],
            }
        ],
    )
    _patch_minimal_claude_otel_invocation(monkeypatch, raw_response)

    result = _run_minimal_claude_otel_invocation(
        tmp_path,
        schema_path=CLAUDE_OTEL_OUTPUT_SCHEMA_PATH,
        final_raw_response=raw_response,
    )

    assert result.command_result["ok"] is False
    assert "without OTEL-derived evaluable context" in str(result.diagnostic_note)


def test_setup_claude_otel_run_does_not_require_otel_derived_context(tmp_path, monkeypatch) -> None:
    raw_response = _claude_otel_raw_response(
        api_request_bodies=[{"source": "claude.otel.api_request_body", "tool_results": []}],
        api_response_bodies=[{"source": "claude.otel.api_response_body", "tool_uses": []}],
    )
    _patch_minimal_claude_otel_invocation(monkeypatch, raw_response)

    result = _run_minimal_claude_otel_invocation(
        tmp_path,
        schema_path=None,
        final_raw_response=raw_response,
    )

    assert result.command_result["ok"] is True
    assert "empty-context fallback" not in str(result.diagnostic_note)
