# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.agents.codex_otel_v2.adapter import CodexOtelV2Adapter
from contextbench.agents.codex_otel_v2.otel import OtlpJsonCaptureServer, flatten_otlp_logs, sanitize_otel_value
from contextbench.agents.codex_otel_v2.parser import CodexOtelV2AgentParser
from contextbench.agents.codex_otel_v2.runtime import (
    build_codex_otel_raw_response,
    prepare_runtime_env,
    run_invocation,
    runtime_root,
    write_otel_config,
)
from contextbench.agents.codex_otel_v2.prompting import build_prompt as build_codex_otel_v2_prompt
from contextbench.agents.registry import get_coding_agent_adapter, normalize_coding_agent_name
from contextbench.coding_agents import convert_run_record
from contextbench.coding_agents.runtime import missing_required_tool_call_patterns


def test_otel_capture_export_omits_raw_requests_by_default() -> None:
    server = OtlpJsonCaptureServer()
    server.start()
    try:
        with server._lock:
            server._requests.append({"path": "/v1/logs", "json": {"resourceLogs": []}})
        compact = server.export()
        debug = server.export(include_requests=True)
    finally:
        server.stop()

    assert compact == {"request_count": 1, "logs": [], "traces": [], "metrics": []}
    assert debug["requests"] == [{"path": "/v1/logs", "json": {"resourceLogs": []}}]


def test_build_raw_response_omits_stdout_text(tmp_path) -> None:
    final_message = {"status": "completed", "final_answer": "done"}
    final_path = tmp_path / "final.json"
    config_path = tmp_path / "config.toml"
    final_path.write_text(json.dumps(final_message, separators=(",", ":")), encoding="utf-8")
    config_path.write_text("[otel]\n", encoding="utf-8")

    raw_response = build_codex_otel_raw_response(
        agent_name="codex-otel-v2",
        otel_capture={"request_count": 0, "logs": [], "traces": [], "metrics": []},
        final_output_path=final_path,
        config_path=config_path,
    )

    assert raw_response["final_message"] == final_message
    assert "stdout_text" not in raw_response


def test_build_raw_response_does_not_fallback_to_stdout_when_final_message_is_missing(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[otel]\n", encoding="utf-8")

    raw_response = build_codex_otel_raw_response(
        agent_name="codex-otel-v2",
        otel_capture={"request_count": 0, "logs": [], "traces": [], "metrics": []},
        final_output_path=tmp_path / "missing.json",
        config_path=config_path,
    )

    assert "final_message" not in raw_response
    assert "stdout_text" not in raw_response


def _patch_successful_codex_invocation(monkeypatch, *, task_dir: Path, final_output_filename: str) -> None:
    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        (task_dir / final_output_filename).write_text(
            json.dumps({"status": "completed", "final_answer": "done", "notes": ""}),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex_otel_v2.runtime.run_command", fake_run_command)


def _patch_otel_capture(monkeypatch, capture: dict[str, object]) -> None:
    class FakeOtlpJsonCaptureServer:
        port = 4318

        def __init__(self, *, bind_host: str = "127.0.0.1") -> None:
            del bind_host

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def export(self) -> dict[str, object]:
            return capture

    monkeypatch.setattr("contextbench.agents.codex_otel_v2.runtime.OtlpJsonCaptureServer", FakeOtlpJsonCaptureServer)


def test_run_invocation_fails_without_otel_payloads(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    task_dir.mkdir()
    codex_home.mkdir()

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        (task_dir / "final-output.json").write_text(
            json.dumps({"status": "completed", "final_answer": "done", "notes": ""}),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex_otel_v2.runtime.run_command", fake_run_command)

    result = run_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt="Inspect files.",
        prompt_filename="prompt.txt",
        stderr_filename="stderr.log",
        raw_response_filename="raw-response.json",
        raw_output_filename="codex-stdout.txt",
        final_output_filename="final-output.json",
        timeout=30,
        model=None,
        reasoning_effort=None,
        extra_args=(),
        env={"CODEX_HOME": str(codex_home), "HOME": str(tmp_path)},
        schema_path=CodexOtelV2Adapter().output_schema_path,
    )

    assert result.command_result["ok"] is False
    assert "without receiving OTLP payloads" in str(result.diagnostic_note)
    assert result.retry == {
        "attempts": 1,
        "max_attempts": 1,
        "retried": False,
        "suppressed": False,
        "suppression_reason": None,
        "events": [],
    }


def test_scored_run_invocation_fails_without_successful_otel_tool_result(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    task_dir.mkdir()
    codex_home.mkdir()
    _patch_successful_codex_invocation(monkeypatch, task_dir=task_dir, final_output_filename="final-output.json")
    _patch_otel_capture(
        monkeypatch,
        {
            "request_count": 1,
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.sse_event",
                    "attributes": {"event.kind": "response.completed"},
                }
            ],
            "traces": [],
            "metrics": [],
        },
    )

    result = run_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt="Inspect files.",
        prompt_filename="prompt.txt",
        stderr_filename="stderr.log",
        raw_response_filename="raw-response.json",
        raw_output_filename="codex-stdout.txt",
        final_output_filename="final-output.json",
        timeout=30,
        model=None,
        reasoning_effort=None,
        extra_args=(),
        env={"CODEX_HOME": str(codex_home), "HOME": str(tmp_path)},
        schema_path=CodexOtelV2Adapter().output_schema_path,
    )

    assert result.command_result["ok"] is False
    assert "without successful OTEL tool-result telemetry" in str(result.diagnostic_note)


def test_scored_run_invocation_fails_without_otel_derived_context(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    task_dir.mkdir()
    codex_home.mkdir()
    _patch_successful_codex_invocation(monkeypatch, task_dir=task_dir, final_output_filename="final-output.json")
    _patch_otel_capture(
        monkeypatch,
        {
            "request_count": 1,
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "tool_name": "exec_command",
                        "success": True,
                        "command": "date",
                        "output": "Fri Jun 26 12:00:00 CEST 2026\n",
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    )

    result = run_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt="Inspect files.",
        prompt_filename="prompt.txt",
        stderr_filename="stderr.log",
        raw_response_filename="raw-response.json",
        raw_output_filename="codex-stdout.txt",
        final_output_filename="final-output.json",
        timeout=30,
        model=None,
        reasoning_effort=None,
        extra_args=(),
        env={"CODEX_HOME": str(codex_home), "HOME": str(tmp_path)},
        schema_path=CodexOtelV2Adapter().output_schema_path,
    )

    assert result.command_result["ok"] is False
    assert "without OTEL-derived evaluable context" in str(result.diagnostic_note)


def test_setup_run_invocation_does_not_require_otel_derived_context(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    task_dir.mkdir()
    codex_home.mkdir()
    _patch_successful_codex_invocation(monkeypatch, task_dir=task_dir, final_output_filename="setup-last-message.txt")
    _patch_otel_capture(
        monkeypatch,
        {
            "request_count": 1,
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.sse_event",
                    "attributes": {"event.kind": "response.completed"},
                }
            ],
            "traces": [],
            "metrics": [],
        },
    )

    result = run_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt="Prepare dependencies.",
        prompt_filename="setup-prompt.txt",
        stderr_filename="setup-stderr.log",
        raw_response_filename="setup-raw-response.json",
        raw_output_filename="setup-codex-stdout.txt",
        final_output_filename="setup-last-message.txt",
        timeout=30,
        model=None,
        reasoning_effort=None,
        extra_args=(),
        env={"CODEX_HOME": str(codex_home), "HOME": str(tmp_path)},
        schema_path=None,
    )

    assert result.command_result["ok"] is True
    assert result.diagnostic_note is None
