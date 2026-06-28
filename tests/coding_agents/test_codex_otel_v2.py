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


def test_codex_otel_v2_adapter_is_registered() -> None:
    assert normalize_coding_agent_name("codex-otel-v2") == "codex-otel-v2"
    assert normalize_coding_agent_name("codex_v2") == "codex-otel-v2"
    adapter = get_coding_agent_adapter("codex-otel")
    assert isinstance(adapter, CodexOtelV2Adapter)
    assert adapter.output_schema_path.name == "output.schema.json"
    assert adapter.output_schema_path.parent.name == "codex_otel_v2"
    assert adapter.scored_context_source == "otel_tool_results"
    assert adapter.score_inferred_context is True


def test_prepare_runtime_env_enables_codex_home_without_disabling_otel(tmp_path, monkeypatch) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    env = prepare_runtime_env(tmp_path / "task", source_codex_dir=source_codex_dir, include_host_env=False)

    codex_home = Path(env["CODEX_HOME"])
    assert codex_home == Path(env["HOME"]) / ".codex"
    assert (codex_home / "auth.json").exists()
    assert str(runtime_root(tmp_path / "task") / "bin") in env["PATH"].split(":")
    assert "OTEL_SDK_DISABLED" not in env


def test_write_otel_config_targets_local_otlp_json_endpoints(tmp_path) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()

    config_path = write_otel_config(
        codex_home=codex_home,
        logs_endpoint="http://127.0.0.1:4318/v1/logs",
        traces_endpoint="http://127.0.0.1:4318/v1/traces",
    )

    text = config_path.read_text(encoding="utf-8")
    assert "exporter = { otlp-http = {" in text
    assert "trace_exporter = { otlp-http = {" in text
    assert 'metrics_exporter = "none"' in text
    assert 'endpoint = "http://127.0.0.1:4318/v1/logs"' in text
    assert 'endpoint = "http://127.0.0.1:4318/v1/traces"' in text
    assert 'protocol = "json"' in text


def test_flatten_otlp_logs_normalizes_log_records() -> None:
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "codex"}},
                    ]
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": "1",
                                "body": {"stringValue": "codex.sse_event"},
                                "attributes": [
                                    {"key": "event.kind", "value": {"stringValue": "response.completed"}},
                                    {"key": "input_token_count", "value": {"intValue": "10"}},
                                    {"key": "user.email", "value": {"stringValue": "person@example.com"}},
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }

    logs = flatten_otlp_logs(payload)

    assert logs == [
        {
            "signal": "log",
            "name": "codex.sse_event",
            "body": "codex.sse_event",
            "attributes": {
                "event.kind": "response.completed",
                "input_token_count": 10,
                "user.email": "[redacted]",
            },
            "resource": {"service.name": "codex"},
            "scope": {},
            "time_unix_nano": "1",
            "observed_time_unix_nano": None,
            "severity_text": None,
        }
    ]


def test_sanitize_otel_value_redacts_raw_otlp_key_value_attributes() -> None:
    payload = {
        "key": "user.account_id",
        "value": {"stringValue": "53625ed5-8688-4080-90d9-3cdb2db32d6c"},
    }

    assert sanitize_otel_value(payload) == {"key": "user.account_id", "value": "[redacted]"}


def test_parser_extracts_structured_output_usage_and_tool_calls_from_otel(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "final_message": {
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": ["app.py"],
            "retrieved_context_spans": [{"file": "app.py", "start": 1, "end": 3}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.sse_event",
                    "attributes": {
                        "event.kind": "response.completed",
                        "input_token_count": 10,
                        "output_token_count": 5,
                        "cached_token_count": 4,
                        "reasoning_token_count": 2,
                    },
                },
                {
                    "signal": "log",
                    "name": "codex.tool_decision",
                    "attributes": {
                        "tool_name": "shell",
                        "call_id": "call_1",
                        "decision": "approved",
                    },
                },
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "tool.name": "shell",
                        "call_id": "call_1",
                        "arguments": "{\"cmd\":\"nl -ba app.py | sed -n '1,240p'\"}",
                        "output": "1  def f():\n2      return 1\n3  ",
                        "success": True,
                    },
                },
            ],
            "traces": [],
            "metrics": [],
        },
    }

    parser = CodexOtelV2AgentParser()

    structured_output = parser.extract_structured_output(raw_response)
    assert structured_output is not None
    assert structured_output["final_answer"] == "done"
    assert "retrieved_context_files" not in structured_output
    assert "retrieved_context_spans" not in structured_output
    assert "retrieved_context_symbols" not in structured_output
    assert parser.extract_token_usage(raw_response) == {
        "source": "codex-otel-v2.sse_event",
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_input_tokens": 4,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
        "reasoning_tokens": 2,
    }
    assert parser.extract_tool_calls(raw_response) == [
        {
            "source": "codex-otel-v2.log",
            "tool_name": "shell",
            "payload": {
                "tool.name": "shell",
                "call_id": "call_1",
                "arguments": "{\"cmd\":\"nl -ba app.py | sed -n '1,240p'\"}",
                "arguments_json": {"cmd": "nl -ba app.py | sed -n '1,240p'"},
                "command": "nl -ba app.py | sed -n '1,240p'",
                "output": "1  def f():\n2      return 1\n3  ",
                "success": True,
                "otel_event_name": "codex.tool_result",
                "otel_signal": "log",
                "ok": True,
            },
        }
    ]
    trajectory = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})
    assert trajectory is not None
    assert trajectory["pred_files"] == ["app.py"]
    assert trajectory["pred_spans"] == {"app.py": [{"start": 1, "end": 2}]}


def test_convert_run_record_ignores_v2_self_reported_files_and_spans(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "tool_name": "exec_command",
                        "arguments": "{\"cmd\":\"nl -ba src/app.py | sed -n '1,3p'\"}",
                        "output": "     1\tdef f():\n     2\t    return 1\n",
                        "success": True,
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }
    record = {
        "agent": "codex-otel-v2",
        "instance_id": "task-otel",
        "workspace_path": str(workspace),
        "raw_response": raw_response,
        "final_output": {
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": ["src/app.py", "legacy.py"],
            "retrieved_context_spans": [
                {"file": "src/app.py", "start": 1, "end": 2},
                {"file": "src/app.py", "start": 99, "end": 100},
                {"file": "legacy.py", "start": 5, "end": 8},
            ],
            "retrieved_context_symbols": [
                {"file": "src/app.py", "name": "f"},
                {"file": "legacy.py", "name": "legacy"},
            ],
            "notes": "",
        },
    }

    converted = convert_run_record(record)
    traj = converted["traj_data"]

    assert traj["pred_files"] == ["src/app.py"]
    assert traj["pred_spans"] == {"src/app.py": [{"start": 1, "end": 2}]}
    assert traj["pred_symbols"] == {}
    assert traj["pred_files_provenance"] == {
        "src/app.py": "otel_tool_results",
    }
    assert traj["pred_spans_provenance"] == {
        "src/app.py": [{"start": 1, "end": 2, "source": "otel_tool_results"}],
    }
    assert traj["pred_symbols_provenance"] == {}
    assert traj["pred_files_source"] == ["otel_tool_results"]
    assert traj["pred_spans_source"] == ["otel_tool_results"]
    assert traj["pred_symbols_source"] == []
    assert traj["context_source"] == "otel_tool_results"
    assert traj["context_fallback_used"] is False
    assert traj["structured_output_context_available"] is False
    assert traj["otel_context_available"] is True


def test_codex_otel_v2_prompt_removes_redundant_context_self_reporting() -> None:
    prompt = build_codex_otel_v2_prompt({"repo": "demo/repo", "prompt": "Fix the bug."})

    assert "Use retrieved_context_files and retrieved_context_spans for that final relied-on context." not in prompt
    assert "tool-result telemetry records the files and line spans used for evaluation" in prompt
    assert "Set retrieved_context_files and retrieved_context_spans to empty arrays" not in prompt
    assert "Do not include retrieved_context_files, retrieved_context_spans, or retrieved_context_symbols." in prompt


def test_codex_otel_v2_schema_disallows_context_self_reporting() -> None:
    schema = json.loads(CodexOtelV2Adapter().output_schema_path.read_text(encoding="utf-8"))

    assert "retrieved_context_files" not in schema["properties"]
    assert "retrieved_context_spans" not in schema["properties"]
    assert "retrieved_context_symbols" not in schema["properties"]
    assert schema["required"] == ["status", "final_answer", "notes"]


def test_parser_canonicalizes_mcp_tool_calls_from_otel_attributes() -> None:
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "mcp_server": "cortex",
                        "mcp_tool": "context_search",
                        "success": True,
                        "result": {"matches": [{"path": "src/a.py", "line": 12}]},
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }

    tool_calls = CodexOtelV2AgentParser().extract_tool_calls(raw_response)

    assert tool_calls[0]["tool_name"] == "mcp__cortex__context_search"
    assert tool_calls[0]["payload"]["mcp_server"] == "cortex"
    assert tool_calls[0]["payload"]["mcp_tool"] == "context_search"
    assert missing_required_tool_call_patterns(tool_calls, [r"^mcp__cortex__"]) == []


def test_parser_refines_sed_span_from_otel_output_line_count(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "tool_name": "exec_command",
                        "arguments": "{\"cmd\":\"sed -n '1,220p' README.md\"}",
                        "output": (
                            "Chunk ID: test\n"
                            "Output:\n"
                            "# Demo\n"
                            "\n"
                            "Short file.\n"
                        ),
                        "success": True,
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }

    trajectory = CodexOtelV2AgentParser().infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})

    assert trajectory is not None
    assert trajectory["pred_spans"] == {"README.md": [{"start": 1, "end": 3}]}


def test_parser_does_not_double_prefix_qualified_mcp_tool_names() -> None:
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "mcp_tool": "mcp__cortex__context_search",
                        "success": True,
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }

    tool_call = CodexOtelV2AgentParser().extract_tool_calls(raw_response)[0]

    assert tool_call["tool_name"] == "mcp__cortex__context_search"
    assert tool_call["payload"]["mcp_server"] == "cortex"
    assert tool_call["payload"]["mcp_tool"] == "context_search"


def test_otel_raw_response_fixture_is_json_serializable(tmp_path) -> None:
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {"request_count": 0, "logs": [], "traces": [], "metrics": []},
    }
    path = tmp_path / "raw-response.json"
    path.write_text(json.dumps(raw_response), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["agent"] == "codex-otel-v2"


def test_parser_does_not_scan_raw_otel_response_for_structured_output() -> None:
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "stdout_text": json.dumps(
            {
                "status": "completed",
                "final_answer": "from stdout",
                "retrieved_context_files": ["legacy.py"],
                "retrieved_context_spans": [],
                "retrieved_context_symbols": [],
                "notes": "",
            }
        ),
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "output": json.dumps(
                            {
                                "status": "completed",
                                "final_answer": "from telemetry",
                                "retrieved_context_files": ["telemetry.py"],
                                "retrieved_context_spans": [],
                                "retrieved_context_symbols": [],
                                "notes": "",
                            }
                        )
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }

    assert CodexOtelV2AgentParser().extract_structured_output(raw_response) is None


def test_parser_does_not_parse_text_final_message_as_structured_output() -> None:
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "final_message": json.dumps(
            {
                "status": "completed",
                "final_answer": "from text",
                "notes": "",
            }
        ),
        "otel": {"logs": [], "traces": [], "metrics": []},
    }

    assert CodexOtelV2AgentParser().extract_structured_output(raw_response) is None


def test_parser_ignores_tool_results_without_explicit_success(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex-otel-v2",
        "response_format": "otlp-json",
        "otel": {
            "logs": [
                {
                    "signal": "log",
                    "name": "codex.tool_result",
                    "attributes": {
                        "tool_name": "exec_command",
                        "arguments": "{\"cmd\":\"sed -n '1,10p' README.md\"}",
                        "output": "README content\n",
                    },
                }
            ],
            "traces": [],
            "metrics": [],
        },
    }

    assert CodexOtelV2AgentParser().infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)}) is None


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
