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
