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


def test_force_command_failure_preserves_timeout_status() -> None:
    forced = force_command_failure({"ok": True, "exit_code": None, "signal": None, "timeout": True})

    assert forced == {"ok": False, "exit_code": None, "signal": None, "timeout": True}


def test_run_coding_agent_task_claude_otel_captures_local_otlp(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    source_claude_dir = tmp_path / "source-claude"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"abc"}', encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(source_claude_dir))

    task = {
        "bench": "Verified",
        "instance_id": "otel-task",
        "original_inst_id": "otel-task",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_workspace_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.run_invocation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("v2 must not call v1 run_invocation")),
    )

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout
        captured["env"] = dict(env or {})
        logs_endpoint = captured["env"]["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"]
        traces_endpoint = captured["env"]["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        api_bodies_dir = Path(str(captured["env"]["OTEL_LOG_RAW_API_BODIES"]).removeprefix("file:"))
        api_bodies_dir.mkdir(parents=True, exist_ok=True)
        persisted_path = stdout_path.parent / "tool-results" / "toolu_1.txt"
        persisted_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_path.write_text("full persisted output\n", encoding="utf-8")
        structured_output = {
            "status": "completed",
            "final_answer": "Updated the implementation and verified the change.",
            "notes": "",
        }
        api_request_body_path = api_bodies_dir / "request.json"
        api_request_body_path.write_text(
            json.dumps(
                {
                    "model": "claude-test",
                    "tools": [{"name": "Read"}, {"name": "StructuredOutput"}],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": (
                                        "<persisted-output>\n"
                                        f"Output too large (20KB). Full output saved to: {persisted_path}\n\n"
                                        "Preview (first 2KB):\npreview\n</persisted-output>"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        api_response_body_path = api_bodies_dir / "response.json"
        api_response_body_path.write_text(
            json.dumps(
                {
                    "id": "msg_1",
                    "type": "message",
                    "model": "claude-test",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "app.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_structured",
                            "name": "StructuredOutput",
                            "input": structured_output,
                        },
                    ],
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 3,
                        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
                    },
                    "stop_reason": "tool_use",
                }
            ),
            encoding="utf-8",
        )
        urllib.request.urlopen(  # noqa: S310 - test posts to local collector selected by the adapter
            urllib.request.Request(
                str(logs_endpoint),
                data=json.dumps(
                    logs_payload(
                        log_record(
                            "claude_code.api_request",
                            [
                                otel_attr("input_tokens", 12),
                                otel_attr("output_tokens", 5),
                                otel_attr("cost_usd", 0.02),
                                otel_attr("model", "claude-test"),
                            ],
                        ),
                        log_record(
                            "claude_code.tool_decision",
                            [
                                otel_attr("tool_name", "Read"),
                                otel_attr("tool_use_id", "toolu_1"),
                                otel_attr("decision", "accept"),
                                otel_attr("source", "config"),
                            ],
                        ),
                        log_record(
                            "claude_code.api_request_body",
                            [
                                otel_attr("request_id", "req_1"),
                                otel_attr("body_ref", str(api_request_body_path)),
                                otel_attr("body_length", api_request_body_path.stat().st_size),
                            ],
                        ),
                        log_record(
                            "claude_code.tool_result",
                            [
                                otel_attr("tool_name", "Read"),
                                otel_attr("tool_use_id", "toolu_1"),
                                otel_attr("success", "true"),
                                otel_attr("duration_ms", 9),
                                otel_attr("tool_input", json.dumps({"file_path": "app.py"})),
                            ],
                        ),
                        log_record(
                            "claude_code.tool_result",
                            [
                                otel_attr("tool_name", "StructuredOutput"),
                                otel_attr("tool_use_id", "toolu_structured"),
                                otel_attr("success", "true"),
                                otel_attr(
                                    "tool_input",
                                    json.dumps(
                                        structured_output
                                    ),
                                ),
                            ],
                        ),
                        log_record(
                            "claude_code.api_response_body",
                            [
                                otel_attr("request_id", "req_1"),
                                otel_attr("body_ref", str(api_response_body_path)),
                                otel_attr("body_length", api_response_body_path.stat().st_size),
                            ],
                        ),
                    )
                ).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            ),
            timeout=2,
        ).read()
        urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(
                str(traces_endpoint),
                data=json.dumps(
                    traces_payload(
                        span(
                            "claude_code.tool",
                            [
                                otel_attr("tool_name", "Read"),
                                otel_attr("tool_use_id", "toolu_1"),
                                otel_attr("file_path", "app.py"),
                            ],
                        )
                    )
                ).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            ),
            timeout=2,
        ).read()
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read", "StructuredOutput"],
                            "plugins": [],
                            "mcp_servers": [],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": "",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude_otel.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="claude-otel",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OTEL_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
    )

    raw_response = json.loads(Path(record["raw_response_path"]).read_text(encoding="utf-8"))
    assert record["agent"] == "claude-otel"
    assert record["runtime"]["backend"] == "host"
    assert record["final_output"] == {
        "status": "completed",
        "final_answer": "Updated the implementation and verified the change.",
        "notes": "",
    }
    assert record["token_usage"]["source"] == "claude.otel.api_response_body+api_request_cost"
    assert record["token_usage"]["cost_usd"] == 0.02
    assert record["token_usage"]["model_usage"]["claude-test"]["cost_usd"] == 0.02
    assert record["token_usage"]["server_tool_use"] == {"web_search_requests": 0, "web_fetch_requests": 0}
    persisted = record["persisted_tool_results"][0]
    artifact_path = Path(str(persisted["artifact_path"]))
    assert persisted["status"] == "archived"
    assert persisted["label"] == "20KB"
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == "full persisted output\n"
    assert record["tool_calls"][0]["source"] == "claude.otel.api_body_tool_use"
    assert record["tool_calls"][0]["payload"]["input"] == {"file_path": "app.py"}
    assert record["tool_calls"][0]["payload"]["decision"] == "accept"
    assert record["tool_calls"][0]["payload"]["decision_source"] == "config"
    assert record["available_tools"] == ["Read", "StructuredOutput"]
    assert raw_response["otel"]["summary"]["request_count"] == 2
    assert raw_response["otel"]["summary"]["api_body_ref_count"] == 2
    assert raw_response["otel"]["summary"]["api_body_ref_error_count"] == 0
    assert raw_response["otel"]["summary"]["api_request_body_count"] == 1
    assert raw_response["otel"]["summary"]["api_response_body_count"] == 1
    assert raw_response["otel"]["api_body_ref_errors"] == []
    assert raw_response["otel"]["api_response_bodies"][0]["stop_reason"] == "tool_use"
    assert "structured_outputs" not in raw_response["otel"]
    assert "structured_output_count" not in raw_response["otel"]["summary"]
    assert raw_response["otel"]["artifact_retention"] == "compact"
    assert raw_response["otel"]["capture_retained"] is False
    assert raw_response["otel"]["api_body_artifacts_retained"] is False
    assert "requests" not in raw_response["otel"]
    assert "fallback" not in raw_response
    assert "runtime_observations" not in raw_response
    assert not Path(raw_response["otel"]["capture_path"]).exists()
    for ref in raw_response["otel"]["api_body_refs"]:
        assert not Path(ref["body_ref"]).exists()
    assert "fallback_artifacts" not in raw_response
    assert captured["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert captured["env"]["OTEL_LOG_TOOL_DETAILS"] == "1"
    assert str(captured["env"]["OTEL_LOG_RAW_API_BODIES"]).startswith("file:")
    assert Path(str(captured["env"]["OTEL_LOG_RAW_API_BODIES"]).removeprefix("file:")).is_absolute()
    assert "OTEL_SDK_DISABLED" not in captured["env"]
    assert (Path(record["task_dir"]) / "otel-task.claude-otel-record.json").exists()
