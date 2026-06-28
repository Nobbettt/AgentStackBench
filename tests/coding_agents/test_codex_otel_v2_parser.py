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


