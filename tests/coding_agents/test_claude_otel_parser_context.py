# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.agents.claude_otel.parser import ClaudeOtelAgentParser
from contextbench.coding_agents.conversion import convert_run_record
from contextbench.coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from contextbench.coding_agents.runtime import missing_required_tool_call_patterns

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def test_claude_otel_token_usage_does_not_fallback_without_api_response_usage() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "logs": [
                {
                    "name": "claude_code.api_request",
                    "attributes": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_tokens": 3,
                        "cache_creation_tokens": 2,
                        "cost_usd": 0.01,
                        "model": "claude-test",
                    },
                }
            ],
            "spans": [
                {
                    "name": "claude_code.llm_request",
                    "attributes": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_tokens": 3,
                        "cache_creation_tokens": 2,
                    },
                }
            ],
            "metrics": [
                {
                    "name": "claude_code.token.usage",
                    "points": [
                        {"attributes": {"type": "input"}, "as_int": 10},
                        {"attributes": {"type": "output"}, "as_int": 5},
                    ],
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_token_usage(raw_response) is None


def test_claude_otel_conversion_scores_otel_context_not_agent_report() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "app.py"},
                        }
                    ],
                }
            ],
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_results": [
                        {
                            "tool_use_id": "toolu_read",
                            "content": "1\tdef main():\n2\t    return 1\n",
                        }
                    ],
                }
            ],
        },
    }
    record = {
        "agent": "claude-otel",
        "instance_id": "task-1",
        "workspace_path": str(Path.cwd()),
        "final_output": {
            "status": "completed",
            "final_answer": "done",
            "notes": "",
            "retrieved_context_files": ["/outside/workspace.py"],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
        },
        "raw_response": raw_response,
        "model_patch": "",
    }

    converted = convert_run_record(record)
    traj = converted["traj_data"]

    assert traj["pred_files"] == ["app.py"]
    assert traj["pred_spans"] == {"app.py": [{"start": 1, "end": 2}]}
    assert traj["pred_files_provenance"] == {"app.py": "otel_tool_results"}
    assert traj["pred_files_source"] == ["otel_tool_results"]


def test_claude_otel_conversion_does_not_fallback_to_agent_report_without_otel_context() -> None:
    record = {
        "agent": "claude-otel",
        "instance_id": "task-1",
        "workspace_path": str(Path.cwd()),
        "final_output": {
            "status": "completed",
            "final_answer": "done",
            "notes": "",
            "retrieved_context_files": ["app.py"],
            "retrieved_context_spans": {"app.py": [{"start": 1, "end": 2}]},
            "retrieved_context_symbols": {"app.py": ["main"]},
        },
        "raw_response": {
            "agent": "claude-otel",
            "response_format": "otel-http-json",
            "otel": {
                "api_response_bodies": [],
                "api_request_bodies": [],
            },
        },
        "model_patch": "",
    }

    converted = convert_run_record(record)
    traj = converted["traj_data"]

    assert traj["pred_files"] == []
    assert traj["pred_spans"] == {}
    assert traj["pred_symbols"] == {}
    assert traj["pred_files_source"] == []


def test_claude_otel_trajectory_does_not_fallback_to_span_tool_output() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "app.py"},
                        }
                    ],
                }
            ],
            "spans": [
                {
                    "name": "claude_code.tool",
                    "attributes": {"tool_name": "Read", "tool_use_id": "toolu_read"},
                    "events": [
                        {
                            "name": "tool.output",
                            "attributes": {"tool_output": "1\tdef main():\n2\t    return 1\n"},
                        }
                    ],
                }
            ],
        },
    }

    traj = ClaudeOtelAgentParser().infer_trajectory_data(raw_response, record={"workspace_path": str(Path.cwd())})

    assert traj is None


def test_claude_otel_trajectory_uses_api_body_tool_input_only() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {},
                        },
                        {
                            "id": "toolu_bash",
                            "name": "Bash",
                            "input": {},
                        },
                    ],
                }
            ],
            "spans": [
                {
                    "name": "claude_code.tool",
                    "attributes": {
                        "tool_name": "Read",
                        "tool_use_id": "toolu_read",
                        "file_path": "app.py",
                    },
                },
                {
                    "name": "claude_code.tool",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "full_command": "sed -n '1,2p' app.py",
                    },
                },
            ],
            "logs": [
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "success": True,
                        "tool_parameters": json.dumps({"full_command": "sed -n '1,2p' app.py"}),
                    },
                }
            ],
        },
    }

    traj = ClaudeOtelAgentParser().infer_trajectory_data(raw_response, record={"workspace_path": str(Path.cwd())})

    assert traj is None


def test_claude_otel_trajectory_drops_truncated_read_output_spans(tmp_path) -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "app.py"},
                        }
                    ],
                }
            ],
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_results": [
                        {
                            "tool_use_id": "toolu_read",
                            "content": "1 first line\n2 second line\n",
                            "content_chars": len("1 first line\n2 second line\n"),
                            "original_content_chars": MAX_COMMAND_OUTPUT_CHARS + 1,
                            "content_truncated": True,
                            "is_error": False,
                        }
                    ],
                }
            ],
        },
    }

    traj = ClaudeOtelAgentParser().infer_trajectory_data(raw_response, record={"workspace_path": str(tmp_path)})

    assert traj is not None
    assert traj["pred_files"] == ["app.py"]
    assert traj["pred_spans"] == {}
    assert traj["trace_inference_meta"]["dropped_large_command_outputs"] == 1

