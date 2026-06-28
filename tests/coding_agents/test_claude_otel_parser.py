# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from contextbench.agents.claude_otel.parser import ClaudeOtelAgentParser
from contextbench.coding_agents.conversion import convert_run_record
from contextbench.coding_agents.runtime import missing_required_tool_call_patterns

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def test_claude_otel_parser_prefers_otel_usage_and_tools() -> None:
    otel_structured = {"status": "completed", "final_answer": "from otel api body", "notes": ""}
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "request_id": "req_1",
                    "model": "claude-test",
                    "tool_names": ["Read", "Bash", "StructuredOutput"],
                    "tool_count": 3,
                    "tool_results": [
                        {
                            "tool_use_id": "toolu_read",
                            "content": "1\tdef main():\n2\t    return 1\n",
                        }
                    ],
                }
            ],
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "request_id": "req_1",
                    "message_id": "msg_1",
                    "model": "claude-test",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 1,
                        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
                    },
                    "tool_uses": [
                        {
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "app.py"},
                        },
                        {
                            "id": "toolu_structured",
                            "name": "StructuredOutput",
                            "input": otel_structured,
                        },
                    ],
                },
                {
                    "source": "claude.otel.api_response_body",
                    "request_id": "req_2",
                    "message_id": "msg_2",
                    "model": "claude-test",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 4,
                        "cache_creation_input_tokens": 1,
                        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 1},
                    },
                    "tool_uses": [],
                },
            ],
            "logs": [
                {
                    "name": "claude_code.api_request",
                    "attributes": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "cache_read_tokens": 3,
                        "cache_creation_tokens": 2,
                        "cost_usd": 0.012,
                        "model": "claude-test",
                    },
                },
                {
                    "name": "claude_code.tool_decision",
                    "attributes": {
                        "tool_name": "Read",
                        "tool_use_id": "toolu_read",
                        "decision": "accept",
                        "source": "config",
                    },
                },
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "StructuredOutput",
                        "tool_use_id": "toolu_structured",
                        "success": "true",
                        "tool_input": json.dumps(
                            {
                                **otel_structured,
                                "retrieved_context_spans": ["<nested>"],
                                "retrieved_context_symbols": ["<nested>"],
                            }
                        ),
                    },
                },
            ],
            "spans": [
                {
                    "name": "claude_code.tool",
                    "attributes": {
                        "tool_name": "Read",
                        "tool_use_id": "toolu_read",
                        "file_path": "app.py",
                    },
                    "events": [
                        {
                            "name": "tool.output",
                            "attributes": {"tool_output": "def main():\n    return 1\n"},
                        }
                    ],
                }
            ],
        },
    }

    parser = ClaudeOtelAgentParser()

    structured = parser.extract_structured_output(raw_response)
    assert structured == otel_structured
    usage = parser.extract_token_usage(raw_response)
    assert usage["source"] == "claude.otel.api_response_body+api_request_cost"
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 4
    assert usage["cache_read_input_tokens"] == 7
    assert usage["cache_creation_input_tokens"] == 2
    assert usage["cost_usd"] == 0.012
    assert usage["server_tool_use"] == {"web_search_requests": 1, "web_fetch_requests": 1}
    assert usage["model_usage"]["claude-test"]["cost_usd"] == 0.012
    assert usage["last_api_response_input_tokens"] == 3
    assert usage["last_api_response_output_tokens"] == 2
    calls = parser.extract_tool_calls(raw_response)
    assert [call["source"] for call in calls] == [
        "claude.otel.api_body_tool_use",
        "claude.otel.api_body_tool_use",
        "claude.otel.server_tool_use",
    ]
    assert calls[0]["tool_name"] == "Read"
    assert calls[0]["payload"]["input"] == {"file_path": "app.py"}
    assert calls[0]["payload"]["decision"] == "accept"
    assert calls[0]["payload"]["decision_source"] == "config"
    assert calls[2]["payload"] == {"web_search_requests": 1, "web_fetch_requests": 1}
    assert parser.extract_available_tools(raw_response) == ["Read", "Bash", "StructuredOutput"]
    traj = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(Path.cwd())})
    assert traj is not None
    assert traj["pred_steps"][0]["files"] == ["app.py"]
    assert traj["pred_steps"][0]["spans"] == {"app.py": [{"start": 1, "end": 2}]}


def test_claude_otel_available_tools_does_not_fallback_to_runtime_list() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_names": ["Read", "StructuredOutput"],
                    "tool_count": 2,
                }
            ]
        },
        "runtime_observations": {
            "runtime_available_tools": ["Read", "StructuredOutput", "WebSearch", "mcp__cortex__context_get_rules"],
            "runtime_available_tools_source": "claude.stream.init",
        },
    }

    assert ClaudeOtelAgentParser().extract_available_tools(raw_response) == ["Read", "StructuredOutput"]


def test_claude_otel_tool_calls_do_not_fallback_to_decision_only_logs() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "logs": [
                {
                    "name": "claude_code.tool_decision",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "decision": "reject",
                        "source": "policy",
                        "tool_parameters": json.dumps(
                            {
                                "bash_command": "curl",
                                "full_command": "curl https://example.com",
                                "description": "network request",
                            }
                        ),
                    },
                }
            ],
        },
    }

    calls = ClaudeOtelAgentParser().extract_tool_calls(raw_response)

    assert calls == []
    assert missing_required_tool_call_patterns(calls, ["Bash"]) == ["Bash"]


def test_claude_otel_api_body_ref_does_not_fallback_to_inline_body_when_ref_is_missing(tmp_path) -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "requests": [
                {
                    "path": "/v1/logs",
                    "body": logs_payload(
                        log_record(
                            "claude_code.api_response_body",
                            [
                                otel_attr("body_ref", str(tmp_path / "missing-response-body.json")),
                                otel_attr(
                                    "body",
                                    json.dumps(
                                        {
                                            "content": [
                                                {
                                                    "type": "tool_use",
                                                    "id": "toolu_read",
                                                    "name": "Read",
                                                    "input": {"file_path": "app.py"},
                                                }
                                            ]
                                        }
                                    ),
                                ),
                            ],
                        )
                    ),
                }
            ]
        },
    }

    assert ClaudeOtelAgentParser().extract_tool_calls(raw_response) == []

def test_claude_otel_api_body_event_without_ref_does_not_use_inline_body() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "requests": [
                {
                    "path": "/v1/logs",
                    "body": logs_payload(
                        log_record(
                            "claude_code.api_response_body",
                            [
                                otel_attr(
                                    "body",
                                    json.dumps(
                                        {
                                            "content": [
                                                {
                                                    "type": "tool_use",
                                                    "id": "toolu_read",
                                                    "name": "Read",
                                                    "input": {"file_path": "app.py"},
                                                }
                                            ]
                                        }
                                    ),
                                ),
                            ],
                        )
                    ),
                }
            ]
        },
    }

    assert ClaudeOtelAgentParser().extract_tool_calls(raw_response) == []


def test_claude_otel_raw_request_body_ref_without_compact_summary_is_ignored(tmp_path) -> None:
    response_body = tmp_path / "response-body.json"
    response_body.write_text(
        json.dumps(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read",
                        "name": "Read",
                        "input": {"file_path": "app.py"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "requests": [
                {
                    "path": "/v1/logs",
                    "body": logs_payload(
                        log_record(
                            "claude_code.api_response_body",
                            [otel_attr("body_ref", str(response_body))],
                        )
                    ),
                }
            ]
        },
    }

    assert ClaudeOtelAgentParser().extract_tool_calls(raw_response) == []


def test_claude_otel_stored_empty_api_bodies_do_not_fallback_to_raw_requests() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [],
            "requests": [
                {
                    "path": "/v1/logs",
                    "body": logs_payload(
                        log_record(
                            "claude_code.api_response_body",
                            [
                                otel_attr(
                                    "body",
                                    json.dumps(
                                        {
                                            "content": [
                                                {
                                                    "type": "tool_use",
                                                    "id": "toolu_read",
                                                    "name": "Read",
                                                    "input": {"file_path": "app.py"},
                                                }
                                            ]
                                        }
                                    ),
                                ),
                            ],
                        )
                    ),
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_tool_calls(raw_response) == []
