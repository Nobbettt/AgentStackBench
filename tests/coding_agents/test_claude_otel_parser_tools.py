# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.agents.claude_otel.parser import ClaudeOtelAgentParser
from contextbench.coding_agents.conversion import convert_run_record
from contextbench.coding_agents.runtime import missing_required_tool_call_patterns

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def test_claude_otel_token_usage_does_not_fallback_to_raw_request_cost_log() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 2,
                    },
                    "tool_uses": [],
                }
            ],
            "requests": [
                {
                    "path": "/v1/logs",
                    "body": logs_payload(
                        log_record(
                            "claude_code.api_request",
                            [
                                otel_attr("cost_usd", 0.5),
                                otel_attr("model", "claude-test"),
                            ],
                        )
                    ),
                }
            ],
        },
    }

    usage = ClaudeOtelAgentParser().extract_token_usage(raw_response)

    assert usage is not None
    assert usage["source"] == "claude.otel.api_response_body"
    assert "cost_usd" not in usage


def test_claude_otel_tool_call_payload_does_not_fallback_to_log_or_span_parameters() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_bash",
                            "name": "Bash",
                            "input": {},
                        }
                    ],
                }
            ],
            "logs": [
                {
                    "name": "claude_code.tool_decision",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "decision": "accept",
                        "tool_parameters": json.dumps(
                            {
                                "bash_command": "sed -n '1,2p' app.py",
                                "full_command": "sed -n '1,2p' app.py",
                            }
                        ),
                    },
                },
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "success": True,
                        "tool_parameters": json.dumps(
                            {
                                "bash_command": "sed -n '1,2p' app.py",
                                "full_command": "sed -n '1,2p' app.py",
                            }
                        ),
                    },
                },
            ],
            "spans": [
                {
                    "name": "claude_code.tool",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "full_command": "sed -n '1,2p' app.py",
                    },
                }
            ],
        },
    }

    calls = ClaudeOtelAgentParser().extract_tool_calls(raw_response)

    assert calls == [
        {
            "source": "claude.otel.api_body_tool_use",
            "tool_name": "Bash",
            "payload": {
                "id": "toolu_bash",
                "input": {},
                "request_id": None,
                "message_id": None,
                "decision": "accept",
                "ok": False,
                "status": "missing_result",
            },
        }
    ]
    assert missing_required_tool_call_patterns(calls, ["sed -n"]) == ["sed -n"]


def test_claude_otel_tool_call_success_uses_api_body_tool_result_error_over_log_success() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_bash",
                            "name": "Bash",
                            "input": {"command": "sed -n '1,2p' missing.py"},
                        }
                    ],
                }
            ],
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_results": [
                        {
                            "tool_use_id": "toolu_bash",
                            "content": "No such file or directory",
                            "is_error": True,
                        }
                    ],
                }
            ],
            "logs": [
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "success": True,
                    },
                }
            ],
        },
    }

    calls = ClaudeOtelAgentParser().extract_tool_calls(raw_response)

    assert calls[0]["payload"]["ok"] is False
    assert calls[0]["payload"]["result"] == {"content_chars": 25, "is_error": True}
    assert missing_required_tool_call_patterns(calls, ["Bash"]) == ["Bash"]


def test_claude_otel_tool_call_success_does_not_fallback_to_log_without_api_body_result() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_bash",
                            "name": "Bash",
                            "input": {"command": "sed -n '1,2p' app.py"},
                        }
                    ],
                }
            ],
            "logs": [
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "Bash",
                        "tool_use_id": "toolu_bash",
                        "success": True,
                    },
                }
            ],
        },
    }

    calls = ClaudeOtelAgentParser().extract_tool_calls(raw_response)

    assert calls[0]["payload"]["ok"] is False
    assert calls[0]["payload"]["status"] == "missing_result"
    assert "result" not in calls[0]["payload"]
    assert missing_required_tool_call_patterns(calls, ["Bash"]) == ["Bash"]


def test_claude_otel_missing_api_body_tool_result_does_not_satisfy_required_mcp_call() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_mcp",
                            "name": "mcp__cortex__context_search",
                            "input": {"query": "target"},
                        }
                    ],
                }
            ],
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_results": [],
                }
            ],
        },
    }

    calls = ClaudeOtelAgentParser().extract_tool_calls(raw_response)

    assert calls[0]["payload"]["ok"] is False
    assert calls[0]["payload"]["status"] == "missing_result"
    assert "result" not in calls[0]["payload"]
    assert missing_required_tool_call_patterns(calls, [r"^mcp__cortex__"]) == [r"^mcp__cortex__"]


def test_claude_otel_tool_calls_skip_zero_count_server_tool_use() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
                    },
                    "tool_uses": [],
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_tool_calls(raw_response) == []


