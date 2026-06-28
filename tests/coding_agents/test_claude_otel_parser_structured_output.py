# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.agents.claude_otel.parser import ClaudeOtelAgentParser
from contextbench.coding_agents.conversion import convert_run_record
from contextbench.coding_agents.runtime import missing_required_tool_call_patterns

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def test_claude_otel_structured_output_does_not_fallback_to_tool_result_log(make_final_output) -> None:
    legacy_structured = make_final_output(final_answer="from tool log")
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "logs": [
                {
                    "name": "claude_code.tool_result",
                    "attributes": {
                        "tool_name": "StructuredOutput",
                        "tool_use_id": "toolu_structured",
                        "success": True,
                        "tool_input": json.dumps(legacy_structured),
                    },
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_structured_output(raw_response) is None


def test_claude_otel_structured_output_does_not_scan_nested_or_string_values() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_structured",
                            "name": "StructuredOutput",
                            "input": {
                                "wrapper": {
                                    "status": "completed",
                                    "final_answer": "nested",
                                    "notes": "",
                                },
                                "text": json.dumps(
                                    {
                                        "status": "completed",
                                        "final_answer": "string",
                                        "notes": "",
                                    }
                                ),
                            },
                        }
                    ],
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_structured_output(raw_response) is None


def test_claude_otel_structured_output_ignores_legacy_context_fields(make_final_output) -> None:
    legacy_structured = make_final_output(
        final_answer="from tool log",
        retrieved_context_files=["app.py"],
        retrieved_context_spans=["<nested>"],
        retrieved_context_symbols=["<nested>"],
    )
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_structured",
                            "name": "StructuredOutput",
                            "input": legacy_structured,
                        }
                    ],
                }
            ],
        },
    }

    assert ClaudeOtelAgentParser().extract_structured_output(raw_response) == {
        "status": "completed",
        "final_answer": "from tool log",
        "notes": "",
    }


def test_claude_otel_structured_output_accepts_minimal_v2_schema() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_response_bodies": [
                {
                    "source": "claude.otel.api_response_body",
                    "tool_uses": [
                        {
                            "id": "toolu_structured",
                            "name": "StructuredOutput",
                            "input": {
                                "status": "completed",
                                "final_answer": "done",
                                "notes": "",
                            },
                        }
                    ],
                }
            ]
        },
    }

    structured = ClaudeOtelAgentParser().extract_structured_output(raw_response)

    assert structured == {"status": "completed", "final_answer": "done", "notes": ""}
