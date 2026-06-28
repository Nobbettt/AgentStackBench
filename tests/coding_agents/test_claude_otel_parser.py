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
