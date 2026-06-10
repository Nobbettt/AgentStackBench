# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import SanitizationContext
from scripts.export_comparison_data import (
    ComparisonExportError,
    _extract_trace_entries,
    _trace_action_counts,
    _trace_entry_counts,
    _extract_skill_counts,
    build_comparison_export,
    build_comparison_payload,
)

from .helpers import _record, _write

def test_extract_skill_counts_ignores_non_object_trace_events() -> None:
    raw_response = {
        "events": [
            "stream marker",
            {"type": "item.completed", "item": "not-an-object"},
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "cat /tmp/home/.agents/skills/superpowers/debug/SKILL.md",
                },
            },
        ],
    }

    assert _extract_skill_counts(raw_response) == {"debug": 1}
def test_extract_trace_entries_ignores_non_object_trace_events() -> None:
    raw_response = {
        "events": [
            "stream marker",
            {"type": "item.completed", "item": "not-an-object"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "Done",
                },
            },
        ],
    }

    assert _extract_trace_entries(raw_response, sanitize_context=SanitizationContext()) == [
        {"kind": "assistant_message", "text": "Done"}
    ]


def test_extract_trace_entries_supports_claude_verbose_response() -> None:
    context = SanitizationContext()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning should not be exported"},
                        {"type": "text", "text": "I will inspect /Users/nobbe/project/app.py"},
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "sed -n 1,5p /Users/nobbe/project/app.py"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "output from /Users/nobbe/.claude/settings.json",
                        }
                    ],
                },
                "tool_use_result": {"stdout": "output", "stderr": ""},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-2",
                            "name": "mcp__cortex__context_search",
                            "input": {"query": "django queryset delete"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-2",
                            "content": [{"type": "text", "text": "Cortex result"}],
                        }
                    ],
                },
                "tool_use_result": {"content": "Cortex result"},
            },
        ],
    }

    entries = _extract_trace_entries(raw_response, sanitize_context=context)

    assert entries[0] == {"kind": "assistant_message", "text": "I will inspect <home>/project/app.py"}
    assert entries[1]["kind"] == "command_execution"
    assert entries[1]["status"] == "completed"
    assert entries[1]["command"] == "sed -n 1,5p <home>/project/app.py"
    assert entries[1]["output"] == "output from <home>/.claude/settings.json"
    assert entries[2]["kind"] == "tool_use"
    assert entries[2]["status"] == "completed"
    assert entries[2]["command"].startswith("mcp__cortex__context_search")
    assert entries[2]["output"] == "Cortex result"
    assert "private reasoning" not in json.dumps(entries)


def test_extract_trace_entries_renders_codex_mcp_tool_calls() -> None:
    context = SanitizationContext()
    raw_response = {
        "agent": "codex",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "cortex",
                    "name": "context_search",
                    "input": {"query": "django queryset delete"},
                    "result": {"matches": [{"path": "src/models.py", "title": "Model.delete"}]},
                    "status": "completed",
                },
            }
        ],
    }

    entries = _extract_trace_entries(raw_response, sanitize_context=context)

    assert entries == [
        {
            "kind": "tool_use",
            "status": "completed",
            "command": 'mcp__cortex__context_search {"query": "django queryset delete"}',
            "output": '{"matches": [{"path": "src/models.py", "title": "Model.delete"}]}',
            "payload": {
                "toolName": "mcp__cortex__context_search",
                "input": {"query": "django queryset delete"},
                "result": {"matches": [{"path": "src/models.py", "title": "Model.delete"}]},
            },
        }
    ]


def test_extract_trace_entries_preserves_codex_file_change_paths() -> None:
    raw_response = {
        "agent": "codex",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "file_change",
                    "changes": [
                        {"path": "pkg/a.go", "kind": "update"},
                        {"path": "pkg/b.go", "kind": "create"},
                    ],
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "file_change",
                    "changes": [{"path": "pkg/c.go", "kind": "update"}],
                    "status": "completed",
                },
            },
        ],
    }

    entries = _extract_trace_entries(raw_response, sanitize_context=SanitizationContext())

    assert entries == [
        {
            "kind": "file_change",
            "status": "completed",
            "payload": {
                "changes": [
                    {"path": "pkg/a.go", "kind": "update"},
                    {"path": "pkg/b.go", "kind": "create"},
                ],
            },
        },
        {
            "kind": "file_change",
            "status": "completed",
            "payload": {
                "changes": [{"path": "pkg/c.go", "kind": "update"}],
                "path": "pkg/c.go",
                "kind": "update",
            },
        },
    ]


def test_trace_entry_counts_split_visible_rows_from_agent_actions() -> None:
    entries = [
        {"kind": "assistant_message", "text": "I will inspect the repo"},
        {"kind": "todo_list", "payload": {"items": []}},
        {"kind": "command_execution", "command": "rg bug"},
        {"kind": "tool_use", "command": "mcp__cortex__context_search"},
        {"kind": "tool_result", "command": "Result for orphaned-tool"},
        {"kind": "file_change", "payload": {"path": "a.py"}},
    ]

    assert _trace_entry_counts(entries) == {
        "rawTraceEvents": 6,
        "rawAgentActions": 3,
    }


def test_raw_trace_counts_match_exported_trace_entries() -> None:
    raw_response = {
        "agent": "codex",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": f"echo {index}",
                    "aggregated_output": "",
                    "status": "completed",
                    "exit_code": 0,
                },
            }
            for index in range(125)
        ],
    }

    entries = _extract_trace_entries(raw_response, sanitize_context=SanitizationContext())

    assert len(entries) == 125
    assert _trace_entry_counts(entries) == {
        "rawTraceEvents": 125,
        "rawAgentActions": 125,
    }


def test_trace_action_counts_split_claude_tool_buckets() -> None:
    raw_response = {
        "agent": "claude",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "pytest"}},
                        {"type": "tool_use", "id": "tool-2", "name": "Read", "input": {"file_path": "a.py"}},
                        {"type": "tool_use", "id": "tool-3", "name": "Edit", "input": {"file_path": "a.py"}},
                        {
                            "type": "tool_use",
                            "id": "tool-4",
                            "name": "mcp__cortex__context_search",
                            "input": {"query": "bug"},
                        },
                    ],
                },
            }
        ],
    }
    record = {
        "tool_calls": [
            {"tool_name": "Bash", "payload": {"result": {"is_error": False}}},
            {"tool_name": "Read", "payload": {"result": {"is_error": False}}},
            {"tool_name": "Edit", "payload": {"result": {"is_error": False}}},
            {
                "tool_name": "mcp__cortex__context_search",
                "payload": {"mcp_server": "cortex", "mcp_tool": "context_search", "result": {"is_error": False}},
            },
        ],
        "tool_call_summary": {"total": 4, "mcp_total": 1, "mcp_successful_total": 1},
    }

    assert _trace_action_counts(raw_response, record) == {
        "toolCalls": 4,
        "mcpToolCalls": 1,
        "successfulMcpToolCalls": 1,
        "commandExecutions": 1,
        "readToolCalls": 1,
        "editToolCalls": 1,
    }


def test_trace_action_counts_split_codex_trace_buckets() -> None:
    raw_response = {
        "agent": "codex",
        "events": [
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "pytest", "status": "completed"},
            },
            {
                "type": "item.completed",
                "item": {"type": "file_change", "path": "a.py", "status": "completed"},
            },
            {
                "type": "item.completed",
                "item": {"type": "mcp_tool_call", "tool_name": "mcp__cortex__context_search"},
            },
        ],
    }
    record = {
        "tool_calls": [
            {
                "tool_name": "mcp__cortex__context_search",
                "payload": {"mcp_server": "cortex", "mcp_tool": "context_search", "result": {"is_error": False}},
            }
        ],
        "tool_call_summary": {"total": 1, "mcp_total": 1, "mcp_successful_total": 1},
    }

    assert _trace_action_counts(raw_response, record) == {
        "toolCalls": 1,
        "mcpToolCalls": 1,
        "successfulMcpToolCalls": 1,
        "commandExecutions": 1,
        "readToolCalls": 0,
        "editToolCalls": 1,
    }


def test_trace_action_counts_treats_mcp_ok_false_as_unsuccessful_without_summary() -> None:
    raw_response = {"agent": "codex", "events": []}
    record = {
        "tool_calls": [
            {
                "tool_name": "mcp__cortex__context_search",
                "payload": {"mcp_server": "cortex", "mcp_tool": "context_search", "ok": False},
            }
        ]
    }

    assert _trace_action_counts(raw_response, record)["successfulMcpToolCalls"] == 0
