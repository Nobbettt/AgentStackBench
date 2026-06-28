from __future__ import annotations

from pathlib import Path

from contextbench.artifact_sanitization import SanitizationContext
from scripts.export_comparison_data import _extract_mcp_usage, _trace_action_counts


def test_extract_mcp_usage_marks_returned_paths_followed_and_overlapping(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    record = {
        "available_tools": [
            "mcp__example__context_search",
            "mcp__example__context_get_related",
        ],
        "tool_call_summary": {
            "mcp_total": 1,
            "mcp_successful_total": 1,
            "by_name": {"mcp__example__context_search": 1},
            "successful_by_name": {"mcp__example__context_search": 1},
        },
    }
    raw_response = {
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "mcp__example__context_search",
                            "input": {"query": "target symbol", "top_k": 3},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "content": ""}]},
                "tool_use_result": {
                    "content": (
                        '{"results": ['
                        '{"path": "src/target.py", "title": "target"},'
                        '{"path": "tests/test_target.py", "title": "test_target"}'
                        '], "total_candidates": 42}'
                    )
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "Read",
                            "input": {"file_path": str(workspace / "src" / "target.py")},
                        }
                    ]
                },
            },
        ]
    }

    usage = _extract_mcp_usage(
        raw_response,
        record,
        final_output={"retrieved_context_files": ["src/target.py"]},
        model_patch="diff --git a/src/target.py b/src/target.py\n",
        workspace_path=str(workspace),
        candidates={"src/target.py", "tests/test_target.py"},
        sanitize_context=SanitizationContext(repo_root=tmp_path, workspace_path=workspace),
    )

    assert usage["toolCalls"] == 1
    assert usage["successfulToolCalls"] == 1
    assert usage["callsWithResults"] == 1
    assert usage["meaningfulCalls"] == 1
    assert usage["callsWithFinalContextOverlap"] == 1
    assert usage["callsWithPatchOverlap"] == 1
    assert usage["callsWithFollowupOnReturnedPath"] == 1
    assert usage["calls"][0]["topPaths"] == ["src/target.py", "tests/test_target.py"]
    assert usage["calls"][0]["overlapFinalContextFiles"] == ["src/target.py"]
    assert usage["calls"][0]["overlapPatchFiles"] == ["src/target.py"]
    assert usage["calls"][0]["followedReturnedPaths"] == ["src/target.py"]


def test_extract_mcp_usage_uses_codex_mcp_events_for_call_details(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    record = {
        "tool_calls": [
            {
                "source": "codex.item",
                "tool_name": "mcp__cortex__context_search",
                "payload": {
                    "mcp_server": "cortex",
                    "mcp_tool": "context_search",
                    "input": {"query": "target symbol", "top_k": 5},
                    "result": {
                        "matches": [
                            {"path": "src/target.py", "title": "target"},
                            {"path": "tests/test_target.py", "title": "test_target"},
                        ],
                        "total_candidates": 9,
                    },
                    "status": "completed",
                },
            }
        ],
        "tool_call_summary": {
            "mcp_total": 1,
            "mcp_successful_total": 1,
            "by_name": {"mcp__cortex__context_search": 1},
            "successful_by_name": {"mcp__cortex__context_search": 1},
        },
    }
    raw_response = {
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "cortex",
                    "name": "context_search",
                    "input": {"query": "target symbol", "top_k": 5},
                    "result": {
                        "matches": [
                            {"path": "src/target.py", "title": "target"},
                            {"path": "tests/test_target.py", "title": "test_target"},
                        ],
                        "total_candidates": 9,
                    },
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": f"sed -n '1,80p' {workspace / 'src' / 'target.py'}",
                    "status": "completed",
                },
            },
        ]
    }

    usage = _extract_mcp_usage(
        raw_response,
        record,
        final_output={"retrieved_context_files": ["src/target.py"]},
        model_patch="diff --git a/src/target.py b/src/target.py\n",
        workspace_path=str(workspace),
        candidates={"src/target.py", "tests/test_target.py"},
        sanitize_context=SanitizationContext(repo_root=tmp_path, workspace_path=workspace),
    )

    assert usage["toolCalls"] == 1
    assert usage["successfulToolCalls"] == 1
    assert usage["callsWithResults"] == 1
    assert usage["meaningfulCalls"] == 1
    assert usage["callsWithFollowupOnReturnedPath"] == 1
    assert usage["calls"][0]["toolName"] == "mcp__cortex__context_search"
    assert usage["calls"][0]["query"] == "target symbol"
    assert usage["calls"][0]["topK"] == 5
    assert usage["calls"][0]["totalCandidates"] == 9
    assert usage["calls"][0]["topPaths"] == ["src/target.py", "tests/test_target.py"]
    assert usage["calls"][0]["followedReturnedPaths"] == ["src/target.py"]


def test_extract_mcp_usage_uses_codex_mcp_tool_bridge_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    command = (
        "$CONTEXTBENCH_RUNTIME_ROOT/bin/mcp-tool find_code "
        '\'{"query": "target symbol", "limit": 5}\''
    )
    result_payload = {
        "content": [
            {"type": "text", "text": "Found 2 matches."},
            {
                "type": "text",
                "text": (
                    '{"count": 2, "results": ['
                    '{"path": "src/target.py", "title": "target"},'
                    '{"path": "tests/test_target.py", "title": "test_target"}'
                    ']}'
                ),
            },
        ],
        "isError": False,
    }
    raw_response = {
        "agent": "codex",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": command,
                    "aggregated_output": result_payload,
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": f"sed -n '1,80p' {workspace / 'src' / 'target.py'}",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
        ],
    }
    record = {
        "tool_call_summary": {
            "mcp_total": 0,
            "mcp_successful_total": 0,
            "by_name": {},
            "successful_by_name": {},
        },
    }

    usage = _extract_mcp_usage(
        raw_response,
        record,
        final_output={"retrieved_context_files": ["src/target.py"]},
        model_patch="diff --git a/src/target.py b/src/target.py\n",
        workspace_path=str(workspace),
        candidates={"src/target.py", "tests/test_target.py"},
        sanitize_context=SanitizationContext(repo_root=tmp_path, workspace_path=workspace),
    )

    assert _trace_action_counts(raw_response, record)["mcpToolCalls"] == 1
    assert _trace_action_counts(raw_response, record)["successfulMcpToolCalls"] == 1
    assert usage["availableTools"] == ["mcp__cli__find_code"]
    assert usage["toolCalls"] == 1
    assert usage["successfulToolCalls"] == 1
    assert usage["callsWithResults"] == 1
    assert usage["meaningfulCalls"] == 1
    assert usage["byTool"] == [{"name": "mcp__cli__find_code", "calls": 1, "successfulCalls": 1}]
    assert usage["calls"][0]["toolName"] == "mcp__cli__find_code"
    assert usage["calls"][0]["query"] == "target symbol"
    assert usage["calls"][0]["resultCount"] == 2
    assert usage["calls"][0]["totalCandidates"] == 2
    assert usage["calls"][0]["topPaths"] == ["src/target.py", "tests/test_target.py"]
    assert usage["calls"][0]["followedReturnedPaths"] == ["src/target.py"]
