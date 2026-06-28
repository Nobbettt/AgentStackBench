# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-09 from upstream ContextBench.
# Summary of changes: cover safe repo-root path inference, conservative search inference, trace guards, and effective file normalization for coding-agent parsers.

from __future__ import annotations

import json

import jsonschema
import pytest

from contextbench.agents.claude import ClaudeAgentParser
from contextbench.agents.codex import CodexAgentParser
from contextbench.coding_agents import (
    build_claude_raw_response,
    build_codex_raw_response,
    convert_run_record,
    extract_structured_output_from_value,
)
from contextbench.coding_agents.runtime import (
    missing_required_command_patterns,
    missing_required_tool_call_patterns,
    summarize_tool_calls,
)
from contextbench.coding_agents.constants import CLAUDE_OUTPUT_SCHEMA_PATH, CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.trace_inference import (
    infer_file_list_from_text,
    infer_grep_spans_from_text,
    infer_retrieval_step_from_command,
    trajectory_from_steps,
)
from contextbench.parsers.trajectory import load_pred, parse_trajectory

def test_codex_parser_parses_observed_raw_response_fixture(fixtures_root) -> None:
    parser = CodexAgentParser()
    raw_response = json.loads((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"))

    structured = parser.extract_structured_output(raw_response)
    usage = parser.extract_token_usage(raw_response)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert "task_id" not in structured
    assert structured["retrieved_context_files"] == ["a.py"]
    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 12667,
        "output_tokens": 35,
        "cached_input_tokens": 5504,
        "total_tokens": 12702,
        "cache_read_input_tokens": 5504,
    }
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "repo.search"


def test_load_pred_fails_on_git_lfs_pointer(tmp_path) -> None:
    pred_path = tmp_path / "output.jsonl"
    pred_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc123\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Prediction file is a Git LFS pointer"):
        load_pred(str(pred_path))

def test_codex_parser_ignores_overly_large_command_output_for_inference(tmp_path) -> None:
    parser = CodexAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/zsh -lc 'rg -n \"foo\" src tests -S'",
                    "aggregated_output": "x" * 200000,
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ],
    }

    inferred = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})

    assert inferred is not None
    assert inferred.get("trace_inference_meta", {}).get("dropped_large_command_outputs") == 1


def test_claude_parser_ignores_overly_large_bash_output_for_inference(tmp_path) -> None:
    parser = ClaudeAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "claude",
        "response_format": "stream-json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "rg -n foo src tests"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "src/a.py:1:foo\n" + ("x" * 200000),
                        }
                    ]
                },
            },
        ],
    }

    inferred = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})

    assert inferred is not None
    assert inferred["pred_files"] == []
    assert inferred.get("trace_inference_meta", {}).get("dropped_large_command_outputs") == 1


def test_claude_parser_preserves_read_file_identity_when_output_is_large(tmp_path) -> None:
    parser = ClaudeAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "claude",
        "response_format": "stream-json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "src/a.py"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "1→foo\n" + ("x" * 200000),
                        }
                    ]
                },
            },
        ],
    }

    inferred = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})

    assert inferred is not None
    assert inferred["pred_files"] == ["src/a.py"]
    assert inferred["pred_spans"] == {}
    assert inferred.get("trace_inference_meta", {}).get("dropped_large_command_outputs") == 1


def test_codex_parser_does_not_infer_failed_command_paths(tmp_path) -> None:
    parser = CodexAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "sed -n '1,120p' index.d.ts",
                    "aggregated_output": "sed: can't read index.d.ts: No such file or directory\n",
                    "exit_code": 2,
                    "status": "failed",
                },
            }
        ],
    }

    assert parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)}) is None


def test_codex_parser_extracts_item_level_mcp_tool_calls_for_metadata(tmp_path) -> None:
    parser = CodexAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "mcp_tool_call",
                    "server": "cortex",
                    "name": "context_search",
                    "status": "completed",
                    "result": {"matches": [{"path": "src/a.py", "line": 12}]},
                },
            }
        ],
    }

    tool_calls = parser.extract_tool_calls(raw_response)
    inferred = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})
    summary = summarize_tool_calls(tool_calls)

    assert tool_calls[0]["source"] == "codex.item"
    assert tool_calls[0]["tool_name"] == "mcp__cortex__context_search"
    assert tool_calls[0]["payload"]["mcp_server"] == "cortex"
    assert tool_calls[0]["payload"]["mcp_tool"] == "context_search"
    assert summary["mcp_total"] == 1
    assert summary["mcp_successful_total"] == 1
    assert missing_required_tool_call_patterns(tool_calls, [r"^mcp__cortex__"]) == []
    assert inferred is not None
    assert inferred["pred_files"] == ["src/a.py"]


def test_codex_parser_extracts_native_mcp_tool_field_for_metadata() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "mcp_tool_call",
                    "server": "memtrace",
                    "tool": "find_code",
                    "status": "completed",
                    "result": {"content": [{"type": "text", "text": "[]"}]},
                },
            }
        ],
    }

    tool_calls = parser.extract_tool_calls(raw_response)
    summary = summarize_tool_calls(tool_calls)

    assert tool_calls[0]["source"] == "codex.item"
    assert tool_calls[0]["tool_name"] == "mcp__memtrace__find_code"
    assert tool_calls[0]["payload"]["mcp_server"] == "memtrace"
    assert tool_calls[0]["payload"]["mcp_tool"] == "find_code"
    assert summary["mcp_total"] == 1
    assert summary["mcp_successful_total"] == 1
    assert summary["mcp_by_server"] == {"memtrace": 1}
    assert summary["mcp_by_tool"] == {"memtrace/find_code": 1}
    assert missing_required_tool_call_patterns(tool_calls, [r"^mcp__memtrace__"]) == []


def test_codex_mcp_ok_false_does_not_satisfy_required_tool_call() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "mcp.tool.result",
                "tool_name": "mcp__cortex__context_search",
                "ok": False,
            }
        ],
    }

    tool_calls = parser.extract_tool_calls(raw_response)
    summary = summarize_tool_calls(tool_calls)

    assert summary["mcp_total"] == 1
    assert summary["mcp_successful_total"] == 0
    assert missing_required_tool_call_patterns(tool_calls, [r"^mcp__cortex__"]) == [r"^mcp__cortex__"]


def test_codex_parser_extracts_successful_command_executions_for_requirements() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "mcp-tool list_indexed_repositories '{}'",
                    "aggregated_output": '{"repos":[]}',
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "command_execution",
                    "command": "mcp-tool find_code '{bad json'",
                    "aggregated_output": "error",
                    "exit_code": 1,
                    "status": "failed",
                },
            },
        ],
    }

    executions = parser.extract_command_executions(raw_response)

    assert executions[0]["source"] == "codex.item"
    assert executions[0]["command"] == "mcp-tool list_indexed_repositories '{}'"
    assert missing_required_command_patterns(executions, [r"\bmcp-tool\s+list_indexed_repositories\b"]) == []
    assert missing_required_command_patterns(executions, [r"\bmcp-tool\s+find_code\b"]) == [
        r"\bmcp-tool\s+find_code\b"
    ]


def test_claude_parser_extracts_successful_bash_command_executions_for_requirements() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "stream-json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "mcp-tool list_indexed_repositories '{}'"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "Bash",
                            "input": {"command": "mcp-tool find_code '{bad json'"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": '{"repos":[]}',
                            "is_error": False,
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": "error",
                            "is_error": True,
                        },
                    ]
                },
            },
        ],
    }

    executions = parser.extract_command_executions(raw_response)

    assert executions[0]["source"] == "claude.bash"
    assert executions[0]["command"] == "mcp-tool list_indexed_repositories '{}'"
    assert missing_required_command_patterns(executions, [r"\bmcp-tool\s+list_indexed_repositories\b"]) == []
    assert missing_required_command_patterns(executions, [r"\bmcp-tool\s+find_code\b"]) == [
        r"\bmcp-tool\s+find_code\b"
    ]


def test_codex_parser_does_not_double_prefix_qualified_mcp_tool_names() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "mcp_server": "cortex",
                    "tool_name": "mcp__cortex__context_search",
                    "status": "completed",
                    "result": {},
                },
            }
        ],
    }

    tool_calls = parser.extract_tool_calls(raw_response)

    assert tool_calls[0]["tool_name"] == "mcp__cortex__context_search"

def test_claude_parser_parses_observed_raw_response_fixture(fixtures_root) -> None:
    parser = ClaudeAgentParser()
    raw_response = json.loads((fixtures_root / "claude" / "raw_response.json").read_text(encoding="utf-8"))

    structured = parser.extract_structured_output(raw_response)
    usage = parser.extract_token_usage(raw_response)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert "task_id" not in structured
    assert structured["retrieved_context_files"] == ["a.py"]
    assert usage == {
        "source": "claude.response.usage",
        "input_tokens": 20,
        "output_tokens": 9,
        "total_tokens": 29,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 7,
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    }
    assert tool_calls == [
        {
            "source": "claude.server_tool_use",
            "tool_name": "server_tool_use",
            "payload": {"web_search_requests": 1, "web_fetch_requests": 0},
        }
    ]

def test_codex_parser_ignores_search_output_environment_variables(tmp_path) -> None:
    parser = CodexAgentParser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/zsh -lc 'env | rg \"^PATH=\"'",
                    "aggregated_output": "PATH=/usr/local/bin:/usr/bin:/bin\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ],
    }

    inferred = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace)})

    assert inferred is None

def test_codex_parser_infers_trajectory_from_command_events() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {"type": "thread.started"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/zsh -lc 'rg -n \"fill_value\" sklearn/impute/_iterative.py'",
                    "aggregated_output": "sklearn/impute/_iterative.py:120:    fill_value : str or numerical value, default=None\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "command_execution",
                    "command": "/bin/zsh -lc \"nl -ba sklearn/impute/_iterative.py | sed -n '115,123p'\"",
                    "aggregated_output": "   115→    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \\\n   123→        passed to :class:`~sklearn.impute.SimpleImputer`.\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "file_change",
                    "changes": [
                        {"path": "/tmp/workspace/sklearn/impute/_iterative.py", "kind": "update"},
                    ],
                    "status": "completed",
                },
            },
        ],
    }
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "touched_files": [],
            "retrieval_steps": [],
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
        "raw_response": raw_response,
        "model_patch": "",
    }

    traj = parser.infer_trajectory_data(raw_response, record=record)

    assert traj is not None
    assert traj["pred_files"] == ["sklearn/impute/_iterative.py"]
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"] == [
        {"start": 115, "end": 123},
        {"start": 120, "end": 120},
    ]

def test_claude_parser_infers_trajectory_from_verbose_tool_history() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "grep-1",
                            "name": "Grep",
                            "input": {
                                "pattern": "fill_value",
                                "path": "/tmp/workspace/sklearn/impute/_iterative.py",
                                "output_mode": "content",
                            },
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "grep-1",
                            "content": "120:    fill_value : str or numerical value, default=None\n",
                        }
                    ]
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
                            "input": {"file_path": "/tmp/workspace/sklearn/impute/_iterative.py"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": "   115→    initial_strategy : {'mean', 'median'}\n   123→        passed to :class:`~sklearn.impute.SimpleImputer`.\n",
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "edit-1",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/workspace/sklearn/impute/_iterative.py"},
                        }
                    ]
                },
            },
        ],
    }
    record = {
        "agent": "claude",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "touched_files": [],
            "retrieval_steps": [],
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
        "raw_response": raw_response,
        "model_patch": "",
    }

    traj = parser.infer_trajectory_data(raw_response, record=record)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert traj is not None
    assert traj["pred_files"] == ["sklearn/impute/_iterative.py"]
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 115
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][-1]["end"] == 123
    assert [call["tool_name"] for call in tool_calls] == ["Grep", "Read", "Edit"]
    assert tool_calls[0]["payload"]["result"]["content_chars"] == 58


def test_claude_parser_ignores_directory_scoped_grep_output_as_context() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "grep-1",
                            "name": "Grep",
                            "input": {
                                "pattern": "fill_value",
                                "path": "/tmp/workspace/sklearn",
                                "output_mode": "content",
                            },
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "grep-1",
                            "content": "sklearn/impute/_iterative.py:120: fill_value\n",
                        }
                    ]
                },
            },
        ],
    }

    assert parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"}) is None


def test_claude_parser_ignores_grep_without_matches() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "grep-1",
                            "name": "Grep",
                            "input": {
                                "pattern": "fill_value",
                                "path": "/tmp/workspace/sklearn/impute/_iterative.py",
                                "output_mode": "content",
                            },
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "grep-1",
                            "content": "No matches found\n",
                        }
                    ]
                },
            },
        ],
    }

    assert parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"}) is None


def test_claude_parser_ignores_grep_scoped_to_dotted_directory() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "grep-1",
                            "name": "Grep",
                            "input": {
                                "pattern": "deploy",
                                "path": "/tmp/workspace/.github",
                                "output_mode": "content",
                            },
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "grep-1",
                            "content": ".github/workflows/ci.yml:12: deploy\n",
                        }
                    ]
                },
            },
        ],
    }

    assert parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"}) is None


def test_claude_parser_infers_trajectory_from_generic_mcp_tool_result() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "stream-json",
        "response": [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mcp-1",
                            "name": "mcp__demo__context_search",
                            "input": {"query": "fill_value"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "mcp-1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "matches": [
                                                {"file": "sklearn/impute/_iterative.py", "start": 120, "end": 123},
                                                {"path": "sklearn/impute/_base.py:44"},
                                            ]
                                        }
                                    ),
                                }
                            ],
                        }
                    ]
                },
            },
        ],
    }

    traj = parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"})

    assert traj is not None
    assert traj["pred_files"] == ["sklearn/impute/_base.py", "sklearn/impute/_iterative.py"]
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"] == [{"start": 120, "end": 123}]
    assert traj["pred_spans"]["sklearn/impute/_base.py"] == [{"start": 44, "end": 44}]


def test_claude_parser_does_not_infer_trajectory_from_edit_tool_results() -> None:
    parser = ClaudeAgentParser()
    response = []
    for index, tool_name in enumerate(("Edit", "MultiEdit", "Write", "NotebookEdit", "TodoWrite"), start=1):
        tool_id = f"edit-{index}"
        response.extend(
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": {"file_path": "/tmp/workspace/src/app.py"},
                            }
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": {"file_path": "/tmp/workspace/src/app.py", "line": 3},
                            }
                        ]
                    },
                },
            ]
        )
    raw_response = {"agent": "claude", "response_format": "stream-json", "response": response}

    assert parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"}) is None


def test_codex_parser_infers_trajectory_from_generic_tool_result_event() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "status": "completed",
                    "tool_name": "mcp__demo__context_search",
                    "result": {
                        "matches": [
                            {"file_path": "sklearn/impute/_iterative.py", "line": 120},
                            {"file": "sklearn/impute/_base.py", "start_line": 44, "end_line": 47},
                        ]
                    },
                },
            }
        ],
    }

    traj = parser.infer_trajectory_data(raw_response, record={"workspace_path": "/tmp/workspace"})

    assert traj is not None
    assert traj["pred_files"] == ["sklearn/impute/_base.py", "sklearn/impute/_iterative.py"]
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"] == [{"start": 120, "end": 120}]
    assert traj["pred_spans"]["sklearn/impute/_base.py"] == [{"start": 44, "end": 47}]

def test_convert_run_record_keeps_inferred_codex_trajectory_out_of_empty_final_context() -> None:
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/zsh -lc 'rg -n \"fill_value\" sklearn/impute/_iterative.py'",
                    "aggregated_output": "sklearn/impute/_iterative.py:120:    fill_value : str or numerical value, default=None\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ],
    }
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "touched_files": [],
            "retrieval_steps": [],
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
        "raw_response": raw_response,
        "model_patch": "",
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == []
    assert converted["traj_data"]["pred_spans"] == {}
    assert converted["traj_data"]["pred_steps"][0]["files"] == ["sklearn/impute/_iterative.py"]
    assert converted["traj_data"]["pred_steps"][0]["spans"] == {
        "sklearn/impute/_iterative.py": [{"start": 120, "end": 120}]
    }
