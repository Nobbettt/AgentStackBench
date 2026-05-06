# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: cover safe repo-root path inference, trace guards, and effective file normalization for coding-agent parsers.

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

    assert structured["task_id"] == "task-1"
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

def test_claude_parser_parses_observed_raw_response_fixture(fixtures_root) -> None:
    parser = ClaudeAgentParser()
    raw_response = json.loads((fixtures_root / "claude" / "raw_response.json").read_text(encoding="utf-8"))

    structured = parser.extract_structured_output(raw_response)
    usage = parser.extract_token_usage(raw_response)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert structured["task_id"] == "task-1"
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

def test_codex_parser_propagates_trace_inference_meta(tmp_path) -> None:
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

    assert inferred is not None
    assert inferred["trace_inference_meta"]["dropped_env_var_lines"] >= 1

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
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 120
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][-1]["end"] == 123

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

    assert traj is not None
    assert traj["pred_files"] == ["sklearn/impute/_iterative.py"]
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 115
    assert traj["pred_spans"]["sklearn/impute/_iterative.py"][-1]["end"] == 123

def test_convert_run_record_uses_inferred_codex_trajectory_when_schema_retrieval_empty() -> None:
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

    assert converted["traj_data"]["pred_files"] == ["sklearn/impute/_iterative.py"]
    assert converted["traj_data"]["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 120
