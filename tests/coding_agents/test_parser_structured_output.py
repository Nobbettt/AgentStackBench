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
from contextbench.parsers.trajectory import parse_trajectory

def test_extract_structured_output_from_nested_value() -> None:
    payload = {
        "result": {
            "content": json.dumps(
                {
                    "status": "completed",
                    "final_answer": "done",
                    "retrieved_context_files": [],
                    "retrieved_context_spans": [],
                    "retrieved_context_symbols": [],
                    "notes": "",
                }
            )
        }
    }

    structured = extract_structured_output_from_value(payload)

    assert structured is not None
    assert structured["status"] == "completed"

def test_extract_structured_output_from_invalid_value_returns_none() -> None:
    assert extract_structured_output_from_value({"result": "not-json"}) is None

def test_build_codex_raw_response_reads_events_and_final_output(tmp_path, make_final_output, output_schema) -> None:
    events_path = tmp_path / "codex-events.jsonl"
    final_output_path = tmp_path / "final-output.json"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_output_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "final_answer": "done",
                "retrieved_context_files": ["a.py"],
                "retrieved_context_spans": [],
                "retrieved_context_symbols": [],
                "notes": "",
            }
        ),
        encoding="utf-8",
    )

    raw_response = build_codex_raw_response(events_path, final_output_path)

    assert raw_response["response_format"] == "jsonl-events"
    assert len(raw_response["events"]) == 2
    jsonschema.validate(raw_response["final_message"], output_schema)

def test_build_claude_raw_response_preserves_verbose_event_array(tmp_path) -> None:
    raw_output_path = tmp_path / "claude-output.json"
    raw_output_path.write_text(
        json.dumps(
            [
                {
                    "type": "system",
                    "subtype": "init",
                    "plugins": [],
                    "mcp_servers": {},
                    "slash_commands": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    raw_response = build_claude_raw_response(raw_output_path)

    assert raw_response["agent"] == "claude"
    assert isinstance(raw_response["response"], list)
    assert raw_response["response"][0]["subtype"] == "init"

def test_observed_fixture_outputs_match_schema(fixtures_root) -> None:
    codex_structured = CodexAgentParser().extract_structured_output(
        json.loads((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"))
    )
    claude_structured = ClaudeAgentParser().extract_structured_output(
        json.loads((fixtures_root / "claude" / "raw_response.json").read_text(encoding="utf-8"))
    )
    codex_schema = json.loads(CODEX_OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    claude_schema = json.loads(CLAUDE_OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    codex_projected = {key: codex_structured[key] for key in codex_schema["properties"]}

    jsonschema.validate(codex_projected, codex_schema)
    jsonschema.validate(claude_structured, claude_schema)

def test_extract_structured_output_accepts_minimal_payload() -> None:
    payload = {
        "status": "completed",
        "final_answer": "done",
        "retrieved_context_files": ["a.py"],
        "retrieved_context_spans": [],
        "retrieved_context_symbols": [],
        "notes": "",
    }

    structured = extract_structured_output_from_value(payload)

    assert structured is not None
    assert structured["final_answer"] == "done"

def test_parser_normalize_record_uses_raw_response_path(tmp_path, fixtures_root) -> None:
    raw_path = tmp_path / "raw-response.json"
    raw_path.write_text((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"), encoding="utf-8")
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "original_inst_id": "task-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "raw_response_path": str(raw_path),
        "final_output": None,
        "token_usage": None,
        "model_patch": "",
    }

    parser = CodexAgentParser()
    normalized = parser.normalize_record(record)

    assert normalized["final_output"]["task_id"] == "task-1"
    assert normalized["token_usage"]["input_tokens"] == 12667
    assert normalized["tool_calls"][0]["tool_name"] == "repo.search"
