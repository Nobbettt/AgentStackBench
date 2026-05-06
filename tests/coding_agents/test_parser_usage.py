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

def test_codex_parser_extracts_usage_from_turn_completed_event() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {"type": "thread.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 4, "output_tokens": 3}},
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 12,
        "output_tokens": 3,
        "cached_input_tokens": 4,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }

def test_codex_parser_returns_none_without_turn_completed_usage() -> None:
    parser = CodexAgentParser()

    usage = parser.extract_token_usage({"agent": "codex", "response_format": "jsonl-events", "events": [{"type": "thread.started"}]})

    assert usage is None

def test_codex_parser_extracts_reasoning_tokens_from_output_token_details() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 30},
                    "total_tokens": 150,
                },
            }
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 20,
        "total_tokens": 150,
        "cache_read_input_tokens": 20,
        "reasoning_tokens": 30,
    }

def test_claude_parser_extracts_usage_from_response_usage() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "result",
                "usage": {
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "output_tokens": 9,
                    "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
                },
            }
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "claude.response.usage",
        "input_tokens": 20,
        "output_tokens": 9,
        "total_tokens": 29,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 7,
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    }

def test_claude_parser_returns_none_when_usage_missing() -> None:
    parser = ClaudeAgentParser()

    usage = parser.extract_token_usage({"agent": "claude", "response_format": "json", "response": [{"type": "result", "result": "{}"}]})

    assert usage is None
