# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextbench.agents.registry import get_coding_agent_adapter, normalize_coding_agent_name
from contextbench.coding_agents.constants import CLAUDE_OTEL_OUTPUT_SCHEMA_PATH


def test_claude_otel_adapter_is_registered() -> None:
    assert normalize_coding_agent_name("claude-v2") == "claude-otel"
    adapter = get_coding_agent_adapter("claude-otel")
    assert adapter.record_suffix == "claude-otel"
    assert adapter.output_schema_path == CLAUDE_OTEL_OUTPUT_SCHEMA_PATH
    assert adapter.scored_context_source == "otel_tool_results"
    assert adapter.score_inferred_context is True


def test_claude_otel_prompt_is_minimal_and_does_not_ask_for_context_self_report() -> None:
    prompt = get_coding_agent_adapter("claude-otel").build_prompt(
        {
            "repo": "example/repo",
            "prompt": "Fix it.",
        }
    )

    assert "status, final_answer, and notes" in prompt
    assert "retrieved context" not in prompt
    assert "trajectories" not in prompt
    assert "tool logs" not in prompt
    assert "retrieved_context_files" not in prompt
    assert "retrieved_context_spans" not in prompt
    assert "retrieved_context_symbols" not in prompt
