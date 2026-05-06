
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import SanitizationContext
from scripts.export_comparison_data import (
    ComparisonExportError,
    _extract_trace_entries,
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
