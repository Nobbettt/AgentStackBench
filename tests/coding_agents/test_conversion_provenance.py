# Fork note: Modified by Norbert Laszlo on 2026-04-16 from upstream ContextBench.
# Summary of changes: add regression coverage for record-path resolution and provenance-source attribution.

from __future__ import annotations

import json

import pytest

from contextbench.agents import extract_trajectory
from contextbench.coding_agents import (
    convert_run_record,
    load_predictions_from_path,
    load_predictions_with_summary_from_path,
    parse_unified_diff,
    record_is_convertible,
)
from contextbench.run_suites_core.postprocess import convert_records_to_jsonl

def test_convert_run_record_does_not_use_patch_as_retrieval_context(make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="task-diff",
        model_patch="""diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,0 +10,2 @@
+x
""",
        final_output={
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    )

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == []
    assert converted["traj_data"]["pred_steps"] == []
    assert converted["traj_data"]["pred_files_provenance"] == {}
    assert converted["traj_data"]["pred_files_source"] == []

def test_convert_run_record_trace_only_sources_do_not_claim_agent_report() -> None:
    record = {
        "agent": "codex",
        "instance_id": "task-trace",
        "workspace_path": "/tmp/workspace",
        "raw_response": {
            "agent": "codex",
            "response_format": "jsonl-events",
            "events": [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "/bin/zsh -lc 'rg -n \"request\" requests/api.py'",
                        "aggregated_output": "requests/api.py:3:def request():\n",
                    },
                }
            ],
        },
        "final_output": {
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files_source"] == ["trace_inference"]
    assert converted["traj_data"]["pred_files_provenance"]["requests/api.py"] == "trace_inference"

def test_convert_run_record_marks_span_only_reported_file_provenance(make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="task-span",
        final_output={
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": [],
            "retrieved_context_spans": [{"file": "pkg/mod.py", "start": 10, "end": 20}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    )

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["pkg/mod.py"]
    assert converted["traj_data"]["pred_files_provenance"]["pkg/mod.py"] == "agent_report"
    assert converted["traj_data"]["pred_files_source"] == ["agent_report"]
    assert converted["traj_data"]["pred_spans_source"] == ["agent_report"]

def test_convert_run_record_marks_symbol_only_reported_file_provenance(make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="task-symbol",
        final_output={
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [{"file": "pkg/mod.py", "name": "Handler"}],
            "notes": "",
        },
    )

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["pkg/mod.py"]
    assert converted["traj_data"]["pred_files_provenance"]["pkg/mod.py"] == "agent_report"
    assert converted["traj_data"]["pred_files_source"] == ["agent_report"]
    assert converted["traj_data"]["pred_symbols_source"] == ["agent_report"]
