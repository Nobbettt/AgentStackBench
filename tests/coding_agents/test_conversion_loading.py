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

def test_load_predictions_from_directory_filters_agent(tmp_path, make_final_output, make_record) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(
                    make_record(
                        agent="codex",
                        instance_id="task-codex",
                        final_output=make_final_output(
                            task_id="task-codex",
                            touched_files=["a.py"],
                            retrieved_context_files=["a.py"],
                        ),
                    )
                ),
                json.dumps(
                    make_record(
                        agent="claude",
                        instance_id="task-claude",
                        final_output=make_final_output(
                            task_id="task-claude",
                            touched_files=["b.py"],
                            retrieved_context_files=["b.py"],
                        ),
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    nested = tmp_path / "codex" / "Verified" / "task-codex"
    nested.mkdir(parents=True)
    (nested / "task-codex.codex-record.json").write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(
                    task_id="task-codex",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(tmp_path, expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"

def test_load_predictions_from_directory_prefers_task_results_record_paths(tmp_path, make_final_output, make_record) -> None:
    variant_dir = tmp_path / "baseline"
    raw_root = variant_dir / "agent_runs" / "codex" / "Verified" / "task-codex"
    raw_root.mkdir(parents=True)
    record_path = raw_root / "task-codex.codex-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-codex",
                "bench": "Verified",
                "status": "completed",
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(variant_dir / "agent_runs" / "codex", expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"

def test_load_predictions_from_directory_resolves_relative_task_results_record_paths(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    variant_dir = tmp_path / "baseline"
    source_dir = variant_dir / "agent_runs" / "codex"
    source_dir.mkdir(parents=True)
    record_path = variant_dir / "archived-records" / "task-codex.codex-record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-codex",
                "bench": "Verified",
                "status": "completed",
                "record_path": "archived-records/task-codex.codex-record.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(source_dir, expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"

def test_load_predictions_from_nested_agent_dir_finds_variant_task_results(tmp_path, make_final_output, make_record) -> None:
    variant_dir = tmp_path / "baseline"
    raw_root = variant_dir / "agent_runs" / "codex" / "Verified" / "task-codex"
    raw_root.mkdir(parents=True)
    record_path = raw_root / "task-codex.codex-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-codex",
                "bench": "Verified",
                "status": "completed",
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(variant_dir / "agent_runs" / "codex", expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"

def test_load_predictions_from_directory_falls_back_to_record_scan_when_task_results_missing(tmp_path, make_final_output, make_record) -> None:
    raw_root = tmp_path / "codex" / "Verified" / "task-codex"
    raw_root.mkdir(parents=True)
    record_path = raw_root / "task-codex.codex-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(tmp_path, expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"

def test_load_predictions_from_json_list_supports_claude_code_alias(tmp_path, make_final_output, make_record) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                make_record(
                    agent="codex",
                    instance_id="task-codex",
                    final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
                ),
                make_record(
                    agent="claude",
                    instance_id="task-claude",
                    final_output=make_final_output(task_id="task-claude", retrieved_context_files=["b.py"]),
                ),
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(records_path, expected_agent="claude-code")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-claude"

def test_load_predictions_from_aggregate_jsonl_resolves_relative_sidecars(
    tmp_path,
    monkeypatch,
    make_final_output,
    make_record,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    (cwd / "raw-response.json").write_text(
        json.dumps({"agent": "codex", "response_format": "jsonl-events", "events": []}),
        encoding="utf-8",
    )
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    raw_response_path = aggregate_dir / "raw-response.json"
    raw_response_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "response_format": "jsonl-events",
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "/bin/zsh -lc 'sed -n 1,5p pkg/mod.py'",
                            "aggregated_output": "line 1\nline 2\n",
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    record = make_record(
        agent="codex",
        instance_id="task-codex",
        final_output=make_final_output(task_id="task-codex"),
    )
    record["workspace_path"] = str(tmp_path / "workspace")
    record["raw_response_path"] = "raw-response.json"
    records_path = aggregate_dir / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    predictions, summary = load_predictions_with_summary_from_path(records_path, expected_agent="codex")

    assert summary["conversion_error_count"] == 0
    assert len(predictions) == 1
    assert predictions[0]["traj_data"]["pred_files"] == ["pkg/mod.py"]
    assert predictions[0]["traj_data"]["pred_files_source"] == ["trace_inference"]


def test_load_predictions_from_aggregate_jsonl_flags_missing_relative_sidecar(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    record = make_record(
        agent="codex",
        instance_id="task-codex",
        final_output=make_final_output(task_id="task-codex"),
    )
    record["raw_response_path"] = "missing-raw-response.json"
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    predictions, summary = load_predictions_with_summary_from_path(records_path, expected_agent="codex")

    assert predictions == []
    assert summary["prediction_count"] == 0
    assert summary["conversion_error_count"] == 1
    assert summary["conversion_errors"][0]["error"] == "missing_artifact_path"
    assert summary["conversion_errors"][0]["artifact_paths"] == [
        {"field": "raw_response_path", "path": "missing-raw-response.json"}
    ]

def test_record_is_convertible_accepts_claude_code_alias(make_final_output, make_record) -> None:
    record = make_record(
        agent="claude",
        instance_id="task-claude",
        final_output=make_final_output(task_id="task-claude"),
    )

    assert record_is_convertible(record, expected_agent="claude-code") is True
