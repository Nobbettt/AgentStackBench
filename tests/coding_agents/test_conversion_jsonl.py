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

def test_convert_records_to_jsonl_resolves_relative_task_results_record_paths(
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
    out_path = variant_dir / "pred.jsonl"

    summary = convert_records_to_jsonl(
        source_dir=source_dir,
        expected_agent="codex",
        out_path=out_path,
    )

    assert summary["prediction_count"] == 1
    assert summary["selected_task_count"] == 1
    assert summary["is_partial"] is False
    assert out_path.exists()
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["instance_id"] == "task-codex"


def test_convert_records_to_jsonl_flags_missing_task_result_sidecar(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    variant_dir = tmp_path / "baseline"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-codex"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-codex.codex-record.json"
    record = make_record(
        agent="codex",
        instance_id="task-codex",
        final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
    )
    record["raw_response_path"] = "missing-raw-response.json"
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
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
    out_path = variant_dir / "pred.jsonl"

    predictions, load_summary = load_predictions_with_summary_from_path(source_dir, expected_agent="codex")
    assert predictions == []
    assert load_summary["conversion_error_count"] == 1
    assert load_summary["conversion_errors"][0]["error"] == "missing_artifact_path"

    summary = convert_records_to_jsonl(
        source_dir=source_dir,
        expected_agent="codex",
        out_path=out_path,
    )

    assert summary["prediction_count"] == 0
    assert summary["conversion_error_count"] == 1
    assert summary["input_error_count"] == 1
    assert summary["is_partial"] is True
    assert summary["conversion_errors"] == [
        {
            "instance_id": "task-codex",
            "record_path": str(record_path),
            "error": "missing_artifact_path",
            "artifact_paths": [{"field": "raw_response_path", "path": "missing-raw-response.json"}],
        }
    ]
    assert out_path.read_text(encoding="utf-8") == ""


def test_load_predictions_with_summary_from_path_reports_partial_conversion(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    variant_dir = tmp_path / "baseline"
    source_dir = variant_dir / "agent_runs" / "codex"
    ok_dir = source_dir / "Verified" / "task-ok"
    fail_dir = source_dir / "Verified" / "task-fail"
    ok_dir.mkdir(parents=True)
    fail_dir.mkdir(parents=True)
    ok_record_path = ok_dir / "task-ok.codex-record.json"
    fail_record_path = fail_dir / "task-fail.codex-record.json"
    ok_record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-ok",
                final_output=make_final_output(task_id="task-ok", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    fail_record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-fail",
                "status": "failed",
                "final_output": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "task-ok", "record_path": str(ok_record_path)}),
                json.dumps({"instance_id": "task-fail", "record_path": str(fail_record_path)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predictions, summary = load_predictions_with_summary_from_path(source_dir, expected_agent="codex")

    assert [row["instance_id"] for row in predictions] == ["task-ok"]
    assert summary["selected_task_count"] == 2
    assert summary["record_count"] == 2
    assert summary["convertible_record_count"] == 1
    assert summary["nonconvertible_record_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["missing_prediction_count"] == 1
    assert summary["coverage_of_attempted_tasks"] == pytest.approx(0.5)
    assert summary["is_partial"] is True

def test_unified_extractor_dispatch_supports_codex_record_file(tmp_path, make_final_output, make_record) -> None:
    record_path = tmp_path / "task.codex-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                final_output=make_final_output(
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extracted = extract_trajectory(str(record_path))

    assert extracted["pred_files"] == ["a.py"]

def test_unified_extractor_dispatch_supports_claude_record_file(tmp_path, make_final_output, make_record) -> None:
    record_path = tmp_path / "task.claude-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="claude",
                final_output=make_final_output(
                    touched_files=["b.py"],
                    retrieved_context_files=["b.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extracted = extract_trajectory(str(record_path))

    assert extracted["pred_files"] == ["b.py"]

def test_load_predictions_from_missing_path_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictions_from_path(tmp_path / "missing-records.json")
