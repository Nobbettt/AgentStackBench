
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

def test_build_comparison_payload_uses_record_status_for_resumed_skips(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "Resumed suite export",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 2, "completed_tasks": 2},
                {"variant": "treatment", "total_tasks": 2, "completed_tasks": 2},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
    for variant_dir, name in ((baseline_dir, "baseline"), (treatment_dir, "treatment")):
        _write(
            variant_dir / "effective-config.json",
            json.dumps(
                {
                    "effective_config": {
                        "name": name,
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                        "timeout": 2400,
                        "setup": {"copy_paths": []},
                    }
                }
            ),
        )

    baseline_task_dir = baseline_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    treatment_task_dir = treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    baseline_record = _record(baseline_task_dir, 1000, 1200, 2)
    treatment_record = _record(treatment_task_dir, 1000, 1300, 3)

    _write(
        baseline_dir / "task-results.jsonl",
        "\n".join(
            [
                json.dumps({"status": "skipped", "record_path": baseline_record}),
                json.dumps({"status": "skipped", "record_path": baseline_record}),
            ]
        ),
    )
    _write(
        treatment_dir / "task-results.jsonl",
        "\n".join(
            [
                json.dumps({"status": "skipped", "record_path": treatment_record}),
                json.dumps({"status": "skipped", "record_path": treatment_record}),
            ]
        ),
    )

    eval_row = json.dumps(
        {
            "final": {
                "file": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                "symbol": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                "span": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                "line": {"intersection": 1, "gold_size": 2, "pred_size": 2},
            },
            "trajectory": {
                "auc_coverage": {"file": 0.5, "symbol": 0.5, "span": 0.5},
                "redundancy": {"file": 0.1, "symbol": 0.1, "span": 0.1},
            },
        }
    )
    _write(baseline_dir / "eval.jsonl", eval_row)
    _write(treatment_dir / "eval.jsonl", eval_row)
    _write(baseline_dir / "resolution-summary.json", json.dumps({"pass_at_1": 1.0, "resolved_count": 2}))
    _write(treatment_dir / "resolution-summary.json", json.dumps({"pass_at_1": 1.0, "resolved_count": 2}))

    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-04-14T20:25:07Z",
                "completed_at": "2026-04-14T21:45:05Z",
                "task_set": {"count": 2, "bench_counts": {"Verified": 2}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "name": "treatment",
                        "effective_config_path": str(treatment_dir / "effective-config.json"),
                        "task_results_path": str(treatment_dir / "task-results.jsonl"),
                        "output_dir": str(treatment_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir)

    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["completedRuns"] == 2
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["completedRunRate"] == "100.0%"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["officialPassAt1"] == "100.0%"
    assert payload["comparisonCards"][0]["variants"][1]["results"]["outcome"]["completedRuns"] == 2
    assert payload["comparisonCards"][0]["variants"][1]["results"]["outcome"]["completedRunRate"] == "100.0%"
    assert payload["comparisonCards"][0]["variants"][1]["results"]["outcome"]["officialPassAt1"] == "100.0%"
def test_build_comparison_payload_treats_completed_ok_false_as_failure(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "ok false export",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))

    baseline_dir = suite_dir / "variants" / "baseline"
    _write(
        baseline_dir / "effective-config.json",
        json.dumps(
            {
                "effective_config": {
                    "name": "baseline",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "timeout": 2400,
                    "setup": {"copy_paths": []},
                }
            }
        ),
    )
    record = _record(
        baseline_dir / "agent_runs" / "codex" / "Verified" / "task-a",
        1000,
        1200,
        2,
        status="completed",
        ok=False,
    )
    _write(
        baseline_dir / "task-results.jsonl",
        json.dumps(
            {
                "instance_id": "task-a",
                "original_inst_id": "task-a",
                "bench": "Verified",
                "status": "skipped",
                "record_path": record,
            }
        )
        + "\n",
    )
    _write(
        baseline_dir / "eval.jsonl",
        json.dumps(
            {
                "instance_id": "task-a",
                "final": {
                    "file": {"intersection": 0, "gold_size": 1, "pred_size": 0},
                    "symbol": {"intersection": 0, "gold_size": 1, "pred_size": 0},
                    "span": {"intersection": 0, "gold_size": 1, "pred_size": 0},
                    "line": {"intersection": 0, "gold_size": 1, "pred_size": 0},
                }
            }
        )
        + "\n",
    )
    _write(baseline_dir / "resolution-summary.json", json.dumps({"pass_at_1": 0.0, "resolved_count": 0}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-04-14T20:25:07Z",
                "completed_at": "2026-04-14T21:45:05Z",
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    }
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir)
    variant = payload["comparisonCards"][0]["variants"][0]

    assert variant["results"]["outcome"]["completedRuns"] == 0
    assert variant["results"]["outcome"]["failures"] == 1
    assert variant["instances"][0]["outcome"]["status"] == "failed"
