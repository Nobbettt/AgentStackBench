
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

def test_build_comparison_payload_single_variant_mode(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "Single variant export",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 10, "completed_tasks": 6},
                {"variant": "with-superpowers-mounted", "total_tasks": 10, "completed_tasks": 8},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "with-superpowers-mounted"
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "with-superpowers-mounted", "model": "gpt-5.4", "reasoning_effort": "high", "timeout": 2400, "setup": {"copy_paths": []}}}))
    treatment_task_dir = treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    treatment_record = _record(treatment_task_dir, 2000, 1500, 3)
    _write(treatment_dir / "task-results.jsonl", json.dumps({"status": "completed", "record_path": treatment_record}))
    _write(
        treatment_dir / "eval.jsonl",
        json.dumps(
            {
                "final": {
                    "file": {"intersection": 3, "gold_size": 4, "pred_size": 4},
                    "span": {"intersection": 60, "gold_size": 100, "pred_size": 80},
                },
                "editloc": {"intersection": 2, "gold_size": 4, "pred_size": 2},
            }
        ),
    )
    _write(treatment_dir / "resolution-summary.json", json.dumps({"pass_at_1": 0.2, "resolved_count": 2}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-03-23T21:30:54Z",
                "completed_at": "2026-03-24T02:29:10Z",
                "task_set": {"count": 10, "bench_counts": {"Verified": 10}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "name": "with-superpowers-mounted",
                        "effective_config_path": str(treatment_dir / "effective-config.json"),
                        "task_results_path": str(treatment_dir / "task-results.jsonl"),
                        "output_dir": str(treatment_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="with-superpowers-mounted")

    assert payload["comparisonCards"][0]["title"] == "With Superpowers Mounted"
    assert payload["comparisonCards"][0]["completedAt"] == "2026-03-24T02:29:10Z"
    assert len(payload["comparisonCards"][0]["variants"]) == 1
    assert payload["leaderboardRows"][0]["model"] == "gpt-5.4"
    assert payload["leaderboardRows"][0]["suite"] == "With Superpowers Mounted"
