
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_comparison_data import ComparisonExportError, build_comparison_payload

from .helpers import _record, _write

def test_build_comparison_payload_fails_when_eval_is_missing(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 1, "completed_tasks": 1},
                {"variant": "treatment", "total_tasks": 1, "completed_tasks": 1},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "treatment", "setup": {}}}))
    _write(baseline_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(treatment_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-03-23T21:30:54Z",
                "completed_at": "2026-03-24T02:29:10Z",
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
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

    with pytest.raises(ComparisonExportError, match="Missing eval.jsonl"):
        build_comparison_payload(suite_dir)

def test_build_comparison_payload_exports_null_pass_at_1_when_resolution_summary_is_missing(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 1, "completed_tasks": 1},
                {"variant": "treatment", "total_tasks": 1, "completed_tasks": 1},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "treatment", "setup": {}}}))
    _write(baseline_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(treatment_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(baseline_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(treatment_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-03-23T21:30:54Z",
                "completed_at": "2026-03-24T02:29:10Z",
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
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

    baseline = payload["comparisonCards"][0]["variants"][0]
    treatment = payload["comparisonCards"][0]["variants"][1]
    assert baseline["results"]["outcome"]["officialPassAt1"] is None
    assert baseline["results"]["outcome"]["officialPassAt1Status"] == "missing"
    assert baseline["results"]["integrity"]["resolutionStatus"] == "missing"
    assert treatment["results"]["outcome"]["officialPassAt1"] is None
    assert treatment["results"]["outcome"]["officialPassAt1Status"] == "missing"
