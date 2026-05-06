
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

def test_build_comparison_payload_keeps_unavailable_pass_at_1_null(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    record_path = _record(variant_dir / "agent_runs" / "codex" / "Verified" / "task-a", 1000, 1200, 2)
    _write(variant_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": record_path}))
    _write(variant_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(variant_dir / "resolution-summary.json", json.dumps({"status": "failed", "pass_at_1": None, "resolved_count": 0}))
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
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline")
    outcome = payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]
    integrity = payload["comparisonCards"][0]["variants"][0]["results"]["integrity"]

    assert outcome["officialPassAt1"] is None
    assert outcome["officialPassAt1Status"] == "failed"
    assert integrity["resolutionStatus"] == "failed"
    assert payload["leaderboardRows"][0]["officialPassAt1"] is None
def test_build_comparison_payload_exports_resolution_error_ids(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    record_path = _record(variant_dir / "agent_runs" / "codex" / "Verified" / "task-a", 1000, 1200, 2)
    _write(variant_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": record_path}))
    _write(variant_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(
        variant_dir / "resolution-summary.json",
        json.dumps({"status": "completed", "pass_at_1": 0.0, "resolved_count": 0, "error_ids": ["task-a"]}),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline")

    assert payload["comparisonCards"][0]["variants"][0]["instances"][0]["artifacts"]["resolutionStatus"] == "error"
def test_build_comparison_payload_exports_unknown_resolution_ids_as_errors(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    record_path = _record(variant_dir / "agent_runs" / "codex" / "Verified" / "task-a", 1000, 1200, 2)
    _write(variant_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": record_path}))
    _write(variant_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(
        variant_dir / "resolution-summary.json",
        json.dumps({"status": "failed", "pass_at_1": None, "resolved_count": 0, "unknown_ids": ["task-a"]}),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline")

    assert payload["comparisonCards"][0]["variants"][0]["instances"][0]["artifacts"]["resolutionStatus"] == "error"
def test_build_comparison_payload_keeps_resolution_error_when_aggregate_conflicts(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    record_path = _record(variant_dir / "agent_runs" / "codex" / "Verified" / "task-a", 1000, 1200, 2)
    _write(variant_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": record_path}))
    _write(variant_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(
        variant_dir / "resolution-summary.json",
        json.dumps(
            {
                "status": "failed",
                "pass_at_1": None,
                "resolved_count": 1,
                "resolved_ids": ["task-a"],
                "per_bench": {"Verified": {"error_ids": ["task-a"]}},
            }
        ),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline")

    assert payload["comparisonCards"][0]["variants"][0]["instances"][0]["artifacts"]["resolutionStatus"] == "error"
def test_build_comparison_payload_nulls_partial_resolution_pass_at_1(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {
                    "variant": "baseline",
                    "total_tasks": 2,
                    "completed_tasks": 2,
                    "resolution_is_partial": True,
                }
            ]
        ),
    )
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    record_path = _record(variant_dir / "agent_runs" / "codex" / "Verified" / "task-a", 1000, 1200, 2)
    _write(variant_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": record_path}))
    _write(variant_dir / "eval.jsonl", json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}}))
    _write(
        variant_dir / "resolution-summary.json",
        json.dumps({"status": "partial", "is_partial": True, "pass_at_1": 1.0, "pass_at_1_on_evaluated": 1.0, "resolved_count": 1}),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 2, "bench_counts": {"Verified": 2}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline")
    outcome = payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]

    assert outcome["officialPassAt1"] is None
    assert outcome["officialPassAt1OnEvaluated"] == "100.0%"
