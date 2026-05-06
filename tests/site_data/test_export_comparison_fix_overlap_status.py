
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

def test_build_comparison_payload_computes_variant_fix_overlap_before_sanitizing_patches(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
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
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "treatment", "model": "gpt-5.4", "setup": {}}}))

    baseline_patch = """--- /Users/alice/repo/a.py
+++ /Users/alice/repo/a.py
@@ -1 +1 @@
-x
+y
"""
    treatment_patch = """--- /Users/bob/repo/a.py
+++ /Users/bob/repo/a.py
@@ -1 +1 @@
-x
+y
"""
    baseline_record = _record(
        baseline_dir / "agent_runs" / "codex" / "Verified" / "task-a",
        1000,
        1200,
        2,
        model_patch=baseline_patch,
    )
    treatment_record = _record(
        treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a",
        1000,
        1200,
        2,
        model_patch=treatment_patch,
    )
    _write(baseline_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": baseline_record}))
    _write(treatment_dir / "task-results.jsonl", json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": treatment_record}))
    eval_row = json.dumps({"final": {"file": {"intersection": 1, "gold_size": 1, "pred_size": 1}}})
    _write(baseline_dir / "eval.jsonl", eval_row)
    _write(treatment_dir / "eval.jsonl", eval_row)
    _write(baseline_dir / "resolution-summary.json", json.dumps({"status": "completed", "pass_at_1": 0.0, "resolved_count": 0}))
    _write(treatment_dir / "resolution-summary.json", json.dumps({"status": "completed", "pass_at_1": 0.0, "resolved_count": 0}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
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

    payload, detail_payloads = build_comparison_export(suite_dir)

    assert payload["comparisonCards"][0]["fixOverlapBetweenVariants"]["f1"] == "0.0%"
    assert "_rawModelPatch" not in json.dumps(detail_payloads)
