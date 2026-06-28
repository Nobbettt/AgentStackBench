# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_comparison_data import build_comparison_payload

from .helpers import _write


def test_build_comparison_payload_exports_verification_quality(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    task_dir = variant_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    record_path = task_dir / "task-a.codex-record.json"
    model_patch = (
        "diff --git a/tests/test_app.py b/tests/test_app.py\n"
        "--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n"
        "@@\n"
        "+def test_app():\n"
        "+    assert True\n"
    )

    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]))
    _write(variant_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.5", "setup": {}}}))
    _write(
        record_path,
        json.dumps(
            {
                "agent": "codex",
                "status": "completed",
                "ok": True,
                "duration_ms": 1000,
                "token_usage": {"total_tokens": 100},
                "tool_calls": [],
                "command_executions": [
                    {"command": "python -m py_compile src/app.py", "payload": {"exit_code": 0}}
                ],
                "final_output": {
                    "status": "completed",
                    "final_answer": "Could not run the tests because a missing dependency was not installed.",
                    "retrieved_context_files": [],
                    "retrieved_context_spans": [],
                    "retrieved_context_symbols": [],
                    "notes": "",
                },
                "model_patch": model_patch,
            }
        ),
    )
    _write(
        variant_dir / "task-results.jsonl",
        json.dumps({"instance_id": "task-a", "bench": "Verified", "record_path": str(record_path)}) + "\n",
    )
    _write(
        variant_dir / "pred.jsonl",
        json.dumps({"instance_id": "task-a", "traj_data": {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}}}) + "\n",
    )
    _write(variant_dir / "eval.jsonl", json.dumps({"instance_id": "task-a", "final": {"file": {"intersection": 0, "gold_size": 0, "pred_size": 0}}}) + "\n")
    _write(variant_dir / "resolution-summary.json", json.dumps({"status": "completed", "pass_at_1": 0.0, "resolved_count": 0, "unresolved_ids": ["task-a"]}))
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
    variant = payload["comparisonCards"][0]["variants"][0]
    artifacts = variant["instances"][0]["artifacts"]

    assert artifacts["verificationQuality"]["strongestVerification"] == "syntax_or_static"
    assert artifacts["verificationQuality"]["syntaxOnly"] is True
    assert artifacts["verificationQuality"]["environmentLimited"] is True
    assert artifacts["regressionTest"]["addedRegressionTest"] is True
    assert artifacts["regressionTest"]["regressionTestsRun"] is False
    assert variant["results"]["verification"]["syntaxOnlyRuns"] == 1
    assert variant["results"]["verification"]["addedRegressionTestNotRunRuns"] == 1
