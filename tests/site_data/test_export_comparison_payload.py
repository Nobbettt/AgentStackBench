
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_comparison_data import ComparisonExportError, build_comparison_payload

from .helpers import _record, _write

def test_build_comparison_payload_happy_path(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "A/B comparison",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
                "postprocess": {"gold_path": str(suite_dir / "gold.json")},
            }
        ),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {
                    "variant": "baseline",
                    "total_tasks": 10,
                    "completed_tasks": 6,
                    "postprocess_partial": True,
                    "conversion_is_partial": True,
                    "evaluation_is_partial": True,
                    "warnings": "Conversion covered a subset of selected tasks (1/10).",
                },
                {"variant": "treatment", "total_tasks": 10, "completed_tasks": 8},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
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
    _write(
        treatment_dir / "effective-config.json",
        json.dumps(
            {
                "effective_config": {
                    "name": "with-superpowers-mounted",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "timeout": 2400,
                    "setup": {
                        "copy_paths": [{"source": "agent-resources/superpowers"}],
                        "prompt_preamble": "Use the mounted superpowers resources when they are relevant to solving the task.",
                    },
                }
            }
        ),
    )

    baseline_task_dir = baseline_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    baseline_partial_task_dir = baseline_dir / "agent_runs" / "codex" / "Verified" / "task-b"
    treatment_task_dir = treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    gold_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2,2 +2,2 @@
-old_b
-old_c
+new_b
+new_c
"""
    baseline_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2 +2 @@
-old_b
+new_b
"""
    treatment_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2,2 +2,2 @@
-old_b
-old_c
+new_b
+new_c
"""
    _write(
        suite_dir / "gold.json",
        json.dumps(
            [
                {
                    "inst_id": "task-a",
                    "original_inst_id": "task-a",
                    "repo_url": "https://github.com/example/repo.git",
                    "commit": "abc123",
                    "gold_ctx": [{"file": "src/a.py", "start_line": 1, "end_line": 4}],
                    "patch": gold_patch,
                }
            ]
        ),
    )
    baseline_record = _record(baseline_task_dir, 1000, 1200, 2, model_patch=baseline_patch)
    baseline_partial_record = _record(baseline_partial_task_dir, 1000, 1200, 2, status="partial")
    treatment_record = _record(treatment_task_dir, 2000, 1500, 3, model_patch=treatment_patch)

    _write(
        baseline_dir / "task-results.jsonl",
            "\n".join(
                [
                    json.dumps({"instance_id": "task-a", "status": "completed", "record_path": baseline_record}),
                    json.dumps({"instance_id": "task-b", "status": "partial", "record_path": baseline_partial_record}),
                ]
            ),
        )
    _write(
        treatment_dir / "task-results.jsonl",
        "\n".join(
            [
                json.dumps({"instance_id": "task-a", "status": "completed", "record_path": treatment_record}),
                json.dumps({"instance_id": "task-b", "status": "completed", "record_path": treatment_record}),
            ]
        ),
    )

    eval_row = json.dumps(
        {
            "final": {
                "file": {"intersection": 3, "gold_size": 4, "pred_size": 4},
                "symbol": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                "span": {"intersection": 60, "gold_size": 100, "pred_size": 80},
                "line": {"intersection": 3, "gold_size": 4, "pred_size": 6},
            },
            "editloc": {"intersection": 2, "gold_size": 4, "pred_size": 2},
            "trajectory": {
                "auc_coverage": {"file": 0.8, "symbol": 0.5, "span": 0.6},
                "redundancy": {"file": 0.2, "symbol": 0.1, "span": 0.3},
            },
        }
    )
    _write(baseline_dir / "eval.jsonl", eval_row)
    _write(treatment_dir / "eval.jsonl", eval_row)
    _write(
        baseline_dir / "resolution-summary.json",
        json.dumps({"pass_at_1": 0.1, "resolved_count": 1, "resolved_ids": ["task-a"], "unresolved_ids": ["task-b"]}),
    )
    _write(
        treatment_dir / "resolution-summary.json",
        json.dumps({"pass_at_1": 0.2, "resolved_count": 2, "resolved_ids": ["task-a", "task-b"], "unresolved_ids": []}),
    )

    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-03-23T21:30:54Z",
                "completed_at": "2026-03-24T02:29:10Z",
                "task_set": {
                    "count": 10,
                    "source_count": 1136,
                    "selection_kind": "representative_subset",
                    "bench_counts": {"Verified": 8, "Poly": 2},
                },
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

    assert payload["filterOrder"] == ["all", "codex"]
    assert payload["comparisonCards"][0]["title"] == "Baseline vs With Superpowers Mounted"
    assert payload["comparisonCards"][0]["startedAt"] == "2026-03-23T21:30:54Z"
    assert payload["comparisonCards"][0]["completedAt"] == "2026-03-24T02:29:10Z"
    assert payload["comparisonCards"][0]["taskSet"]["count"] == 10
    assert payload["comparisonCards"][0]["taskSet"]["benchCounts"] == {"Verified": 8, "Poly": 2}
    assert payload["comparisonCards"][0]["taskSet"]["sourceDatasetCount"] == 1136
    assert payload["comparisonCards"][0]["taskSet"]["selectionKind"] == "representative_subset"
    assert payload["comparisonCards"][0]["variants"][1]["parameters"][3]["value"] == "Superpowers snapshot"
    assert payload["comparisonCards"][0]["variants"][1]["parameters"][4] == {
        "label": "Additional Prompt",
        "value": "Use the mounted superpowers resources when they are relevant to solving the task.",
    }
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["spanF1"] == "0.667"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["avgLineF1"] == "0.600"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["fixOverlapVsGold"] == {
        "status": "available",
        "recall": "50.0%",
        "precision": "100.0%",
        "f1": "66.7%",
        "availableInstances": 1,
        "unavailableInstances": 1,
    }
    assert payload["comparisonCards"][0]["variants"][1]["results"]["quality"]["fixOverlapVsGold"]["f1"] == "100.0%"
    assert payload["comparisonCards"][0]["fixOverlapBetweenVariants"] == {
        "status": "available",
        "leftLabel": "A",
        "rightLabel": "B",
        "leftCoveredByRight": "100.0%",
        "rightCoveredByLeft": "50.0%",
        "f1": "66.7%",
        "availableInstances": 1,
        "unavailableInstances": 1,
    }
    assert payload["comparisonCards"][0]["variants"][0]["results"]["efficiency"]["efficiency"] == "0.633"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["completedRuns"] == 1
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["completedRunRate"] == "10.0%"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["officialPassAt1"] == "10.0%"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["outcome"]["comparableToOfficialLeaderboard"] is False
    assert payload["comparisonCards"][0]["variants"][0]["results"]["integrity"]["resolvedTasks"] == 1
    assert payload["comparisonCards"][0]["variants"][0]["results"]["integrity"]["postprocessPartial"] is True
    assert payload["comparisonCards"][0]["variants"][0]["instances"][0]["artifacts"]["resolutionStatus"] == "resolved"
    assert payload["comparisonCards"][0]["variants"][0]["instances"][1]["artifacts"]["resolutionStatus"] == "unresolved"
    assert payload["comparisonCards"][0]["variants"][0]["notes"]
    assert any("partial conversion, evaluation coverage" in note for note in payload["comparisonCards"][0]["notes"])
    assert payload["leaderboardRows"][0]["model"] == "gpt-5.4"
    assert payload["leaderboardRows"][0]["suite"] == "Baseline"
    assert payload["leaderboardRows"][0]["completedRunRate"] == "10.0%"
    assert payload["leaderboardRows"][0]["officialPassAt1"] == "10.0%"
    assert payload["leaderboardRows"][0]["passAt1"] == "10.0%"
    assert payload["leaderboardRows"][0]["contextF1"] == "0.639"
