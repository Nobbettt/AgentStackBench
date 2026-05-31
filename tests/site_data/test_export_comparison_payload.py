
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.export_comparison_data as export_comparison_data
from scripts.export_comparison_data import ComparisonExportError, build_comparison_payload

from .helpers import _record, _write


def test_repository_size_line_counts_are_opt_in_and_local_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_cache = tmp_path / "repo-cache"
    repo_dir = repo_cache / "github.com__example__repo"
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True)
    _write(repo_dir / "src" / "a.py", "one\n\nthree\n")
    _write(repo_dir / "README.md", "title\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True, text=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    monkeypatch.setattr(export_comparison_data, "DEFAULT_REPO_CACHE_DIR", repo_cache)
    export_comparison_data._REPOSITORY_SIZE_CACHE.clear()
    task_row = {"repo_url": "https://github.com/example/repo.git", "commit": commit}
    record = {}

    default_payload = export_comparison_data._repository_size_payload(task_row, record)
    assert default_payload == {
        "status": "available",
        "repo": "example/repo",
        "trackedFiles": 2,
    }

    export_comparison_data._REPOSITORY_SIZE_CACHE.clear()
    line_payload = export_comparison_data._repository_size_payload(task_row, record, include_line_counts=True)
    assert line_payload == {
        "status": "available",
        "repo": "example/repo",
        "trackedFiles": 2,
        "lineCountStatus": "available",
        "trackedTextLines": 4,
    }


def test_repository_size_cache_is_commit_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_cache = tmp_path / "repo-cache"
    repo_dir = repo_cache / "github.com__example__repo"
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True)

    _write(repo_dir / "src" / "a.py", "one\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True, text=True)
    first_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    _write(repo_dir / "src" / "b.py", "two\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=repo_dir, check=True, capture_output=True, text=True)
    second_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    monkeypatch.setattr(export_comparison_data, "DEFAULT_REPO_CACHE_DIR", repo_cache)
    export_comparison_data._REPOSITORY_SIZE_CACHE.clear()

    first_payload = export_comparison_data._repository_size_payload(
        {"repo_url": "https://github.com/example/repo.git", "commit": first_commit},
        {},
    )
    second_payload = export_comparison_data._repository_size_payload(
        {"repo_url": "https://github.com/example/repo.git", "commit": second_commit},
        {},
    )

    assert first_payload == {
        "status": "available",
        "repo": "example/repo",
        "trackedFiles": 1,
    }
    assert second_payload == {
        "status": "available",
        "repo": "example/repo",
        "trackedFiles": 2,
    }


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
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["contextLevels"] == {
        "file": {"recall": "0.750", "precision": "0.750", "f1": "0.750"},
        "block": {"recall": "0.600", "precision": "0.750", "f1": "0.667"},
        "line": {"recall": "0.750", "precision": "0.500", "f1": "0.600"},
        "symbol": {"recall": "0.500", "precision": "0.500", "f1": "0.500"},
    }
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


def test_build_comparison_payload_can_export_aligned_postprocess_artifacts(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "Aligned comparison",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(suite_dir / "summary.json", json.dumps([{"variant": "baseline", "total_tasks": 1}]))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 1},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    }
                ],
            }
        ),
    )
    _write(
        variant_dir / "effective-config.json",
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
    _write(variant_dir / "task-results.jsonl", "")
    _write(
        variant_dir / "eval.jsonl",
        json.dumps(
            {
                "final": {
                    "file": {"intersection": 1, "gold_size": 10, "pred_size": 100},
                    "symbol": {"intersection": 0, "gold_size": 1, "pred_size": 1},
                    "span": {"intersection": 0, "gold_size": 1, "pred_size": 1},
                    "line": {"intersection": 0, "gold_size": 1, "pred_size": 1},
                },
                "trajectory": {"auc_coverage": {}, "redundancy": {}},
            }
        ),
    )
    _write(
        variant_dir / "eval.aligned.jsonl",
        json.dumps(
            {
                "final": {
                    "file": {"intersection": 5, "gold_size": 10, "pred_size": 10},
                    "symbol": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                    "span": {"intersection": 3, "gold_size": 6, "pred_size": 6},
                    "line": {"intersection": 4, "gold_size": 8, "pred_size": 8},
                },
                "trajectory": {
                    "auc_coverage": {"file": 0.5, "symbol": 0.5, "span": 0.5},
                    "redundancy": {"file": 0.1, "symbol": 0.1, "span": 0.1},
                },
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="baseline", artifact_suffix="aligned")

    variant = payload["comparisonCards"][0]["variants"][0]
    assert variant["results"]["quality"]["fileF1"] == "0.500"
    assert variant["results"]["quality"]["contextF1"] == "0.500"
    assert variant["results"]["quality"]["contextLevels"] == {
        "file": {"recall": "0.500", "precision": "0.500", "f1": "0.500"},
        "block": {"recall": "0.500", "precision": "0.500", "f1": "0.500"},
        "line": {"recall": "0.500", "precision": "0.500", "f1": "0.500"},
        "symbol": {"recall": "0.500", "precision": "0.500", "f1": "0.500"},
    }
    assert any("aligned postprocess artifacts" in note for note in payload["comparisonCards"][0]["notes"])
