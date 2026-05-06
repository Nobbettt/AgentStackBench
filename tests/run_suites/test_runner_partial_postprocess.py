
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
import contextbench.run_suites_setup as run_suites_setup
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CODEX_RUNTIME_IMAGE,
)
from contextbench.run_suites_core.postprocess import (
    ResolutionCommandError,
    describe_resolution_backend_support,
    evaluate_resolution_for_suite,
    export_resolution_predictions,
    run_resolution_evaluation,
)


from .helpers import _fake_run_coding_agent_task, _make_fake_agent_record, _write_task_inputs

def test_run_suite_runner_marks_partial_postprocess_when_conversion_and_evaluation_cover_subset(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    cleanup_calls: list[tuple[str, str, str]] = []

    def fake_run(
        *,
        task,
        agent,
        output_dir,
        cache_dir,
        schema_path,
        timeout,
        model=None,
        reasoning_effort=None,
        agent_args=(),
        env_overrides=None,
        prompt_preamble=None,
        setup=None,
        workspace_key=None,
        runtime_backend="host",
        runtime_image=None,
        runtime_env=None,
        runtime_setup_commands=(),
        runtime_keep_failed=False,
    ):
        del (
            cache_dir,
            schema_path,
            timeout,
            model,
            reasoning_effort,
            agent_args,
            env_overrides,
            prompt_preamble,
            setup,
            runtime_backend,
            runtime_image,
            runtime_env,
            runtime_setup_commands,
            runtime_keep_failed,
        )
        task_id = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        workspace_path = task_dir / "workspaces" / safe_path_component(workspace_key or task_id)
        workspace_path.mkdir(parents=True, exist_ok=True)
        completed = str(task.get("instance_id") or "").endswith("1000")
        record = _make_fake_agent_record(
            task=task,
            agent=agent,
            task_dir=task_dir,
            workspace_path=workspace_path,
            status="completed" if completed else "failed",
            timeout=not completed,
        )
        record_path = task_dir / f"{task_id}.codex-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", fake_run)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )
    monkeypatch.setattr("contextbench.run_suites_core.runner.treesitter_available", lambda: True)
    evaluation_calls: list[dict[str, object]] = []

    def fake_evaluate_prediction_file(**kwargs):
        evaluation_calls.append(
            {
                "cache_dir": kwargs["cache_dir"],
                "tmp_root": os.environ.get("CONTEXTBENCH_TMP_ROOT"),
                "selected_task_count": kwargs["selected_task_count"],
            }
        )
        return {
            "num_valid": 1,
            "num_total": 1,
            "prediction_count": 1,
            "evaluated_prediction_count": 1,
            "selected_task_count": kwargs["selected_task_count"],
            "coverage_of_attempted_tasks": 1 / kwargs["selected_task_count"],
            "is_partial": True,
            "scope": "converted_predictions",
            "error_counts": {},
        }

    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_prediction_file",
        fake_evaluate_prediction_file,
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "partial-postprocess",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": True, "runtime_backend": "host", "gold_path": str(task_data)},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "partial-postprocess"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_rows = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]
    summary_row = summary_rows[0]

    assert rc == 1
    assert variant["status"] == "completed_with_failures"
    assert variant["metrics"]["conversion"]["selected_task_count"] == 2
    assert variant["metrics"]["conversion"]["prediction_count"] == 1
    assert variant["metrics"]["conversion"]["is_partial"] is True
    assert variant["metrics"]["evaluation"]["selected_task_count"] == 2
    assert variant["metrics"]["evaluation"]["is_partial"] is True
    assert variant["metrics"]["postprocess_partial"] is True
    assert variant["warnings"]
    expected_eval_cache = tmp_path / "cache" / "postprocess-eval" / "partial-postprocess" / "baseline"
    assert evaluation_calls == [
        {
            "cache_dir": expected_eval_cache.resolve(),
            "tmp_root": str(expected_eval_cache.resolve() / "worktrees"),
            "selected_task_count": 2,
        }
    ]
    assert summary_row["postprocess_partial"] is True
    assert summary_row["conversion_is_partial"] is True
    assert summary_row["evaluation_is_partial"] is True


def test_run_suite_runner_marks_partial_postprocess_when_resolution_covers_subset(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: {
            "status": "partial",
            "backend": "mixed",
            "task_count": 1,
            "prediction_count": 0,
            "evaluated_task_count": 0,
            "evaluated_prediction_count": 0,
            "resolved_count": 0,
            "pass_at_1": None,
            "coverage_of_attempted_tasks": 0.0,
            "is_partial": True,
            "scope": "resolution_predictions",
            "supported_benches": ["Verified"],
            "successful_benches": [],
            "failed_benches": ["Verified"],
            "unsupported_benches": [],
            "per_bench": {
                "Verified": {
                    "bench": "Verified",
                    "backend": "swebench",
                    "status": "no_predictions",
                    "task_count": 1,
                    "prediction_count": 0,
                    "resolved_count": 0,
                    "pass_at_1": None,
                    "resolved_ids": [],
                    "unresolved_ids": [],
                    "coverage_of_attempted_tasks": 0.0,
                    "is_partial": True,
                }
            },
        },
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "partial-resolution",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "resolve": True, "runtime_backend": "host"},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "partial-resolution"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert rc == 1
    assert variant["status"] == "postprocess_failed"
    assert variant["metrics"]["resolution"]["is_partial"] is True
    assert variant["metrics"]["postprocess_partial"] is True
    assert variant["warnings"]


def test_run_suite_runner_rejects_partial_postprocess_exit_code(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: {
            "status": "partial",
            "backend": "mixed",
            "task_count": 1,
            "prediction_count": 1,
            "evaluated_task_count": 0,
            "evaluated_prediction_count": 0,
            "resolved_count": 0,
            "pass_at_1": None,
            "coverage_of_attempted_tasks": 1.0,
            "is_partial": True,
            "scope": "resolution_predictions",
            "supported_benches": ["Verified"],
            "successful_benches": [],
            "failed_benches": ["Verified"],
            "unsupported_benches": [],
            "per_bench": {},
        },
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "rejected-partial-resolution",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "resolve": True, "runtime_backend": "host"},
        }
    )

    assert RunSuiteRunner(config).run() == 1


def test_run_suite_runner_marks_postprocess_failed_when_resolution_stage_fully_fails(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: {
            "status": "failed",
            "backend": "mixed",
            "task_count": 1,
            "prediction_count": 1,
            "evaluated_task_count": 0,
            "evaluated_prediction_count": 0,
            "resolved_count": 0,
            "pass_at_1": None,
            "coverage_of_attempted_tasks": 0.0,
            "is_partial": False,
            "scope": "resolution_predictions",
            "supported_benches": ["Verified"],
            "successful_benches": [],
            "failed_benches": ["Verified"],
            "unsupported_benches": [],
            "per_bench": {
                "Verified": {
                    "bench": "Verified",
                    "backend": "swebench",
                    "status": "failed",
                    "task_count": 1,
                    "prediction_count": 1,
                    "resolved_count": 0,
                    "pass_at_1": None,
                    "resolved_ids": [],
                    "unresolved_ids": [],
                    "coverage_of_attempted_tasks": 1.0,
                    "is_partial": False,
                    "error_detail": "backend crashed",
                }
            },
        },
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "failed-resolution-stage",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "resolve": True, "runtime_backend": "host"},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "failed-resolution-stage"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert rc == 1
    assert variant["status"] == "postprocess_failed"
    assert variant["metrics"]["resolution"]["status"] == "failed"
