
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

def test_run_suite_runner_resume_skips_completed_tasks(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-run",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    first_rc = RunSuiteRunner(config).run()
    second_rc = RunSuiteRunner(config, resume=True).run()

    manifest = json.loads((tmp_path / "results" / "resume-run" / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 1
    assert len(cleanup_calls) == 1
    assert variant["task_counts"]["completed"] == 1
    assert variant["task_counts"]["skipped"] == 0
    task_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "resume-run" / "variants" / "baseline" / "task-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert task_rows == [
        {
            "instance_id": "psf__requests-1000",
            "bench": "Verified",
            "status": "completed",
            "record_path": str(
                tmp_path
                / "results"
                / "resume-run"
                / "variants"
                / "baseline"
                / "agent_runs"
                / "codex"
                / "Verified"
                / "psf__requests-1000"
                / "psf__requests-1000.codex-record.json"
            ),
            "resumed": True,
            "timeout": False,
            "ok": True,
        }
    ]


def test_run_suite_runner_resume_reruns_completed_ok_false_records(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-ok-false",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    assert RunSuiteRunner(config).run() == 0
    record_path = (
        tmp_path
        / "results"
        / "resume-ok-false"
        / "variants"
        / "baseline"
        / "agent_runs"
        / "codex"
        / "Verified"
        / "psf__requests-1000"
        / "psf__requests-1000.codex-record.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["ok"] = False
    record["exit_code"] = 1
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert RunSuiteRunner(config, resume=True).run() == 0

    assert len(call_log) == 2
    manifest = json.loads((tmp_path / "results" / "resume-ok-false" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variants"][0]["task_counts"]["completed"] == 1
    assert manifest["variants"][0]["task_counts"]["failed"] == 0


def test_run_suite_runner_resume_regenerates_conversion_when_record_changes(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-conversion-input-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    assert RunSuiteRunner(config).run() == 0
    variant_dir = tmp_path / "results" / "resume-conversion-input-change" / "variants" / "baseline"
    record_path = (
        variant_dir
        / "agent_runs"
        / "codex"
        / "Verified"
        / "psf__requests-1000"
        / "psf__requests-1000.codex-record.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["final_output"]["retrieved_context_files"] = ["requests/sessions.py"]
    record["final_output"]["retrieved_context_spans"] = {"requests/sessions.py": [{"start": 1, "end": 2}]}
    record["final_output"]["retrieved_context_symbols"] = {}
    record["final_output"]["retrieval_steps"] = []
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert RunSuiteRunner(config, resume=True).run() == 0

    pred_rows = [
        json.loads(line)
        for line in (variant_dir / "pred.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    conversion_summary = json.loads((variant_dir / "conversion-summary.json").read_text(encoding="utf-8"))
    assert len(call_log) == 1
    assert pred_rows[0]["traj_data"]["pred_files"] == ["requests/sessions.py"]
    assert conversion_summary["input_fingerprint"]


def test_run_suite_runner_resume_regenerates_resolution_when_record_changes(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    resolution_calls: list[dict[str, object]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: None,
    )

    def fake_resolution(**kwargs):
        resolution_calls.append(dict(kwargs))
        return {
            "status": "completed",
            "is_partial": False,
            "task_count": 1,
            "evaluated_task_count": 1,
            "resolved_count": 0,
            "pass_at_1": 0.0,
        }

    monkeypatch.setattr("contextbench.run_suites_core.runner.evaluate_resolution_for_suite", fake_resolution)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-resolution-input-change",
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

    assert RunSuiteRunner(config).run() == 0
    variant_dir = tmp_path / "results" / "resume-resolution-input-change" / "variants" / "baseline"
    record_path = (
        variant_dir
        / "agent_runs"
        / "codex"
        / "Verified"
        / "psf__requests-1000"
        / "psf__requests-1000.codex-record.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["model_patch"] = "diff --git a/changed.py b/changed.py\n"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert RunSuiteRunner(config, resume=True).run() == 0

    summary = json.loads((variant_dir / "resolution-summary.json").read_text(encoding="utf-8"))
    assert len(call_log) == 1
    assert len(resolution_calls) == 2
    assert summary["input_fingerprint"]


def test_run_suite_runner_resume_allows_limit_increase(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config_first = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-limit-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "limit": 1,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )
    config_second = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-limit-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "limit": 2,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    first_rc = RunSuiteRunner(config_first).run()
    second_rc = RunSuiteRunner(config_second, resume=True).run()

    manifest = json.loads((tmp_path / "results" / "resume-limit-change" / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 2
    assert [call["task_id"] for call in call_log] == ["psf__requests-1000", "psf__requests-1001"]
    assert len(cleanup_calls) == 2
    assert variant["task_counts"]["total"] == 2
    assert variant["task_counts"]["completed"] == 2
    assert variant["task_counts"]["skipped"] == 0


def test_run_suite_runner_resume_rejects_model_change(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config_first = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-model-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "model": "gpt-5.4",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )
    config_second = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-model-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "model": "gpt-5.3-codex",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    first_rc = RunSuiteRunner(config_first).run()

    assert first_rc == 0
    with pytest.raises(RuntimeError, match="already exists with a different effective config"):
        RunSuiteRunner(config_second, resume=True).run()


def test_run_suite_runner_resume_reruns_full_task_fanout_when_one_variant_is_missing(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-partial-task",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}, {"name": "plugin"}],
            "parallelism": {"max_workers": 2},
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    first_rc = RunSuiteRunner(config).run()

    plugin_record = (
        tmp_path
        / "results"
        / "resume-partial-task"
        / "variants"
        / "plugin"
        / "agent_runs"
        / "codex"
        / "Verified"
        / "psf__requests-1000"
        / "psf__requests-1000.codex-record.json"
    )
    plugin_record.unlink()

    second_rc = RunSuiteRunner(config, resume=True).run()
    manifest = json.loads((tmp_path / "results" / "resume-partial-task" / "manifest.json").read_text(encoding="utf-8"))

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 4
    assert len(cleanup_calls) == 4
    assert all(variant["task_counts"]["completed"] == 1 for variant in manifest["variants"])
    assert all(variant["task_counts"]["skipped"] == 0 for variant in manifest["variants"])
