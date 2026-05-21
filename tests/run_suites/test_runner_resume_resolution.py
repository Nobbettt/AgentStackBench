# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner

from .helpers import _fake_run_coding_agent_task, _write_task_inputs


def test_run_suite_runner_resume_keeps_empty_patch_records_by_default_when_resolution_enabled(tmp_path, monkeypatch) -> None:
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
            "experiment_name": "resume-empty-patch-resolution",
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
    assert RunSuiteRunner(config, resume=True).run() == 0

    assert len(call_log) == 1
    assert len(resolution_calls) == 2


def test_run_suite_runner_resume_can_rerun_empty_patch_records_when_explicitly_configured(tmp_path, monkeypatch) -> None:
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
            "experiment_name": "resume-empty-patch-resolution-repair",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {
                "convert": True,
                "evaluate": False,
                "resolve": True,
                "runtime_backend": "host",
                "rerun_empty_patch_records_on_resume": True,
            },
        }
    )

    assert RunSuiteRunner(config).run() == 0
    assert RunSuiteRunner(config, resume=True).run() == 0

    assert len(call_log) == 2
    assert len(resolution_calls) == 2


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
