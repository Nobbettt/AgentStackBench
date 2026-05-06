
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

def test_run_suite_runner_writes_manifest_and_variant_outputs(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-variants",
            "description": "Compare baseline and plugin setup.",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "reasoning_effort": "medium",
                "runtime_env": {"RUNTIME_BASE": "1"},
            },
            "variants": [
                {"name": "baseline"},
                {
                    "name": "with-plugin",
                    "reasoning_effort": "xhigh",
                    "agent_args_add": ["--plugin"],
                    "env_add": {"PLUGIN": "1"},
                    "runtime_env_add": {"RUNTIME_PLUGIN": "1"},
                    "setup": {
                        "prompt_preamble": "Plugin enabled",
                        "setup_prompt": "Bootstrap plugin",
                        "setup_prompt_timeout": 45,
                    },
                },
            ],
            "parallelism": {"max_workers": 2},
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "codex-variants"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_rows = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert manifest["status"] == "completed"
    assert len(manifest["variants"]) == 2
    assert len(call_log) == 4
    assert len(cleanup_calls) == 4
    assert all(Path(row["pred_path"]).exists() for row in summary_rows)
    assert (experiment_dir / "summary.csv").exists()
    assert (experiment_dir / "public-artifacts" / "manifest.json").exists()
    assert [call["task_id"] for call in call_log[:2]] == ["psf__requests-1000", "psf__requests-1000"]
    assert [call["task_id"] for call in call_log[2:]] == ["psf__requests-1001", "psf__requests-1001"]
    assert len({call["workspace_key"] for call in call_log[:2]}) == 2
    assert len({call["workspace_key"] for call in call_log[2:]}) == 2

    plugin_calls = [call for call in call_log if call["prompt_preamble"] == "Plugin enabled"]
    assert len(plugin_calls) == 2
    assert all(call["agent_args"] == ["--plugin"] for call in plugin_calls)
    assert all(call["reasoning_effort"] == "xhigh" for call in plugin_calls)
    assert all(call["env"] == {"PLUGIN": "1"} for call in plugin_calls)
    assert all(call["setup"]["setup_prompt"] == "Bootstrap plugin" for call in plugin_calls)
    assert all(call["setup"]["setup_prompt_timeout"] == 45 for call in plugin_calls)
    assert all(call["runtime_backend"] == "docker" for call in plugin_calls)
    assert all(call["runtime_image"] == DEFAULT_CODEX_RUNTIME_IMAGE for call in plugin_calls)
    assert all(call["runtime_env"] == {"RUNTIME_BASE": "1", "RUNTIME_PLUGIN": "1"} for call in plugin_calls)
    assert all(call["runtime_setup_commands"] == [] for call in plugin_calls)

    with open(experiment_dir / "summary.csv", "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in rows] == ["baseline", "with-plugin"]


def test_run_suite_runner_fails_fast_when_evaluation_dependencies_missing(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.treesitter_available", lambda: False)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "missing-treesitter",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": True, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Tree-sitter is not available for evaluation"):
        RunSuiteRunner(config)


def test_run_suite_runner_preflight_rejects_limited_full_suite(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-full-suite",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "limit": 1,
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline", "runtime_backend": "host"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "codex-full-suite" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "limited_full_suite_config"


def test_run_suite_runner_preflight_rejects_selected_full_suite(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-full-suite",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "limit": 0,
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline", "runtime_backend": "host"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "codex-full-suite" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "selected_full_suite_config"
    assert proof["failures"][0]["selectors"]["task_csv"] == str(task_csv)


def test_run_suite_runner_preflight_requires_task_repo_metadata(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    rows = json.loads(task_data.read_text(encoding="utf-8"))
    rows[0]["repo_url"] = ""
    task_data.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "missing-task-repo-metadata",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline", "runtime_backend": "host"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "missing-task-repo-metadata" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "missing_task_repo_metadata"
    assert proof["failures"][0]["instance_ids"] == ["psf__requests-1000"]


def test_run_suite_config_rejects_claude_docker_until_auth_is_defined(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="not supported for Claude"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "claude-docker-auth",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "runtime_backend": "docker",
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False, "runtime_backend": "host"},
            }
        )


def test_run_suite_runner_does_not_fail_fast_when_resolution_harness_missing(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    resolution_preflight_calls: list[list[str]] = []

    def fail_if_resolution_preflight_runs(benches: list[str]) -> list[dict[str, object]]:
        resolution_preflight_calls.append(benches)
        raise AssertionError("resolution backend preflight should not run when postprocess.resolve is disabled")

    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.describe_resolution_backend_support",
        fail_if_resolution_preflight_runs,
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "missing-swebench-harness",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False, "resolve": False},
        }
    )

    runner = RunSuiteRunner(config)
    tasks, _task_set = runner._load_tasks()
    runner._validate_preflight(tasks, [build_run_suite_variant(config, config.variants[0])])

    assert resolution_preflight_calls == []


def test_run_suite_runner_cleans_successful_worktrees_but_keeps_failed_runs(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
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
        suffix = "codex" if agent == "codex" else "claude"
        status = "completed" if "baseline" in str(workspace_key) else "failed"
        record = _make_fake_agent_record(
            task=task,
            agent=agent,
            task_dir=task_dir,
            workspace_path=workspace_path,
            status=status,
            timeout=status != "completed",
        )
        record_path = task_dir / f"{task_id}.{suffix}-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", fake_run)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "cleanup-run",
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

    rc = RunSuiteRunner(config).run()
    manifest = json.loads((tmp_path / "results" / "cleanup-run" / "manifest.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert len(cleanup_calls) == 1
    assert "baseline" in cleanup_calls[0][2]
    assert any(variant["status"] == "completed_with_failures" for variant in manifest["variants"])
