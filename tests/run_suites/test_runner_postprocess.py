
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


class _PreflightGold:
    repo_url = "https://github.com/psf/requests.git"
    commit = "abc120"


class _PreflightGoldLoader:
    def __init__(self, path: str):
        self.path = path

    def get(self, instance_id: str):
        if instance_id == "psf__requests-1000":
            return _PreflightGold()
        return None


def test_postprocess_rejects_no_context_error_when_prediction_has_final_context() -> None:
    with pytest.raises(RuntimeError, match="predictions contain final context"):
        postprocess._assert_evaluation_artifact_consistent(
            results=[{"instance_id": "task-1", "error": "no_context_extracted"}],
            predictions_by_instance_id={
                "task-1": {
                    "instance_id": "task-1",
                    "traj_data": {
                        "pred_files": ["src/a.py"],
                        "pred_spans": {},
                        "pred_symbols": {},
                    },
                }
            },
        )


def test_run_suite_finalization_refreshes_summary_from_stage_artifacts(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.treesitter_available", lambda: True)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "refresh-rollups",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": True, "resolve": False, "runtime_backend": "host"},
        }
    )
    runner = RunSuiteRunner(config)
    tasks, task_set = runner._load_tasks()
    variant = build_run_suite_variant(config, config.variants[0])
    entry = runner._initial_variant_entry(variant)
    state = runner._prepare_variant_state(variant, entry, total_tasks=len(tasks))

    task = tasks[0]
    task_dir = runner._task_output_dir(state, task)
    workspace_path = task_dir / "workspaces" / "baseline"
    workspace_path.mkdir(parents=True, exist_ok=True)
    record_path = runner._task_record_path(state, task)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            _make_fake_agent_record(
                task=task,
                agent="codex",
                task_dir=task_dir,
                workspace_path=workspace_path,
            )
        ),
        encoding="utf-8",
    )
    state.pred_path.write_text(json.dumps({"instance_id": task["instance_id"]}) + "\n", encoding="utf-8")
    state.eval_results_path.write_text(json.dumps({"instance_id": task["instance_id"]}) + "\n", encoding="utf-8")
    (state.pred_path.parent / "conversion-summary.json").write_text(
        json.dumps(
            {
                "selected_task_count": 1,
                "record_count": 1,
                "convertible_record_count": 1,
                "prediction_count": 1,
                "missing_record_path_count": 0,
                "nonconvertible_record_count": 0,
                "is_partial": False,
                "fingerprint_version": 6,
            }
        ),
        encoding="utf-8",
    )
    state.eval_summary_path.write_text(
        json.dumps(
            {
                "num_valid": 1,
                "num_total": 1,
                "prediction_count": 1,
                "evaluated_prediction_count": 1,
                "selected_task_count": 1,
                "has_errors": False,
                "error_counts": {},
                "is_partial": False,
                "fingerprint_version": 6,
            }
        ),
        encoding="utf-8",
    )
    entry["metrics"] = {
        "evaluation": {
            "num_valid": 0,
            "evaluated_prediction_count": 0,
            "selected_task_count": 1,
            "is_partial": True,
            "fingerprint_version": 5,
        },
        "evaluation_partial": True,
        "postprocess_partial": True,
        "integrity": {"ok": False, "failed_checks": [{"name": "stale"}]},
    }
    entry["status"] = "postprocess_failed"
    entry["completed_at"] = "2026-06-01T00:00:00Z"
    entry["duration_ms"] = 12345
    entry["errors"] = ["integrity checks failed: stale"]
    runner._write_manifest(
        started_at="2026-06-01T00:00:00Z",
        completed_at="2026-06-01T00:01:00Z",
        task_set=task_set,
        variant_entries=[entry],
    )

    assert runner.refresh_rollup_artifacts() == 0

    summary_row = json.loads(runner.summary_json_path.read_text(encoding="utf-8"))[0]
    public_summary_row = json.loads((runner.public_artifacts_dir / "summary.json").read_text(encoding="utf-8"))[0]
    refreshed_manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    refreshed_variant = refreshed_manifest["variants"][0]
    integrity = json.loads((state.pred_path.parent / "integrity.json").read_text(encoding="utf-8"))

    assert refreshed_variant["status"] == "completed"
    assert refreshed_variant["completed_at"] == "2026-06-01T00:00:00Z"
    assert refreshed_variant["duration_ms"] == 12345
    assert refreshed_variant["errors"] == []
    assert summary_row["evaluation_partial"] is False
    assert summary_row["evaluation_is_partial"] is False
    assert summary_row["evaluation_num_valid"] == 1
    assert summary_row["evaluation_fingerprint_version"] == 6
    assert summary_row["integrity_ok"] is True
    assert summary_row["integrity_failed_checks"] == []
    assert integrity["ok"] is True
    for key in (
        "evaluation_partial",
        "evaluation_is_partial",
        "evaluation_num_valid",
        "evaluation_fingerprint_version",
        "integrity_ok",
        "integrity_failed_checks",
    ):
        assert public_summary_row[key] == summary_row[key]


def test_refresh_rollups_can_publish_aligned_artifacts(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.treesitter_available", lambda: True)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "refresh-aligned-rollups",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": True, "resolve": False, "runtime_backend": "host"},
        }
    )
    runner = RunSuiteRunner(config, refresh_rollups=True, refresh_artifact_suffix="aligned")
    tasks, task_set = runner._load_tasks()
    variant = build_run_suite_variant(config, config.variants[0])
    variant_dir = runner.experiment_dir / "variants" / variant.slug
    variant_dir.mkdir(parents=True, exist_ok=True)
    task_results_path = variant_dir / "task-results.jsonl"
    task_results_path.write_text(
        json.dumps(
            {
                "instance_id": tasks[0]["instance_id"],
                "original_inst_id": tasks[0]["instance_id"],
                "bench": tasks[0]["bench"],
                "status": "completed",
                "ok": True,
                "timeout": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "effective-config.json").write_text(json.dumps({"effective_config": {"name": "baseline"}}), encoding="utf-8")
    (variant_dir / "pred.jsonl").write_text(json.dumps({"instance_id": tasks[0]["instance_id"]}) + "\n", encoding="utf-8")
    (variant_dir / "eval.jsonl").write_text(json.dumps({"instance_id": tasks[0]["instance_id"], "legacy": True}) + "\n", encoding="utf-8")
    (variant_dir / "eval-summary.json").write_text(
        json.dumps({"num_valid": 0, "evaluated_prediction_count": 0, "selected_task_count": 1, "is_partial": True}),
        encoding="utf-8",
    )
    (variant_dir / "pred.aligned.jsonl").write_text(json.dumps({"instance_id": tasks[0]["instance_id"]}) + "\n", encoding="utf-8")
    aligned_eval_row = {
        "instance_id": tasks[0]["instance_id"],
        "predicted_context_path_diagnostics": {
            "missing_final_path_count": 1,
            "missing_trajectory_path_count": 2,
        },
    }
    (variant_dir / "eval.aligned.jsonl").write_text(json.dumps(aligned_eval_row) + "\n", encoding="utf-8")
    (variant_dir / "conversion-summary.aligned.json").write_text(
        json.dumps(
            {
                "selected_task_count": 1,
                "record_count": 1,
                "convertible_record_count": 1,
                "prediction_count": 1,
                "missing_record_path_count": 0,
                "nonconvertible_record_count": 0,
                "is_partial": False,
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "eval-summary.aligned.json").write_text(
        json.dumps(
            {
                "num_valid": 1,
                "num_total": 1,
                "prediction_count": 1,
                "evaluated_prediction_count": 1,
                "selected_task_count": 1,
                "has_errors": False,
                "error_counts": {},
                "is_partial": False,
            }
        ),
        encoding="utf-8",
    )

    runner._write_manifest(
        started_at="2026-06-01T00:00:00Z",
        completed_at="2026-06-01T00:01:00Z",
        task_set=task_set,
        variant_entries=[
            {
                "name": variant.name,
                "slug": variant.slug,
                "status": "postprocess_failed",
                "completed_at": "2026-06-01T00:00:00Z",
                "duration_ms": 123,
                "task_counts": {"total": 1, "completed": 1, "failed": 0, "timeout": 0, "skipped": 0},
                "metrics": {},
                "errors": [],
                "warnings": [],
                "output_dir": str(variant_dir),
                "raw_runs_dir": str(variant_dir / "agent_runs"),
                "task_results_path": str(task_results_path),
                "effective_config_path": str(variant_dir / "effective-config.json"),
            }
        ],
    )

    assert runner.refresh_rollup_artifacts() == 0

    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    summary_row = json.loads(runner.summary_json_path.read_text(encoding="utf-8"))[0]
    public_eval_summary = json.loads(
        (runner.public_artifacts_dir / "variants" / variant.slug / "eval-summary.json").read_text(encoding="utf-8")
    )
    public_eval_row = json.loads(
        (runner.public_artifacts_dir / "variants" / variant.slug / "eval.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert manifest["postprocess_artifact_suffix"] == "aligned"
    assert summary_row["eval_results_path"].endswith("eval.aligned.jsonl")
    assert summary_row["evaluation_num_valid"] == 1
    assert public_eval_summary["num_valid"] == 1
    assert public_eval_row["predicted_context_path_diagnostics"]["missing_final_path_count"] == 1


def test_run_suite_runner_uses_postprocess_container_without_duplicate_python(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr("contextbench.run_suites_core.runner.GoldLoader", _PreflightGoldLoader)
    monkeypatch.setattr("contextbench.run_suites_core.runner.GoldLoader", _PreflightGoldLoader)

    docker_commands: list[list[str]] = []
    variant_dir = tmp_path / "results" / "docker-postprocess" / "variants" / "baseline"
    expected_eval_cache = tmp_path / "cache" / "postprocess-eval" / "docker-postprocess" / "baseline"
    stale_worktree_file = expected_eval_cache / "worktrees" / "stale.txt"
    stale_worktree_file.parent.mkdir(parents=True, exist_ok=True)
    stale_worktree_file.write_text("stale", encoding="utf-8")

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        del log_prefix, env
        docker_commands.append(list(command))
        if "convert" in command:
            out_path = Path(command[command.index("--out-path") + 1].replace("/work/", str(variant_dir) + "/"))
            summary_path = Path(command[command.index("--summary-path") + 1].replace("/work/", str(variant_dir) + "/"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            (variant_dir / "conversion-error.json").write_text('{"log_path":"/Users/nobbe/private.log"}', encoding="utf-8")
            out_path.write_text(json.dumps({"instance_id": "psf__requests-1000"}) + "\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "scope": "converted_predictions",
                        "selected_task_count": 1,
                        "record_count": 1,
                        "convertible_record_count": 1,
                        "prediction_count": 1,
                        "missing_record_path_count": 0,
                        "nonconvertible_record_count": 0,
                        "coverage_of_attempted_tasks": 1.0,
                        "missing_prediction_count": 0,
                        "is_partial": False,
                    }
                ),
                encoding="utf-8",
            )
        elif "evaluate" in command:
            out_path = Path(command[command.index("--out-path") + 1].replace("/work/", str(variant_dir) + "/"))
            summary_path = Path(command[command.index("--summary-path") + 1].replace("/work/", str(variant_dir) + "/"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            (variant_dir / "evaluation-error.json").write_text('{"log_path":"/Users/nobbe/private.log"}', encoding="utf-8")
            out_path.write_text(json.dumps({"instance_id": "psf__requests-1000"}) + "\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "num_valid": 1,
                        "num_total": 1,
                        "prediction_count": 1,
                        "evaluated_prediction_count": 1,
                        "selected_task_count": 1,
                        "coverage_of_attempted_tasks": 1.0,
                        "missing_prediction_count": 0,
                        "is_partial": False,
                    }
                ),
                encoding="utf-8",
            )
        return 0, "ok"

    monkeypatch.setattr("contextbench.run_suites_core.runner._run_resolution_command", fake_run_resolution_command)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "docker-postprocess",
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
                "evaluate": True,
                "resolve": False,
                "runtime_backend": "docker",
                "runtime_image": "contextbench-postprocess:test",
                },
            }
        )

    runner = RunSuiteRunner(config)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner._docker_image_id",
        lambda image: pytest.fail("postprocess fingerprints must use the cached Docker image id"),
    )

    assert runner.run() == 0
    manifest = json.loads((tmp_path / "results" / "docker-postprocess" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["postprocess_runtime"] == {
        "backend": "docker",
        "image": "contextbench-postprocess:test",
        "image_id": "sha256:test-postprocess",
    }
    assert len(docker_commands) == 2
    assert not (variant_dir / "conversion-error.json").exists()
    assert not (variant_dir / "evaluation-error.json").exists()
    for index, command in enumerate(docker_commands):
        assert command[:3] == ["docker", "run", "--rm"]
        assert command[command.index("-w") + 1] == "/repo"
        assert "contextbench-postprocess:test" in command
        assert command.count("python") == 0
        assert "-m" in command
        assert "contextbench.run_suites_postprocess" in command
        if index == 1:
            volume_values = [
                command[i + 1]
                for i, token in enumerate(command[:-1])
                if token == "-v"
            ]
            env_values = [
                command[i + 1]
                for i, token in enumerate(command[:-1])
                if token == "-e"
            ]
            assert f"{expected_eval_cache.resolve()}:/cache/eval:rw" in volume_values
            assert "CONTEXTBENCH_TMP_ROOT=/cache/eval/worktrees" in env_values
            assert "GIT_CONFIG_COUNT=1" in env_values
            assert "GIT_CONFIG_KEY_0=safe.directory" in env_values
            assert "GIT_CONFIG_VALUE_0=/cache/eval/*" in env_values
            assert command[command.index("--cache-dir") + 1] == "/cache/eval"
            assert command[command.index("--workspace-key") + 1] == "docker-postprocess-baseline-evaluation"
            assert command[command.index("--tmp-root") + 1] == "/cache/eval/worktrees"
    assert stale_worktree_file.exists()


def test_run_suite_runner_rejects_postprocess_image_missing_evaluation_parsers(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.GoldLoader", _PreflightGoldLoader)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner._postprocess_image_supports_evaluation",
        lambda image: (False, "missing tree-sitter parsers: c_sharp"),
    )
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "docker-postprocess-missing-parser",
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
                "evaluate": True,
                "resolve": False,
                "runtime_backend": "docker",
                "runtime_image": "contextbench-postprocess:test",
            },
        }
    )

    with pytest.raises(RuntimeError, match="missing required evaluation parsers"):
        RunSuiteRunner(config)


def test_run_suite_runner_stages_conversion_before_resolution_for_all_variants(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    stage_order: list[tuple[str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr("contextbench.run_suites_core.runner.GoldLoader", _PreflightGoldLoader)

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        del log_path, log_prefix, env
        variant_dir = Path(cwd)
        variant_name = variant_dir.name
        if "convert" in command:
            stage_order.append(("convert", variant_name))
            out_path = Path(command[command.index("--out-path") + 1].replace("/work/", str(variant_dir) + "/"))
            summary_path = Path(command[command.index("--summary-path") + 1].replace("/work/", str(variant_dir) + "/"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"instance_id": "psf__requests-1000"}) + "\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "scope": "converted_predictions",
                        "selected_task_count": 1,
                        "record_count": 1,
                        "convertible_record_count": 1,
                        "prediction_count": 1,
                        "missing_record_path_count": 0,
                        "nonconvertible_record_count": 0,
                        "coverage_of_attempted_tasks": 1.0,
                        "missing_prediction_count": 0,
                        "is_partial": False,
                    }
                ),
                encoding="utf-8",
            )
        return 0, "ok"

    def fake_evaluate_resolution_for_suite(**kwargs):
        stage_order.append(("resolve", kwargs["variant_name"]))
        return {
            "status": "completed",
            "backend": "mixed",
            "task_count": 1,
            "prediction_count": 1,
            "evaluated_task_count": 1,
            "evaluated_prediction_count": 1,
            "resolved_count": 1,
            "pass_at_1": 1.0,
            "supported_benches": ["Verified"],
            "successful_benches": ["Verified"],
            "failed_benches": [],
            "unsupported_benches": [],
            "coverage_of_attempted_tasks": 1.0,
            "is_partial": False,
            "per_bench": {},
        }

    monkeypatch.setattr("contextbench.run_suites_core.runner._run_resolution_command", fake_run_resolution_command)
    monkeypatch.setattr("contextbench.run_suites_core.runner.evaluate_resolution_for_suite", fake_evaluate_resolution_for_suite)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "docker-postprocess-staged",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}, {"name": "treatment"}],
            "postprocess": {
                "convert": True,
                "evaluate": False,
                "resolve": True,
                "runtime_backend": "docker",
                "runtime_image": "contextbench-postprocess:test",
            },
        }
    )

    assert RunSuiteRunner(config).run() == 0
    assert stage_order == [
        ("convert", "baseline"),
        ("convert", "treatment"),
        ("resolve", "baseline"),
        ("resolve", "treatment"),
    ]


def test_run_suite_runner_fails_postprocess_container_without_retrying_or_resolving(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr("contextbench.run_suites_core.runner.GoldLoader", _PreflightGoldLoader)

    attempts = {"convert": 0, "evaluate": 0, "resolve": 0}

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        del cwd, log_path, log_prefix, env
        variant_dir = tmp_path / "results" / "docker-postprocess-retry" / "variants" / "baseline"
        if "convert" in command:
            attempts["convert"] += 1
            return 1, "deterministic convert failure"
        if "evaluate" in command:
            attempts["evaluate"] += 1
            return 0, "unexpected evaluation"
        return 0, "ok"

    monkeypatch.setattr("contextbench.run_suites_core.runner._run_resolution_command", fake_run_resolution_command)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: attempts.__setitem__("resolve", attempts["resolve"] + 1) or {},
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "docker-postprocess-retry",
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
                "evaluate": True,
                "resolve": True,
                "runtime_backend": "docker",
                "runtime_image": "contextbench-postprocess:test",
                },
            }
        )

    assert RunSuiteRunner(config).run() == 1
    assert attempts == {"convert": 1, "evaluate": 0, "resolve": 0}
    error_path = tmp_path / "results" / "docker-postprocess-retry" / "variants" / "baseline" / "conversion-error.json"
    assert json.loads(error_path.read_text(encoding="utf-8"))["tail"] == "deterministic convert failure"
