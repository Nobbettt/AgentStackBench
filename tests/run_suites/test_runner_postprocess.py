
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import find_private_path_matches
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


def test_run_suite_runner_writes_resolution_summary(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=secret-token\n", encoding="utf-8")
    call_log: list[dict[str, object]] = []
    resolution_calls: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: resolution_calls.append(kwargs) or {
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
                "per_bench": {
                    "Verified": {
                        "bench": "Verified",
                        "backend": "swebench",
                        "status": "completed",
                        "task_count": 1,
                        "prediction_count": 1,
                        "resolved_count": 1,
                        "pass_at_1": 1.0,
                        "resolved_ids": ["psf__requests-1000"],
                        "unresolved_ids": [],
                        "coverage_of_attempted_tasks": 1.0,
                        "is_partial": False,
                        "log_path": str(tmp_path / "private" / "resolution-command.log"),
                    }
                },
                "evaluation_dir": str(tmp_path / "private" / "resolution-eval"),
            },
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-resolution",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "runtime_env_file": str(env_file),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {
                "convert": True,
                "evaluate": False,
                "resolve": True,
                "runtime_backend": "host",
                "env_file": str(env_file),
            },
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "codex-resolution"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    resolution_summary_path = Path(manifest["variants"][0]["resolution_summary_path"])
    resolution_summary = json.loads(resolution_summary_path.read_text(encoding="utf-8"))
    effective_config = json.loads((experiment_dir / "variants" / "baseline" / "effective-config.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert resolution_calls[0]["env"] == {"HF_TOKEN": "secret-token"}
    assert effective_config["effective_config"]["runtime_env"]["HF_TOKEN"] == "<redacted>"
    assert resolution_calls[0]["run_suffix"]
    assert resolution_calls[0]["resume_existing_resolution"] is False
    assert resolution_calls[0]["clean_resolution_artifacts"] is True
    assert resolution_summary["pass_at_1"] == 1.0
    assert resolution_summary["resolved_count"] == 1
    assert find_private_path_matches(resolution_summary) == []
    assert "<tmp>" in json.dumps(resolution_summary)


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
    assert not stale_worktree_file.exists()


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
