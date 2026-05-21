# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.artifact_sanitization import find_private_path_matches
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner

from .helpers import _fake_run_coding_agent_task, _write_task_inputs


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
        lambda **kwargs: resolution_calls.append(kwargs)
        or {
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
                "self_clean_resolution_artifacts": False,
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
    assert resolution_calls[0]["self_clean_resolution_artifacts"] is False
    assert resolution_calls[0]["self_clean_resolution_docker_images"] is True
    assert resolution_summary["pass_at_1"] == 1.0
    assert resolution_summary["resolved_count"] == 1
    assert find_private_path_matches(resolution_summary) == []
    assert "<tmp>" in json.dumps(resolution_summary)
