
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
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


def test_run_suite_runner_global_scheduler_runs_across_instances(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=3)
    lock = threading.Lock()
    active = 0
    max_active = 0
    call_log: list[dict[str, object]] = []

    def fake_run(
        *,
        task,
        agent,
        output_dir,
        workspace_key=None,
        **kwargs,
    ):
        nonlocal active, max_active
        del kwargs
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            task_id = safe_path_component(task.get("instance_id") or "task")
            task_dir = output_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            workspace_path = task_dir / "workspaces" / safe_path_component(workspace_key or task_id)
            workspace_path.mkdir(parents=True, exist_ok=True)
            record = _make_fake_agent_record(
                task=task,
                agent=agent,
                task_dir=task_dir,
                workspace_path=workspace_path,
            )
            record_path = task_dir / f"{task_id}.{agent}-record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with lock:
                call_log.append({"task_id": task.get("instance_id"), "workspace_key": workspace_key})
            return record
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", fake_run)
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "global-scheduler",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}, {"name": "plugin"}],
            "parallelism": {"max_workers": 2, "agent_workers": 3, "scheduler": "global"},
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "global-scheduler"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline_rows = [
        json.loads(line)
        for line in (experiment_dir / "variants" / "baseline" / "task-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    plugin_rows = [
        json.loads(line)
        for line in (experiment_dir / "variants" / "plugin" / "task-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rc == 0
    assert len(call_log) == 6
    assert max_active == 3
    assert manifest["scheduler"] == {"mode": "global", "agent_workers": 3, "max_workers": 2}
    assert "agent_ms" in manifest["phase_timings"]
    assert [row["instance_id"] for row in baseline_rows] == [
        "psf__requests-1000",
        "psf__requests-1001",
        "psf__requests-1002",
    ]
    assert [row["instance_id"] for row in plugin_rows] == [
        "psf__requests-1000",
        "psf__requests-1001",
        "psf__requests-1002",
    ]


def test_run_suite_runner_prebuilds_runtime_image_generically(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    images = {"base-runtime:1.0"}
    docker_commands: list[list[str]] = []

    def fake_image_available(image: str) -> bool:
        return image in images

    def fake_run(command, capture_output, text, check):
        del capture_output, text, check
        docker_commands.append(list(command))
        if command[:2] == ["docker", "build"]:
            tag = command[command.index("--tag") + 1]
            images.add(tag)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", fake_image_available)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_id", lambda image: f"sha256:{safe_path_component(image)}")
    monkeypatch.setattr("contextbench.run_suites_core.runner.subprocess.run", fake_run)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "runtime-prebuild",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
                "runtime_image": "base-runtime:1.0",
                "runtime_prebuild": {
                    "enabled": True,
                    "commands": ["printf tool-ready"],
                    "env": {"TOOL_HOME": "/opt/tool"},
                },
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "runtime-prebuild"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    dockerfile = experiment_dir / "runtime-prebuild" / "baseline" / "Dockerfile"

    assert rc == 0
    assert docker_commands[0][:4] == ["docker", "build", "--tag", call_log[0]["runtime_image"]]
    assert call_log[0]["runtime_image"].startswith("contextbench-runtime-prebuild:runtime-prebuild-baseline-")
    assert manifest["runtime_prebuilds"][0]["status"] == "built"
    assert manifest["runtime_prebuilds"][0]["base_image"] == "base-runtime:1.0"
    assert dockerfile.read_text(encoding="utf-8").splitlines() == [
        "FROM base-runtime:1.0",
        'SHELL ["/bin/sh", "-lc"]',
        'ENV TOOL_HOME="/opt/tool"',
        "RUN printf tool-ready",
    ]


def test_run_suite_runner_uses_resolution_images_as_agent_runtime(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    call_log: list[dict[str, object]] = []
    prepared: dict[str, object] = {}
    bundle_root = tmp_path / "codex-bundle"
    bundle_root.mkdir()

    def fake_prepare_resolution_images_for_tasks(**kwargs):
        prepared.update(kwargs)
        return {
            "status": "completed",
            "scope": "resolution_runtime_images",
            "task_count": len(kwargs["tasks"]),
            "bench_count": 1,
            "max_workers": kwargs["max_workers"],
            "benches": {},
            "images": [
                {
                    "bench": "Verified",
                    "instance_id": "psf__requests-1000",
                    "image": "sweb.eval.x86_64.psf__requests-1000:latest",
                }
            ],
        }

    def fake_image_available(image: str) -> bool:
        return str(image).startswith("sweb.eval.x86_64.psf__requests-")

    monkeypatch.setattr("contextbench.run_suites_core.runner.codex_tool_bundle_root", lambda runtime_env=None: bundle_root)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", fake_image_available)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_platform", lambda image: "linux/amd64")
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.prepare_resolution_images_for_tasks",
        fake_prepare_resolution_images_for_tasks,
    )
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resolution-runtime-agent",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "runtime_backend": "docker",
                "runtime_image_source": "resolution",
                "runtime_platform": "linux/amd64",
            },
            "variants": [{"name": "baseline"}],
            "parallelism": {"max_workers": 1},
            "postprocess": {
                "convert": True,
                "evaluate": False,
                "runtime_backend": "host",
                "prebuild_resolution_workers": 3,
            },
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "resolution-runtime-agent"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert prepared["max_workers"] == 3
    assert Path(prepared["work_dir"]) == experiment_dir / "runtime-resolution-images"
    assert [call["runtime_image"] for call in call_log] == [
        "sweb.eval.x86_64.psf__requests-1000:latest",
        "sweb.eval.x86_64.psf__requests-1001:latest",
    ]
    assert manifest["runtime_resolution_images"]["status"] == "completed"
    assert manifest["runtime_resolution_images"]["source"] == "resolution"


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


def test_run_suite_runner_preflight_rejects_limited_full_dataset_assertion(tmp_path, monkeypatch) -> None:
    task_data, _task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-explicit-full-suite",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": None,
                "selection_assertion": "full_dataset",
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

    proof = json.loads((tmp_path / "results" / "codex-explicit-full-suite" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "limited_full_dataset_assertion"


def test_run_suite_runner_preflight_rejects_selected_full_dataset_assertion(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-full-suite",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "selection_assertion": "full_dataset",
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
    assert proof["failures"][0]["kind"] == "selected_full_dataset_assertion"
    assert proof["failures"][0]["selectors"]["task_csv"] == str(task_csv)


def test_run_suite_runner_preflight_rejects_full_dataset_configured_selection_assertion(
    tmp_path,
    monkeypatch,
) -> None:
    task_data, _task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-configured-selection-assertion",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": None,
                "selection_assertion": "configured_selection",
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

    proof = json.loads(
        (tmp_path / "results" / "codex-configured-selection-assertion" / "preflight.failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["failures"][0]["kind"] == "configured_selection_assertion"


def test_run_suite_runner_does_not_infer_full_dataset_from_prose(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr("contextbench.run_suites_core.runner.remove_worktree", lambda *args, **kwargs: None)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-full-suite-prose-only",
            "description": "This description mentions a fully configured subset, not an asserted full dataset.",
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
            "postprocess": {"convert": False, "evaluate": False, "runtime_backend": "host"},
        }
    )

    assert RunSuiteRunner(config).run() == 0
    assert len(call_log) == 1


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


def test_run_suite_runner_preflight_rejects_available_tool_requirements_for_unsupported_adapter(
    tmp_path,
    monkeypatch,
) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-tool-availability",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "host",
            },
            "variants": [
                {
                    "name": "requires-tools",
                    "runtime_backend": "host",
                    "required_available_tool_patterns_add": [r"^mcp__demo__"],
                }
            ],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "codex-tool-availability" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "required_available_tools_unsupported"
    assert proof["failures"][0]["requirements"][0]["agent"] == "codex"


def test_run_suite_runner_preflight_rejects_claude_docker_without_portable_auth(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    empty_claude_dir = tmp_path / "empty-claude"
    empty_claude_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty_claude_dir))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_platform", lambda image: "linux/amd64")
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner._claude_host_auth_status_summary",
        lambda: {
            "checked": True,
            "exit_code": 0,
            "logged_in": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
        },
    )
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-docker-no-portable-auth",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
                "runtime_image": "contextbench-claude-runtime:test",
                "runtime_platform": "linux/amd64",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "claude-docker-no-portable-auth" / "preflight.failure.json").read_text(encoding="utf-8"))
    failure = proof["failures"][0]
    assert failure["kind"] == "claude_docker_portable_auth_unavailable"
    assert failure["variants"][0]["variant"] == "baseline"
    assert failure["variants"][0]["source_config_dir"] == str(empty_claude_dir)
    assert "CLAUDE_CODE_OAUTH_TOKEN" in failure["variants"][0]["checked_env_vars"]
    assert failure["host_auth"]["logged_in"] is True
    assert "setup-token" in failure["message"]


def test_run_suite_runner_preflight_accepts_claude_docker_portable_auth_from_runtime_env(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty-claude"))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_platform", lambda image: "linux/amd64")

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-docker-runtime-env-auth",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
                "runtime_image": "contextbench-claude-runtime:test",
                "runtime_platform": "linux/amd64",
                "runtime_env": {"CLAUDE_CODE_OAUTH_TOKEN": "configured-token"},
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    runner = RunSuiteRunner(config)
    tasks, task_set = runner._load_tasks()
    runner._validate_preflight(tasks, [build_run_suite_variant(config, config.variants[0])], task_set)


def test_run_suite_runner_preflight_rejects_runtime_platform_mismatch(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task([]))
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_platform", lambda image: "linux/arm64")

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "runtime-platform-mismatch",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
                "runtime_image": "contextbench-claude-runtime:test-amd64",
                "runtime_platform": "linux/amd64",
                "runtime_env": {"CLAUDE_CODE_OAUTH_TOKEN": "configured-token"},
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False, "runtime_backend": "host"},
        }
    )

    with pytest.raises(RuntimeError, match="Run-suite preflight failed"):
        RunSuiteRunner(config).run()

    proof = json.loads((tmp_path / "results" / "runtime-platform-mismatch" / "preflight.failure.json").read_text(encoding="utf-8"))
    assert proof["failures"][0]["kind"] == "runtime_image_platform_mismatch"
    assert proof["failures"][0]["images"][0]["expected_platform"] == "linux/amd64"
    assert proof["failures"][0]["images"][0]["actual_platform"] == "linux/arm64"


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
    tasks, task_set = runner._load_tasks()
    runner._validate_preflight(tasks, [build_run_suite_variant(config, config.variants[0])], task_set)

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
        runtime_platform=None,
        runtime_env=None,
        runtime_setup_timeout=None,
        runtime_validation_timeout=None,
        runtime_setup_cache=False,
        runtime_setup_cache_dir=None,
        runtime_setup_commands=(),
        runtime_validation_commands=(),
        diff_exclude_paths=(),
        required_tool_call_patterns=(),
        required_command_patterns=(),
        required_available_tool_patterns=(),
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
            runtime_platform,
            runtime_env,
            runtime_setup_timeout,
            runtime_validation_timeout,
            runtime_setup_cache,
            runtime_setup_cache_dir,
            runtime_setup_commands,
            runtime_validation_commands,
            diff_exclude_paths,
            required_tool_call_patterns,
            required_command_patterns,
            required_available_tool_patterns,
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
