
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.runtime import (
    build_claude_command,
    build_codex_command,
    prepare_claude_runtime_files,
    prepare_codex_runtime_env,
    run_command,
    run_coding_agent_task,
    validate_claude_auth,
    validate_claude_isolation,
)
from contextbench.coding_agents.runtime_backends import (
    RuntimeBackendConfig,
    DockerTaskRuntime,
    normalize_runtime_backend_config,
)
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"

def test_run_coding_agent_task_closes_timed_out_docker_runtime_before_diff(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")
    task = {
        "bench": "Verified",
        "instance_id": "task-timeout",
        "original_inst_id": "task-timeout",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }

    class FakeDockerRuntime:
        config = RuntimeBackendConfig(backend="docker", image="fake")

        def __init__(self) -> None:
            self.closed = False

        def start(self) -> None:
            return None

        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("timed out", encoding="utf-8")
            return {"ok": False, "exit_code": None, "signal": "SIGTERM", "timeout": True}

        def close(self, *, success: bool) -> None:
            assert success is False
            self.closed = True

    fake_runtime = FakeDockerRuntime()
    captured_runtime_kwargs: dict[str, object] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    def fake_create_task_runtime(*args, **kwargs):
        captured_runtime_kwargs.update(kwargs)
        return fake_runtime

    monkeypatch.setattr("contextbench.coding_agents.runtime.create_task_runtime", fake_create_task_runtime)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: prepare_codex_runtime_env(
            task_dir,
            source_codex_dir=source_codex_dir,
            include_host_env=False,
        ),
    )

    def fake_git_diff(path):
        assert fake_runtime.closed is True
        return ""

    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", fake_git_diff)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="docker",
        runtime_image="fake",
    )

    assert record["timeout"] is True
    assert fake_runtime.closed is True
    task_dir = (output_dir / "task-timeout").resolve()
    assert captured_runtime_kwargs["extra_writable_dirs"] == [codex_runtime_root(task_dir)]
    assert not (task_dir / "codex-runtime" / "home" / ".codex" / "auth.json").exists()

def test_run_coding_agent_task_codex_setup_prompt_failure_short_circuits_scored_run(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-setup-fail",
        "original_inst_id": "task-setup-fail",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"final_output_paths": {}, "calls": []}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")

    def fake_build_codex_command(**kwargs):
        phase = "setup" if kwargs["schema_path"] is None else "main"
        captured["final_output_paths"][phase] = kwargs["final_output_path"]
        return ["codex", "exec", phase], f"{phase}-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        phase = command[-1]
        captured["calls"].append(phase)
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 8, "output_tokens": 1}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("setup failed", encoding="utf-8")
        if phase != "setup":
            pytest.fail("scored prompt should not run after setup failure")
        final_output_path = captured["final_output_paths"][phase]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text("setup failed", encoding="utf-8")
        return {"ok": False, "exit_code": 9, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        setup={"setup_prompt": "Bootstrap tools"},
        runtime_backend="host",
    )

    task_dir = output_dir / "task-setup-fail"

    assert captured["calls"] == ["setup"]
    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["raw_response_path"] is None
    assert record["token_usage"] is None
    assert record["tool_calls"] == []
    assert record["setup_run"]["status"] == "failed"
    assert record["setup_run"]["exit_code"] == 9
    assert Path(record["setup_run"]["raw_response_path"]).exists()
    assert Path(record["prompt_path"]).exists()
    assert not (task_dir / "raw-response.json").exists()
