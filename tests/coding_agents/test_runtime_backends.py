# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-27 from upstream ContextBench.
# Summary of changes: cover Docker host mapping for OTEL collector access.

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
    _container_bootstrap_command,
    normalize_runtime_backend_config,
    run_runtime_setup_commands,
)
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"

def test_validate_claude_isolation_accepts_clean_verbose_response() -> None:
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "system",
                "subtype": "init",
                "plugins": [],
                "mcp_servers": {},
                "slash_commands": [],
            }
        ],
    }

    validate_claude_isolation(raw_response)

def test_validate_claude_isolation_rejects_loaded_plugins() -> None:
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "system",
                "subtype": "init",
                "plugins": ["skill"],
                "mcp_servers": {},
                "slash_commands": [],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="plugins are still loaded"):
        validate_claude_isolation(raw_response)

def test_validate_claude_auth_rejects_logged_out(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout='{"loggedIn": false, "authMethod": "none"}', stderr="")

    monkeypatch.setattr("contextbench.coding_agents.runtime.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="not logged in"):
        validate_claude_auth()

def test_run_command_timeout_preserves_streamed_output(tmp_path, monkeypatch) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            self.command = command
            kwargs["stdout"].write("partial stdout\n")
            kwargs["stderr"].write("partial stderr\n")

        def communicate(self, input=None, timeout=None):
            raise subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 124

    monkeypatch.setattr("contextbench.coding_agents.runtime_common.subprocess.Popen", FakeProcess)


    result = run_command(
        ["codex", "exec", "-"],
        cwd=tmp_path,
        stdin_text="prompt",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=30,
    )

    assert result == {"ok": False, "exit_code": None, "signal": "SIGTERM", "timeout": True}
    assert stdout_path.read_text(encoding="utf-8") == "partial stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "partial stderr\n"

def test_normalize_runtime_backend_config_requires_docker_image() -> None:
    with pytest.raises(RuntimeError, match="runtime_image is required"):
        normalize_runtime_backend_config(runtime_backend="docker")


def test_normalize_runtime_backend_config_requires_backend() -> None:
    with pytest.raises(RuntimeError, match="runtime_backend is required"):
        normalize_runtime_backend_config(runtime_backend="")


def test_normalize_runtime_backend_config_rejects_host_platform() -> None:
    with pytest.raises(RuntimeError, match="runtime_platform can only be used"):
        normalize_runtime_backend_config(runtime_backend="host", runtime_platform="linux/amd64")


def test_runtime_setup_commands_use_non_login_shell_to_preserve_prepared_env(tmp_path) -> None:
    workspace_path = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    workspace_path.mkdir()
    task_dir.mkdir()
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del host_runner
            calls.append(
                {
                    "command": list(command),
                    "cwd": cwd,
                    "stdin_text": stdin_text,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "timeout": timeout,
                    "env": dict(env or {}),
                }
            )
            stdout_path.write_text("ok", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    failure = run_runtime_setup_commands(
        FakeRuntime(),
        commands=["tool-from-prepared-path --version"],
        workspace_path=workspace_path,
        task_dir=task_dir,
        timeout=30,
        env={"PATH": "/prepared/bin:/usr/bin:/bin"},
    )

    assert failure is None
    assert calls == [
        {
            "command": ["/bin/sh", "-c", "tool-from-prepared-path --version"],
            "cwd": workspace_path,
            "stdin_text": None,
            "stdout_path": task_dir / "runtime-setup-1.stdout.log",
            "stderr_path": task_dir / "runtime-setup-1.stderr.log",
            "timeout": 30,
            "env": {"PATH": "/prepared/bin:/usr/bin:/bin"},
        }
    ]


def test_runtime_setup_command_failure_records_command_timing(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    workspace_path.mkdir()
    task_dir.mkdir()
    times = iter([100.0, 106.25])

    class FakeRuntime:
        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del command, cwd, stdin_text, timeout, env, host_runner
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("failed", encoding="utf-8")
            return {"ok": False, "exit_code": 9, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.time.time", lambda: next(times))

    failure = run_runtime_setup_commands(
        FakeRuntime(),
        commands=["exit 9"],
        workspace_path=workspace_path,
        task_dir=task_dir,
        timeout=30,
        env=None,
    )

    assert failure is not None
    assert failure.started_at == 100.0
    assert failure.completed_at == 106.25


def test_docker_task_runtime_starts_execs_and_cleans_container(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    schema_dir = tmp_path / "schema"
    workspace_path.mkdir()
    task_dir.mkdir()
    schema_dir.mkdir()
    schema_path = schema_dir / "output.schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="container-id\n", stderr="")
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["docker", "image", "inspect", "--format"]:
            return subprocess.CompletedProcess(command, 0, stdout="sha256:test-image\n", stderr="")
        if command[:2] == ["git", "-C"]:
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="not a git repository")
        raise AssertionError(f"unexpected command: {command}")

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(list(command))
            kwargs["stdout"].write("agent stdout")
            kwargs["stderr"].write("agent stderr")

        def communicate(self, input=None, timeout=None):
            return None

    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.run", fake_run)
    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.Popen", FakeProcess)

    runtime = DockerTaskRuntime(
        config=RuntimeBackendConfig(
            backend="docker",
            image="contextbench-agent:test",
            platform="linux/amd64",
            env={"BASE_ENV": "1"},
        ),
        workspace_path=workspace_path,
        task_dir=task_dir,
        schema_path=schema_path,
    )

    result = runtime.run_command(
        ["codex", "exec", "-"],
        cwd=workspace_path,
        stdin_text="prompt",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=30,
        env={"HOME": str(task_dir / "home")},
    )
    runtime.close(success=True)

    docker_run = next(call for call in calls if call[:2] == ["docker", "run"])
    docker_exec = next(call for call in calls if call[:2] == ["docker", "exec"])
    docker_rm = next(call for call in calls if call[:3] == ["docker", "rm", "--force"])

    assert result == {"ok": True, "exit_code": 0, "signal": None, "timeout": False}
    assert stdout_path.read_text(encoding="utf-8") == "agent stdout"
    assert stderr_path.read_text(encoding="utf-8") == "agent stderr"
    assert docker_run[:4] == ["docker", "run", "--detach", "--name"]
    assert "--workdir" in docker_run
    assert_subsequence(docker_run, ["--add-host", "host.docker.internal:host-gateway"])
    assert_subsequence(docker_run, ["--platform", "linux/amd64"])
    assert str(workspace_path) in docker_run
    assert f"type=bind,source={workspace_path.resolve()},target={workspace_path.resolve()}" in docker_run
    assert f"type=bind,source={task_dir.resolve()},target={task_dir.resolve()}" in docker_run
    assert f"type=bind,source={schema_dir.resolve()},target={schema_dir.resolve()},readonly" in docker_run
    assert "contextbench-agent:test" in docker_run
    assert docker_exec[:3] == ["docker", "exec", "-i"]
    if hasattr(__import__("os"), "getuid"):
        assert "--user" in docker_exec
    assert "--env" in docker_exec
    assert "BASE_ENV=1" in docker_exec
    assert f"HOME={task_dir / 'home'}" in docker_exec
    assert_subsequence(docker_exec, ["timeout", "--foreground", "--kill-after", "10s", "30s"])
    assert docker_exec[-3:] == ["codex", "exec", "-"]
    assert docker_rm[:3] == ["docker", "rm", "--force"]


def test_docker_container_bootstrap_creates_sudoers_dir() -> None:
    command = _container_bootstrap_command("501:20")

    assert "mkdir -p /etc/sudoers.d" in command
    assert command.index("mkdir -p /etc/sudoers.d") < command.index("> /etc/sudoers.d/contextbench")


def test_docker_task_runtime_does_not_treat_dead_container_124_as_timeout(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    task_dir = tmp_path / "task"
    workspace_path.mkdir()
    task_dir.mkdir()
    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"

    def fake_run(command, **kwargs):
        del kwargs
        if command[:4] == ["docker", "inspect", "--format", "{{.State.Running}}"]:
            return subprocess.CompletedProcess(command, 0, stdout="false\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    class FakeProcess:
        returncode = 124

        def __init__(self, command, **kwargs):
            del command, kwargs

        def communicate(self, input=None, timeout=None):
            del input, timeout
            return None

    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.run", fake_run)
    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.Popen", FakeProcess)

    runtime = DockerTaskRuntime(
        config=RuntimeBackendConfig(backend="docker", image="contextbench-agent:test"),
        workspace_path=workspace_path,
        task_dir=task_dir,
        schema_path=None,
        container_name="contextbench-dead",
        _started=True,
    )

    result = runtime.run_command(
        ["codex", "--version"],
        cwd=workspace_path,
        stdin_text=None,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=30,
    )

    assert result == {"ok": False, "exit_code": 124, "signal": None, "timeout": False}
    assert runtime._timed_out is False

def test_docker_task_runtime_mounts_linked_worktree_git_metadata(tmp_path) -> None:
    base_repo = tmp_path / "base"
    worktree = tmp_path / "worktree"
    task_dir = tmp_path / "task"
    base_repo.mkdir()
    task_dir.mkdir()
    subprocess.run(["git", "init"], cwd=base_repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=base_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=base_repo, check=True)
    (base_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=base_repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=base_repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=base_repo, check=True, capture_output=True, text=True)

    runtime = DockerTaskRuntime(
        config=RuntimeBackendConfig(backend="docker", image="contextbench-agent:test"),
        workspace_path=worktree,
        task_dir=task_dir,
        schema_path=None,
    )

    mounts = runtime._mounts()
    mounted_sources = {source for source, _target, _readonly in mounts}

    assert str(worktree.resolve()) in mounted_sources
    assert str((base_repo / ".git").resolve()) in mounted_sources

def test_docker_task_runtime_stops_kept_container_after_timeout(tmp_path, monkeypatch) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.run", fake_run)

    runtime = DockerTaskRuntime(
        config=RuntimeBackendConfig(backend="docker", image="contextbench-agent:test", keep_failed=True),
        workspace_path=tmp_path,
        task_dir=task_dir,
        schema_path=None,
        container_name="contextbench-timeout",
        _started=True,
        _timed_out=True,
    )

    runtime.close(success=False)

    assert calls == [["docker", "stop", "--time", "1", "contextbench-timeout"]]
    assert (task_dir / "docker-container.txt").read_text(encoding="utf-8") == "contextbench-timeout\n"
