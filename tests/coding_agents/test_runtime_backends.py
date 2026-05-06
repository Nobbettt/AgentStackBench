
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

def test_run_command_timeout_decodes_byte_output(tmp_path, monkeypatch) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout", 30),
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr("contextbench.coding_agents.runtime.subprocess.run", fake_run)

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
        if command[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(command, 0, stdout="agent stdout", stderr="agent stderr")
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["docker", "image", "inspect", "--format"]:
            return subprocess.CompletedProcess(command, 0, stdout="sha256:test-image\n", stderr="")
        if command[:2] == ["git", "-C"]:
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="not a git repository")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.subprocess.run", fake_run)

    runtime = DockerTaskRuntime(
        config=RuntimeBackendConfig(
            backend="docker",
            image="contextbench-agent:test",
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
