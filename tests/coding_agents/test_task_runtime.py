
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import find_private_path_matches
from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CLAUDE_RUNTIME_IMAGE,
)
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
from contextbench.coding_agents.runtime_backends import RuntimeBackendConfig
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.claude.runtime import runtime_root as claude_runtime_root
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"


def test_run_coding_agent_task_requires_explicit_repo_url(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    checkout_calls: list[object] = []
    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.checkout",
        lambda *args, **kwargs: checkout_calls.append((args, kwargs)) or str(tmp_path / "workspace"),
    )

    with pytest.raises(RuntimeError, match="missing required repo_url"):
        run_coding_agent_task(
            task={
                "bench": "Verified",
                "instance_id": "example__repo-1",
                "original_inst_id": "example__repo-1",
                "commit": "abc123",
                "prompt": "Fix the bug.",
                "language": "python",
            },
            agent="codex",
            output_dir=Path("results"),
            cache_dir=Path("cache"),
            schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
            runtime_backend="host",
        )

    assert checkout_calls == []


def test_run_coding_agent_task_requires_explicit_runtime_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="runtime_backend is required"):
        run_coding_agent_task(
            task={
                "bench": "Verified",
                "instance_id": "example__repo-1",
                "original_inst_id": "example__repo-1",
                "repo_url": "https://github.com/example/repo.git",
                "commit": "abc123",
                "prompt": "Fix the bug.",
                "language": "python",
            },
            agent="codex",
            output_dir=Path("results"),
            cache_dir=Path("cache"),
            schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
        )


def test_run_coding_agent_task_claude_docker_uses_default_image_and_runtime_auth_validation(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    source_claude_dir = tmp_path / "source-claude"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"abc"}', encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(source_claude_dir))
    monkeypatch.setenv("PATH", f"/host-only/bin:{os.environ.get('PATH', '')}")

    task = {
        "bench": "Verified",
        "instance_id": "example__repo-1",
        "original_inst_id": "example__repo-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"commands": [], "envs": []}

    class FakeDockerRuntime:
        def __init__(self, config) -> None:
            self.config = config
            self.closed = False

        def start(self) -> None:
            return None

        def metadata(self) -> dict[str, object]:
            return {"backend": self.config.backend, "image": self.config.image}

        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del host_runner
            captured["commands"].append(list(command))
            captured["envs"].append(dict(env or {}))
            captured["cwd"] = cwd
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text("", encoding="utf-8")
            if list(command) == ["claude", "auth", "status", "--json"]:
                stdout_path.write_text('{"loggedIn":true}\n', encoding="utf-8")
                return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}
            assert stdin_text is not None
            stdout_path.write_text(
                json.dumps(
                    [
                        {
                            "type": "system",
                            "subtype": "init",
                            "cwd": str(workspace_path),
                            "session_id": "fixture-session",
                            "tools": ["Bash", "Read", "StructuredOutput"],
                            "mcp_servers": [],
                            "model": "claude-opus-4-7",
                            "permissionMode": "default",
                            "slash_commands": [],
                            "apiKeySource": "none",
                            "claude_code_version": "2.1.126",
                            "output_style": "default",
                            "agents": ["general-purpose"],
                            "skills": [],
                            "plugins": [],
                            "uuid": "fixture-init",
                            "fast_mode_state": "off",
                        },
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "duration_ms": 25,
                            "duration_api_ms": 10,
                            "num_turns": 1,
                            "result": json.dumps(
                                make_final_output(
                                    task_id="example__repo-1",
                                    touched_files=["a.py"],
                                    retrieved_context_files=["a.py"],
                                )
                            ),
                            "stop_reason": "end_turn",
                            "session_id": "fixture-session",
                            "usage": {"input_tokens": 8, "output_tokens": 3},
                            "modelUsage": {},
                            "permission_denials": [],
                            "fast_mode_state": "off",
                            "uuid": "fixture-result",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

        def close(self, *, success: bool) -> None:
            captured["close_success"] = success
            self.closed = True

    captured_runtime_kwargs: dict[str, object] = {}

    def fake_create_task_runtime(config, **kwargs):
        captured["runtime_config"] = config
        captured_runtime_kwargs.update(kwargs)
        return FakeDockerRuntime(config)

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.create_task_runtime", fake_create_task_runtime)

    record = run_coding_agent_task(
        task=task,
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="docker",
        runtime_image=None,
    )

    task_dir = (Path("results") / "example__repo-1").resolve()
    runtime_config = captured["runtime_config"]
    assert runtime_config.image == DEFAULT_CLAUDE_RUNTIME_IMAGE
    assert captured_runtime_kwargs["extra_writable_dirs"] == [claude_runtime_root(task_dir)]
    assert len(captured["commands"]) == 2
    assert captured["commands"][0] == ["claude", "auth", "status", "--json"]
    assert captured["commands"][1][0] == "claude"
    assert "--json-schema" in captured["commands"][1]
    assert all("claude-auth-canary" not in str(part) for command in captured["commands"] for part in command)
    permission_index = captured["commands"][1].index("--permission-mode")
    assert captured["commands"][1][permission_index + 1] == "bypassPermissions"
    assert captured["envs"][0]["CONTEXTBENCH_WORKSPACE_PATH"] == str(workspace_path)
    assert "/host-only/bin" not in captured["envs"][0]["PATH"]
    assert "/usr/local/bin" in captured["envs"][0]["PATH"].split(":")
    assert record["status"] == "completed"
    assert record["runtime"]["image"] == DEFAULT_CLAUDE_RUNTIME_IMAGE
    assert captured["close_success"] is True


def test_run_coding_agent_task_claude_passes_workspace_path_to_mcp_runtime(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = Path("results")
    cache_dir = Path("cache")
    schema_path = CLAUDE_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "claude-cortex-task",
        "original_inst_id": "claude-cortex-task",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["stdin_text"] = stdin_text
        captured["env"] = dict(env or {})
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "cwd": str(workspace_path),
                        "session_id": "fixture-session",
                        "tools": ["Bash", "Read", "StructuredOutput"],
                        "mcp_servers": ["cortex"],
                        "model": "claude-opus-4-7",
                        "permissionMode": "default",
                        "slash_commands": [],
                        "apiKeySource": "none",
                        "claude_code_version": "2.1.117",
                        "output_style": "default",
                        "agents": ["general-purpose"],
                        "skills": [],
                        "plugins": [],
                        "uuid": "fixture-init",
                        "fast_mode_state": "off",
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "duration_ms": 25,
                        "duration_api_ms": 10,
                        "num_turns": 1,
                        "result": json.dumps(
                            make_final_output(
                                task_id="claude-cortex-task",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "stop_reason": "end_turn",
                        "session_id": "fixture-session",
                        "usage": {"input_tokens": 8, "output_tokens": 3},
                        "modelUsage": {},
                        "permission_denials": [],
                        "fast_mode_state": "off",
                        "uuid": "fixture-result",
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="claude",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        model="claude-opus-4-7",
        runtime_backend="host",
        setup={
            "claude_mcp_config": {
                "mcpServers": {
                    "cortex": {
                        "command": "cortex",
                        "args": ["mcp"],
                        "env": {
                            "CORTEX_PROJECT_ROOT": "${CONTEXTBENCH_WORKSPACE_PATH}",
                        },
                    }
                }
            }
        },
    )

    assert record["status"] == "completed"
    assert captured["cwd"] == workspace_path
    assert "Fix the bug." in str(captured["stdin_text"])
    assert all("Fix the bug." not in str(item) for item in captured["command"])
    assert captured["env"]["CONTEXTBENCH_WORKSPACE_PATH"] == str(workspace_path)
    assert captured["env"]["CONTEXTBENCH_TASK_DIR"] == str((output_dir / "claude-cortex-task").resolve())
    mcp_config = json.loads(((output_dir / "claude-cortex-task").resolve() / "claude.mcp.json").read_text(encoding="utf-8"))
    assert mcp_config["mcpServers"]["cortex"]["env"]["CORTEX_PROJECT_ROOT"] == str(workspace_path)


def test_run_coding_agent_task_records_and_requires_generic_mcp_tool_calls(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    task = {
        "bench": "Verified",
        "instance_id": "claude-mcp-required",
        "original_inst_id": "claude-mcp-required",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "plugins": [],
                        "mcp_servers": ["cortex"],
                        "slash_commands": [],
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "mcp__cortex__search",
                                    "input": {"query": "target"},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "context",
                                }
                            ]
                        },
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="claude-mcp-required",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        setup={"claude_mcp_config": {"mcpServers": {"cortex": {"command": "cortex", "args": ["mcp"]}}}},
        required_tool_call_patterns=[r"^mcp__cortex__"],
    )

    assert record["status"] == "completed"
    assert record["tool_call_summary"]["mcp_total"] == 1
    assert record["tool_call_summary"]["mcp_successful_total"] == 1
    assert record["tool_call_summary"]["mcp_by_server"] == {"cortex": 1}
    assert record["tool_call_summary"]["mcp_successful_by_server"] == {"cortex": 1}
    assert record["tool_call_requirements"]["ok"] is True


def test_run_coding_agent_task_requires_available_tools_without_requiring_use(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": ["Read", "mcp__cortex__search"],
                        "plugins": [],
                        "mcp_servers": ["cortex"],
                        "slash_commands": [],
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Read",
                                    "input": {"file_path": "a.py"},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "context",
                                }
                            ]
                        },
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="available-tool-unused",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "available-tool-unused",
            "original_inst_id": "available-tool-unused",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        setup={"claude_mcp_config": {"mcpServers": {"cortex": {"command": "cortex", "args": ["mcp"]}}}},
        required_available_tool_patterns=[r"^mcp__cortex__"],
    )

    assert record["status"] == "completed"
    assert record["available_tools"] == ["Read", "mcp__cortex__search"]
    assert record["tool_call_summary"]["mcp_total"] == 0
    assert record["tool_availability_requirements"] == {
        "patterns": [r"^mcp__cortex__"],
        "missing": [],
        "ok": True,
    }


def test_run_coding_agent_task_fails_when_required_available_tool_missing(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": ["Read", "Bash"],
                        "plugins": [],
                        "mcp_servers": [],
                        "slash_commands": [],
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="missing-available-tool",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "missing-available-tool",
            "original_inst_id": "missing-available-tool",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        required_available_tool_patterns=[r"^mcp__cortex__"],
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["available_tools"] == ["Read", "Bash"]
    assert record["tool_availability_requirements"] == {
        "patterns": [r"^mcp__cortex__"],
        "missing": [r"^mcp__cortex__"],
        "ok": False,
    }


def test_run_coding_agent_task_records_adapter_validation_failure_proof(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    class FakeDockerRuntime:
        config = type("Config", (), {"backend": "docker"})()

        def __init__(self) -> None:
            self.closed_success: bool | None = None

        def start(self) -> None:
            return None

        def metadata(self) -> dict[str, object]:
            return {"backend": "docker", "image": "fake-claude"}

        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del command, cwd, stdin_text, timeout, env, host_runner
            stdout_path.write_text('{"loggedIn":false}\n', encoding="utf-8")
            stderr_path.write_text("not logged in\n", encoding="utf-8")
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

        def close(self, *, success: bool) -> None:
            self.closed_success = success

    fake_runtime = FakeDockerRuntime()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.create_task_runtime", lambda *args, **kwargs: fake_runtime)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    auth_times = iter([100.0, 104.5])
    monkeypatch.setattr("contextbench.agents.claude.runtime.time.time", lambda: next(auth_times))
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.run_command",
        lambda *args, **kwargs: pytest.fail("scored prompt should not run after adapter validation failure"),
    )

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-adapter-validation-fail",
            "original_inst_id": "claude-adapter-validation-fail",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="docker",
        runtime_env={"ANTHROPIC_API_KEY": "configured-key"},
    )

    task_dir = (Path("results") / "claude-adapter-validation-fail").resolve()
    assert record["status"] == "failed"
    assert record["duration_ms"] == 4500
    assert record["notes"] == "Adapter runtime validation failed before setup prompts or scored work."
    assert record["runtime_failure"] == {
        "phase": "adapter-validation",
        "command": "claude auth status --json",
        "stdout_path": str(task_dir / "claude-auth-status.stdout.log"),
        "stderr_path": str(task_dir / "claude-auth-status.stderr.log"),
    }
    assert fake_runtime.closed_success is False


def test_run_coding_agent_task_codex_docker_preflight_failure_short_circuits(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    calls: list[list[str]] = []

    class FakeRuntime:
        config = RuntimeBackendConfig(backend="docker", image="fake")

        def __init__(self) -> None:
            self.closed_success: bool | None = None

        def start(self) -> None:
            return None

        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del cwd, stdin_text, timeout, env, host_runner
            calls.append(list(command))
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("codex: command not found\n", encoding="utf-8")
            return {"ok": False, "exit_code": 127, "signal": None, "timeout": False}

        def close(self, *, success: bool) -> None:
            self.closed_success = success

    fake_runtime = FakeRuntime()
    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.create_task_runtime", lambda *args, **kwargs: fake_runtime)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_workspace_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.run_command",
        lambda *args, **kwargs: pytest.fail("scored prompt should not run after Codex preflight failure"),
    )

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "codex-preflight-fail",
            "original_inst_id": "codex-preflight-fail",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="docker",
        runtime_image="fake",
    )

    task_dir = (output_dir / "codex-preflight-fail").resolve()
    assert calls == [["codex", "--version"]]
    assert record["status"] == "failed"
    assert record["runtime_failure"] == {
        "phase": "adapter-validation",
        "command": "codex --version",
        "stdout_path": str(task_dir / "codex-version.stdout.log"),
        "stderr_path": str(task_dir / "codex-version.stderr.log"),
    }
    assert record["notes"] == "Adapter runtime validation failed before setup prompts or scored work."
    assert fake_runtime.closed_success is False


def test_run_coding_agent_task_requires_successful_generic_mcp_tool_calls(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "plugins": [],
                        "mcp_servers": ["cortex"],
                        "slash_commands": [],
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "mcp__cortex__search",
                                    "input": {"query": "target"},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "is_error": True,
                                    "content": "Claude requested permissions to use mcp__cortex__search, but you haven't granted it yet.",
                                }
                            ]
                        },
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="denied-required-tool",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "denied-required-tool",
            "original_inst_id": "denied-required-tool",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        setup={"claude_mcp_config": {"mcpServers": {"cortex": {"command": "cortex", "args": ["mcp"]}}}},
        required_tool_call_patterns=[r"^mcp__cortex__"],
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["tool_call_summary"]["mcp_total"] == 1
    assert record["tool_call_summary"]["mcp_successful_total"] == 0
    assert record["tool_call_summary"]["mcp_failed_total"] == 1
    assert record["tool_call_requirements"]["missing"] == [r"^mcp__cortex__"]


def test_run_coding_agent_task_fails_when_required_tool_call_missing(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "plugins": [],
                        "mcp_servers": [],
                        "slash_commands": [],
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="missing-required-tool",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "missing-required-tool",
            "original_inst_id": "missing-required-tool",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        required_tool_call_patterns=[r"^mcp__cortex__"],
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["tool_call_requirements"]["missing"] == [r"^mcp__cortex__"]


def test_run_coding_agent_task_codex_writes_record_and_diff(tmp_path, monkeypatch, make_final_output) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = Path("results")
    cache_dir = Path("cache")
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-1",
        "original_inst_id": "task-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    reset_calls: list[Path] = []

    def fake_reset_workspace(path: Path) -> None:
        reset_calls.append(path)
        setup_marker = path / "setup-ran.txt"
        if setup_marker.exists():
            setup_marker.unlink()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", fake_reset_workspace)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"])
            and captured.setdefault("reasoning_effort", kwargs["reasoning_effort"])
            and captured.setdefault("writable_dirs", kwargs["writable_dirs"])
            and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "diff --git a/a.py b/a.py\n")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["stdin_text"] = stdin_text
        captured["env"] = env
        stdout_path.write_text(
            json.dumps({"type": "message", "message": f"opened {workspace_path / 'a.py'}"}) + "\n"
            + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-1",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                    final_answer=f"checked {workspace_path / 'a.py'}",
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        reasoning_effort="high",
        env_overrides={"EXPERIMENT": "1"},
        prompt_preamble="Variant instructions",
        runtime_backend="host",
    )

    task_dir = (tmp_path / "results" / "task-1").resolve()
    record_path = task_dir / "task-1.codex-record.json"
    public_record_path = task_dir / "task-1.codex-record.public.json"

    assert record["agent"] == "codex"
    assert record["final_output"]["status"] == "completed"
    assert "task_id" not in record["final_output"]
    assert record["tool_calls"] == []
    assert record["model_patch"].startswith("diff --git")
    assert Path(record["raw_response_path"]).exists()
    assert Path(record["diff_path"]).exists()
    assert record_path.exists()
    assert public_record_path.exists()
    assert str(workspace_path / "a.py") in (task_dir / "codex-events.jsonl").read_text(encoding="utf-8")
    assert str(workspace_path / "a.py") in (task_dir / "final-output.json").read_text(encoding="utf-8")
    assert str(workspace_path / "a.py") in (task_dir / "raw-response.json").read_text(encoding="utf-8")
    assert find_private_path_matches(public_record_path.read_text(encoding="utf-8")) == []
    assert "<task-artifacts>" in public_record_path.read_text(encoding="utf-8")
    prompt_text = (task_dir / "prompt.txt").read_text(encoding="utf-8")
    assert prompt_text.startswith("Variant instructions")
    assert "Consider the following PR description:" in prompt_text
    assert "Work inside the checked-out repository workspace for this task." in prompt_text
    assert "Do not add extra bookkeeping fields beyond the required schema." in prompt_text
    assert captured["reasoning_effort"] == "high"
    assert isinstance(captured["final_output_path"], Path)
    assert captured["final_output_path"].is_absolute()
    assert [str(path) for path in captured["writable_dirs"]] == [str(codex_runtime_root(task_dir).resolve())]
    assert Path(record["task_dir"]).is_absolute()
    assert captured["cwd"] == workspace_path
    assert captured["env"] == {
        "HOME": str(task_dir),
        "EXPERIMENT": "1",
        "CONTEXTBENCH_WORKSPACE_PATH": str(workspace_path),
        "CONTEXTBENCH_TASK_DIR": str(task_dir),
    }


def test_run_coding_agent_task_rejects_schema_invalid_structured_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"]) and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        stdout_path.write_text(json.dumps({"type": "turn.completed"}) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "final_answer": "done",
                    "retrieved_context_files": "a.py",
                    "retrieved_context_spans": [],
                    "retrieved_context_symbols": [],
                    "notes": "",
                    "task_id": "schema-invalid",
                }
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "schema-invalid",
            "original_inst_id": "schema-invalid",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="codex",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["final_output"] is None
    assert "Structured output failed schema validation" in record["notes"]
    assert "unexpected property 'task_id'" in record["notes"]


def test_run_coding_agent_task_codex_retries_transient_failure(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-retry",
        "original_inst_id": "task-retry",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"attempt": 0, "final_output_path": None}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.__setitem__("final_output_path", kwargs["final_output_path"]) or ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_workspace_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.agents.codex.runtime.time.sleep", lambda seconds: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.workspace_has_nonexcluded_changes", lambda *args, **kwargs: False)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["attempt"] = int(captured["attempt"]) + 1
        attempt = int(captured["attempt"])
        if attempt == 1:
            stdout_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "thread.started", "thread_id": "t-1"}),
                        json.dumps({"type": "turn.started"}),
                        json.dumps({"type": "error", "message": "Rate limit exceeded while starting Codex turn"}),
                        json.dumps({"type": "turn.failed", "error": {"message": "unexpected status 401 Unauthorized: Missing bearer or basic authentication in header"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text(
                "rate limit while connecting to Codex transport\n",
                encoding="utf-8",
            )
            return {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-retry",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
    )

    task_dir = (output_dir / "task-retry").resolve()

    assert captured["attempt"] == 2
    assert record["status"] == "completed"
    assert record["ok"] is True
    assert record["retry"]["attempts"] == 2
    assert record["retry"]["retried"] is True
    assert record["retry"]["suppressed"] is False
    assert record["retry"]["events"][0]["action"] == "retry"
    assert record["token_usage"]["input_tokens"] == 5
    assert (task_dir / "stderr.attempt1.log").exists()
    assert (task_dir / "raw-response.attempt1.json").exists()
    assert (task_dir / "codex-events.attempt1.jsonl").exists()
    assert "Rate limit exceeded" in (task_dir / "raw-response.attempt1.json").read_text(encoding="utf-8")


def test_run_coding_agent_task_claude_retries_transient_failure_and_preserves_raw_artifacts(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CLAUDE_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "claude-retry",
        "original_inst_id": "claude-retry",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"attempt": 0}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)
    monkeypatch.setattr("contextbench.agents.claude.runtime.time.sleep", lambda seconds: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.workspace_has_nonexcluded_changes", lambda *args, **kwargs: False)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["attempt"] = int(captured["attempt"]) + 1
        attempt = int(captured["attempt"])
        if attempt == 1:
            stdout_path.write_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Rate limit while reading {workspace_path / 'a.py'}",
                    }
                ),
                encoding="utf-8",
            )
            stderr_path.write_text(f"temporarily unavailable in {workspace_path}\n", encoding="utf-8")
            return {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

        stdout_path.write_text(
            json.dumps(
                [
                    {
                        "type": "system",
                        "subtype": "init",
                        "plugins": [],
                        "mcp_servers": [],
                        "slash_commands": [],
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": json.dumps(
                            make_final_output(
                                task_id="claude-retry",
                                touched_files=["a.py"],
                                retrieved_context_files=["a.py"],
                            )
                        ),
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    },
                ]
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="claude",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
    )

    task_dir = (output_dir / "claude-retry").resolve()
    attempt_record = (task_dir / "raw-response.attempt1.json").read_text(encoding="utf-8")

    assert captured["attempt"] == 2
    assert record["status"] == "completed"
    assert record["retry"]["attempts"] == 2
    assert record["retry"]["retried"] is True
    assert record["retry"]["suppressed"] is False
    assert record["retry"]["events"][0]["action"] == "retry"
    assert record["token_usage"]["input_tokens"] == 6
    assert (task_dir / "stderr.attempt1.log").exists()
    assert (task_dir / "claude-output.attempt1.jsonl").exists()
    assert str(workspace_path / "a.py") in attempt_record


def test_run_coding_agent_task_suppresses_retry_when_failed_attempt_modifies_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CLAUDE_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "claude-dirty-retry",
        "original_inst_id": "claude-dirty-retry",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured = {"attempt": 0}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)
    monkeypatch.setattr("contextbench.agents.claude.runtime.time.sleep", lambda seconds: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.workspace_has_nonexcluded_changes", lambda *args, **kwargs: True)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["attempt"] += 1
        if captured["attempt"] > 1:
            raise AssertionError("dirty failed attempts must not be retried")
        stdout_path.write_text(
            json.dumps({"type": "error", "message": "Rate limit after partial edit"}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="claude",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
    )

    assert captured["attempt"] == 1
    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["retry"]["attempts"] == 1
    assert record["retry"]["retried"] is False
    assert record["retry"]["suppressed"] is True
    assert record["retry"]["suppression_reason"] == "workspace_dirty_after_failed_attempt"
    assert record["retry"]["events"][0]["action"] == "suppressed"
    assert "Retry suppressed" in record["notes"]


def test_run_coding_agent_task_claude_stream_json_records_metadata(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        assert command[command.index("--output-format") + 1] == "stream-json"
        assert stdout_path.name == "claude-output.jsonl"
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read", "mcp__cortex__context_get_rules"],
                            "plugins": [],
                            "mcp_servers": ["cortex"],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_1",
                                        "name": "Read",
                                        "input": {"file_path": "a.py"},
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "content": "context",
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": json.dumps(
                                make_final_output(
                                    task_id="claude-stream-json",
                                    touched_files=["a.py"],
                                    retrieved_context_files=["a.py"],
                                )
                            ),
                            "usage": {"input_tokens": 6, "output_tokens": 2},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-stream-json",
            "original_inst_id": "claude-stream-json",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        setup={"claude_mcp_config": {"mcpServers": {"cortex": {"command": "cortex", "args": ["mcp"]}}}},
        required_available_tool_patterns=[r"^mcp__cortex__"],
    )

    raw_response = json.loads(Path(record["raw_response_path"]).read_text(encoding="utf-8"))

    assert record["status"] == "completed"
    assert record["available_tools"] == ["Read", "mcp__cortex__context_get_rules"]
    assert record["tool_call_summary"]["mcp_total"] == 0
    assert record["tool_availability_requirements"]["ok"] is True
    assert raw_response["response_format"] == "stream-json"


def test_run_coding_agent_task_claude_isolation_failure_writes_failed_record(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read"],
                            "plugins": [],
                            "mcp_servers": [],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": json.dumps(
                                make_final_output(
                                    task_id="claude-isolation-failure",
                                    touched_files=["a.py"],
                                    retrieved_context_files=["a.py"],
                                )
                            ),
                            "usage": {"input_tokens": 6, "output_tokens": 2},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-isolation-failure",
            "original_inst_id": "claude-isolation-failure",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
        setup={"claude_mcp_config": {"mcpServers": {"cortex": {"command": "cortex", "args": ["mcp"]}}}},
    )

    task_dir = (Path("results") / "claude-isolation-failure").resolve()
    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["exit_code"] == 1
    assert "missing expected MCP servers: cortex" in record["notes"]
    assert Path(record["raw_response_path"]).exists()
    assert (task_dir / "claude-isolation-failure.claude-record.json").exists()


def test_validate_claude_isolation_rejects_failed_configured_mcp_server() -> None:
    with pytest.raises(RuntimeError, match="configured MCP servers are not connected"):
        validate_claude_isolation(
            {
                "response": [
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": ["Read"],
                        "plugins": [],
                        "mcp_servers": [{"name": "cortex", "status": "failed"}],
                        "slash_commands": [],
                    }
                ]
            },
            allowed_mcp_servers={"cortex"},
        )


def test_run_coding_agent_task_claude_archives_persisted_tool_results(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = Path("results")
    task_dir = (output_dir / "claude-persisted-tool-result").resolve()
    persisted_path = claude_runtime_root(task_dir) / "tool-results" / "toolu_1.txt"

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        persisted_path.parent.mkdir(parents=True)
        persisted_path.write_text(f"full output from {workspace_path / 'a.py'}\n", encoding="utf-8")
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read"],
                            "plugins": [],
                            "mcp_servers": [],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "content": (
                                            "<persisted-output>\n"
                                            f"Output too large (20KB). Full output saved to: {persisted_path}\n\n"
                                            "Preview (first 2KB):\npreview\n</persisted-output>"
                                        ),
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": json.dumps(
                                make_final_output(
                                    task_id="claude-persisted-tool-result",
                                    touched_files=["a.py"],
                                    retrieved_context_files=["a.py"],
                                )
                            ),
                            "usage": {"input_tokens": 6, "output_tokens": 2},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-persisted-tool-result",
            "original_inst_id": "claude-persisted-tool-result",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=output_dir,
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
    )

    persisted = record["persisted_tool_results"][0]
    artifact_path = Path(str(persisted["artifact_path"]))

    assert record["status"] == "completed"
    assert persisted["status"] == "archived"
    assert persisted["label"] == "20KB"
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == f"full output from {workspace_path / 'a.py'}\n"
    assert (Path(record["task_dir"]) / "claude-persisted-tool-results.json").exists()


def test_run_coding_agent_task_claude_rejects_unsafe_persisted_tool_result_source(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    unsafe_path = tmp_path / "outside-runtime" / "toolu_1.txt"
    unsafe_path.parent.mkdir(parents=True)
    unsafe_path.write_text("secret\n", encoding="utf-8")

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read"],
                            "plugins": [],
                            "mcp_servers": [],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "content": (
                                            "<persisted-output>\n"
                                            f"Output too large (20KB). Full output saved to: {unsafe_path}\n"
                                            "</persisted-output>"
                                        ),
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": json.dumps(make_final_output(task_id="claude-unsafe-persisted")),
                            "usage": {"input_tokens": 6, "output_tokens": 2},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-unsafe-persisted",
            "original_inst_id": "claude-unsafe-persisted",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
    )

    persisted = record["persisted_tool_results"][0]

    assert persisted["status"] == "rejected_unsafe_source"
    assert persisted["artifact_path"] is None
    assert not (Path(record["task_dir"]) / "claude-persisted-tool-results").exists()


def test_run_coding_agent_task_claude_fails_when_structured_output_missing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "claude-home")},
    )
    monkeypatch.setattr("contextbench.agents.claude.runtime.validate_auth", lambda *, env=None: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "tools": ["Read"],
                            "plugins": [],
                            "mcp_servers": [],
                            "slash_commands": [],
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "error_max_structured_output_retries",
                            "is_error": True,
                            "result": "Could not produce valid structured output.",
                            "usage": {"input_tokens": 6, "output_tokens": 2},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.claude.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task={
            "bench": "Verified",
            "instance_id": "claude-missing-structured-output",
            "original_inst_id": "claude-missing-structured-output",
            "repo_url": "https://github.com/example/repo.git",
            "commit": "abc123",
            "prompt": "Fix the bug.",
            "language": "python",
        },
        agent="claude",
        output_dir=Path("results"),
        cache_dir=Path("cache"),
        schema_path=CLAUDE_OUTPUT_SCHEMA_PATH.resolve(),
        timeout=30,
        runtime_backend="host",
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["final_output"] is None
    assert "error_max_structured_output_retries" in record["notes"]


def test_run_coding_agent_task_passes_workspace_key_to_checkout(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-workspace-key",
        "original_inst_id": "task-workspace-key",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    def fake_checkout(*args, **kwargs):
        captured["workspace_key"] = kwargs.get("workspace_key")
        return str(workspace_path)

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", fake_checkout)
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"]) and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_workspace_diff", lambda path, **kwargs: "")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-workspace-key",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        workspace_key="suite__task__variant",
        runtime_backend="host",
    )

    assert captured["workspace_key"] == "suite__task__variant"
