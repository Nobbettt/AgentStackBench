
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import types
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

_REPO_ROOT = Path(__file__).resolve().parents[2]


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"


def test_claude_wrapper_uses_host_runtime(monkeypatch, tmp_path) -> None:
    wrapper_path = _REPO_ROOT / "agent-frameworks" / "claude-code" / "run_bench.py"
    spec = importlib.util.spec_from_file_location("claude_code_run_bench_test", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict[str, object] = {}

    def fake_run_coding_agent_task(**kwargs):
        captured.update(kwargs)
        return {"status": "completed", "task_dir": str(tmp_path / "task")}

    monkeypatch.setattr(module, "run_coding_agent_task", fake_run_coding_agent_task)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            schema=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
            model=None,
            agent_arg=[],
            runtime_env=[],
        ),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "bench": "Verified",
                    "instance_id": "task-1",
                    "repo_url": "https://github.com/example/repo.git",
                    "commit": "abc123",
                    "prompt": "Fix the bug.",
                }
            )
        ),
    )

    assert module.main() == 0
    assert captured["agent"] == "claude"
    assert captured["runtime_backend"] == "host"


def test_claude_command_uses_verbose_json_mode(tmp_path, schema_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=schema_path,
        prompt="test prompt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--verbose" in command
    assert command[:4] == ["claude", "--print", "--output-format", "json"]
    assert "--settings" in command
    assert "--mcp-config" in command
    assert "--setting-sources" in command
    assert "--disable-slash-commands" in command
    assert "--strict-mcp-config" in command
    permission_index = command.index("--permission-mode")
    assert command[permission_index + 1] == "auto"

def test_claude_command_omits_schema_when_not_requested(tmp_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=None,
        prompt="bootstrap prompt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--json-schema" not in command

def test_codex_command_uses_json_event_mode(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort=None,
        extra_args=[],
    )

    assert "--json" in command
    assert "--verbose" not in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"

def test_codex_command_accepts_docker_sandbox_mode(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort=None,
        sandbox_mode="danger-full-access",
        extra_args=[],
    )

    assert command[command.index("--sandbox") + 1] == "danger-full-access"

def test_codex_command_omits_schema_when_not_requested(tmp_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=None,
        final_output_path=tmp_path / "setup-last-message.txt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
    )

    assert "--output-schema" not in command
    assert "--output-last-message" in command

def test_codex_command_includes_add_dir_for_runtime_root(tmp_path, schema_path) -> None:
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir()

    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort=None,
        writable_dirs=[runtime_root],
        extra_args=[],
    )

    assert "--add-dir" in command
    assert str(runtime_root.resolve()) in command

def test_claude_command_maps_xhigh_reasoning_effort_to_max(tmp_path, schema_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=schema_path,
        prompt="test prompt",
        model=None,
        reasoning_effort="max",
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--effort" in command
    assert "max" in command

def test_codex_command_includes_reasoning_effort_override(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort="xhigh",
        extra_args=[],
    )

    assert "-c" in command
    assert 'model_reasoning_effort="xhigh"' in command
