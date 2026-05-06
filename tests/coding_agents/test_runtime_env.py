
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

def test_prepare_codex_runtime_env_copies_auth_only(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")
    (source_codex_dir / "config.toml").write_text('profile = "should-not-copy"\n', encoding="utf-8")

    env = prepare_codex_runtime_env(tmp_path / "task", source_codex_dir=source_codex_dir)

    isolated_home = Path(env["HOME"]) / ".codex"
    assert (isolated_home / "auth.json").exists()
    assert not (isolated_home / "config.toml").exists()
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert env["HOME"] != str(Path.home())

def test_prepare_codex_runtime_env_can_copy_auth_without_host_environment_for_docker(tmp_path, monkeypatch) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("PATH", "/host-only/bin")
    monkeypatch.setenv("SECRET_SHOULD_NOT_LEAK", "1")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        include_host_env=False,
    )

    assert "PATH" not in env
    assert "SECRET_SHOULD_NOT_LEAK" not in env
    assert "/.cache/agent-runtimes/codex/" in env["HOME"]
    assert env["HOME"].endswith("/home")
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert json.loads((Path(env["HOME"]) / ".codex" / "auth.json").read_text(encoding="utf-8")) == {"token": "host"}

def test_codex_prepare_runtime_uses_host_auth_without_host_env_for_docker_backend(tmp_path, monkeypatch) -> None:
    host_home = tmp_path / "host-home"
    source_codex_dir = host_home / ".codex"
    source_codex_dir.mkdir(parents=True)
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("SECRET_SHOULD_NOT_LEAK", "1")
    monkeypatch.setenv("PATH", "/host-only/bin")

    prepared = CodexAdapter().prepare_runtime(
        task_dir=tmp_path,
        setup={},
        env_overrides=None,
        runtime_backend="docker",
    )

    assert prepared.env is not None
    assert "SECRET_SHOULD_NOT_LEAK" not in prepared.env
    assert "PATH" not in prepared.env
    assert "/.cache/agent-runtimes/codex/" in prepared.env["HOME"]
    assert prepared.env["HOME"].endswith("/home")
    assert json.loads((Path(prepared.env["HOME"]) / ".codex" / "auth.json").read_text(encoding="utf-8")) == {
        "token": "host"
    }

def test_prepare_codex_runtime_env_applies_runtime_files(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")

    extra_dir = tmp_path / "variant-files"
    extra_dir.mkdir()
    (extra_dir / "plugin.json").write_text('{"enabled":true}', encoding="utf-8")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        copy_paths=[
            {
                "source": str(extra_dir),
                "destination": "plugins",
                "target_root": "codex_home",
            }
        ],
        materialized_files=[
            {
                "path": "settings/variant.json",
                "content": {"mode": "compare"},
                "format": "json",
                "target_root": "xdg_config_home",
            }
        ],
    )

    isolated_home = Path(env["HOME"]) / ".codex"
    assert (isolated_home / "plugins" / "plugin.json").exists()
    assert json.loads((Path(env["XDG_CONFIG_HOME"]) / "settings" / "variant.json").read_text(encoding="utf-8")) == {
        "mode": "compare"
    }

def test_prepare_codex_runtime_env_copies_directory_to_nested_home_path(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")

    skills_dir = tmp_path / "superpowers-skills"
    (skills_dir / "using-superpowers").mkdir(parents=True)
    (skills_dir / "using-superpowers" / "SKILL.md").write_text("name: using-superpowers\n", encoding="utf-8")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        copy_paths=[
            {
                "source": str(skills_dir),
                "destination": ".agents/skills/superpowers",
                "target_root": "home_dir",
            }
        ],
    )

    skill_path = Path(env["HOME"]) / ".agents" / "skills" / "superpowers" / "using-superpowers" / "SKILL.md"
    assert skill_path.exists()

def test_prepare_claude_runtime_files_applies_overrides_and_materialized_files(tmp_path) -> None:
    settings_path, mcp_config_path = prepare_claude_runtime_files(
        tmp_path,
        settings_overrides={"permissions": {"allow": ["Read"]}},
        mcp_config_overrides={"mcpServers": {"demo": {"command": "demo-mcp"}}},
        materialized_files=[
            {
                "path": "notes/setup.txt",
                "content": "variant setup",
                "format": "text",
                "target_root": "task_dir",
            }
        ],
    )

    assert json.loads(settings_path.read_text(encoding="utf-8")) == {"permissions": {"allow": ["Read"]}}
    assert json.loads(mcp_config_path.read_text(encoding="utf-8")) == {
        "mcpServers": {"demo": {"command": "demo-mcp"}}
    }
    assert (tmp_path / "notes" / "setup.txt").read_text(encoding="utf-8") == "variant setup"

def test_claude_prepare_runtime_rejects_docker_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "contextbench.agents.claude.runtime.validate_auth",
        lambda: pytest.fail("Unsupported Docker-backed Claude runtime should fail before auth validation"),
    )

    with pytest.raises(RuntimeError, match="Claude Docker runtime is not supported"):
        ClaudeAdapter().prepare_runtime(
            task_dir=tmp_path,
            setup={},
            env_overrides=None,
            runtime_backend="docker",
        )
