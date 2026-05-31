
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextbench.coding_agents.runtime import (
    prepare_claude_runtime_env,
    prepare_claude_runtime_files,
    prepare_codex_runtime_env,
)
from contextbench.coding_agents.runtime_common import expand_runtime_templates
from contextbench.agents.claude.runtime import (
    runtime_root as claude_runtime_root,
    validate_auth_in_runtime,
)
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


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


def test_prepare_codex_runtime_env_rejects_invalid_auth_json(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Codex auth file is not valid JSON"):
        prepare_codex_runtime_env(tmp_path / "task", source_codex_dir=source_codex_dir)


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

    path_parts = env["PATH"].split(":")
    assert path_parts[0] == str(Path(env["HOME"]) / ".local" / "bin")
    assert path_parts[1] == str(codex_runtime_root(tmp_path / "task") / "bin")
    assert "/usr/local/bin" in path_parts
    assert "/host-only/bin" not in path_parts
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
    assert "/host-only/bin" not in prepared.env["PATH"].split(":")
    assert str(Path(prepared.env["HOME"]) / ".local" / "bin") in prepared.env["PATH"].split(":")
    assert str(codex_runtime_root(tmp_path) / "bin") in prepared.env["PATH"].split(":")
    assert "/usr/local/bin" in prepared.env["PATH"].split(":")
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


def test_codex_prepare_runtime_expands_runtime_env_in_setup_files(tmp_path, monkeypatch) -> None:
    host_home = tmp_path / "host-home"
    source_codex_dir = host_home / ".codex"
    source_codex_dir.mkdir(parents=True)
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))

    prepared = CodexAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={
            "files_to_materialize": [
                {
                    "path": "${VARIANT_NAME}/settings.txt",
                    "content": "${RUNTIME_VALUE}",
                    "format": "text",
                    "target_root": "task_dir",
                }
            ]
        },
        env_overrides=None,
        runtime_backend="docker",
        runtime_env={"VARIANT_NAME": "variant", "RUNTIME_VALUE": "from-runtime"},
    )

    assert prepared.env is not None
    assert "RUNTIME_VALUE" not in prepared.env
    assert (tmp_path / "task" / "variant" / "settings.txt").read_text(encoding="utf-8") == "from-runtime"


def test_codex_prepare_runtime_templates_ignore_host_env_for_host_backend(tmp_path, monkeypatch) -> None:
    host_home = tmp_path / "host-home"
    source_codex_dir = host_home / ".codex"
    source_codex_dir.mkdir(parents=True)
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HOST_ONLY_TEMPLATE_SECRET", "host-secret")

    CodexAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={
            "files_to_materialize": [
                {
                    "path": "host-secret-template.txt",
                    "content": "${HOST_ONLY_TEMPLATE_SECRET}",
                    "format": "text",
                    "target_root": "task_dir",
                }
            ]
        },
        env_overrides=None,
        runtime_backend="host",
        runtime_env={},
    )

    assert (tmp_path / "task" / "host-secret-template.txt").read_text(encoding="utf-8") == "${HOST_ONLY_TEMPLATE_SECRET}"


def test_expand_runtime_templates_does_not_read_host_environment(monkeypatch) -> None:
    monkeypatch.setenv("HOST_ONLY_TEMPLATE_SECRET", "host-secret")

    expanded = expand_runtime_templates(
        {
            "explicit": "${RUNTIME_VALUE}",
            "host": "${HOST_ONLY_TEMPLATE_SECRET}",
        },
        env={"RUNTIME_VALUE": "from-runtime"},
    )

    assert expanded == {
        "explicit": "from-runtime",
        "host": "${HOST_ONLY_TEMPLATE_SECRET}",
    }


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

def test_prepare_claude_runtime_env_copies_auth_and_isolates_home(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "source-claude"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"abc"}', encoding="utf-8")
    (source_claude_dir / "settings.json").write_text('{"should":"not-copy"}', encoding="utf-8")
    monkeypatch.setenv("SECRET_SHOULD_NOT_LEAK", "1")
    monkeypatch.setenv("PATH", "/host/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    env = prepare_claude_runtime_env(
        tmp_path / "task",
        source_claude_dir=source_claude_dir,
        include_host_env=False,
    )

    isolated_home = Path(env["HOME"]) / ".claude"
    assert (isolated_home / ".credentials.json").exists()
    assert not (isolated_home / "settings.json").exists()
    assert "SECRET_SHOULD_NOT_LEAK" not in env
    assert "/host/bin" not in env["PATH"]
    assert str(Path(env["HOME"]) / ".local" / "bin") in env["PATH"].split(":")
    assert str(claude_runtime_root(tmp_path / "task") / "bin") in env["PATH"].split(":")
    assert "/usr/local/bin" in env["PATH"].split(":")
    assert env["HOME"].endswith("/home")
    assert env["OTEL_SDK_DISABLED"] == "true"


def test_prepare_claude_runtime_env_rejects_home_root_claude_json_as_portable_auth(tmp_path, monkeypatch) -> None:
    host_home = tmp_path / "host-home"
    (host_home / ".claude").mkdir(parents=True)
    (host_home / ".claude.json").write_text('{"oauth":"host-root"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Claude portable auth is unavailable"):
        prepare_claude_runtime_env(
            tmp_path / "task",
            include_host_env=False,
        )


def test_prepare_claude_runtime_env_fails_with_explicit_auth_locations(tmp_path, monkeypatch) -> None:
    host_home = tmp_path / "host-home"
    (host_home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Claude portable auth is unavailable"):
        prepare_claude_runtime_env(
            tmp_path / "task",
            include_host_env=False,
        )


def test_prepare_claude_runtime_env_prepends_isolated_tool_bins_to_path(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "source-claude"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"abc"}', encoding="utf-8")
    monkeypatch.setenv("PATH", "/host/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    env = prepare_claude_runtime_env(
        tmp_path / "task",
        source_claude_dir=source_claude_dir,
    )

    path_parts = env["PATH"].split(":")
    assert path_parts[0] == str(Path(env["HOME"]) / ".local" / "bin")
    assert path_parts[1] == str(claude_runtime_root(tmp_path / "task") / "bin")
    assert path_parts[2] == "/host/bin"


def test_prepare_claude_runtime_files_supports_home_and_xdg_roots(tmp_path) -> None:
    source_dir = tmp_path / "variant-files"
    source_dir.mkdir()
    (source_dir / "skill.md").write_text("skill", encoding="utf-8")

    prepare_claude_runtime_files(
        tmp_path / "task",
        copy_paths=[
            {
                "source": str(source_dir),
                "destination": "skills/demo",
                "target_root": "claude_home",
            }
        ],
        materialized_files=[
            {
                "path": "claude/variant.json",
                "content": {"enabled": True},
                "format": "json",
                "target_root": "xdg_config_home",
            }
        ],
    )

    root = claude_runtime_root(tmp_path / "task")
    assert (root / "home" / ".claude" / "skills" / "demo" / "skill.md").exists()
    assert json.loads((root / "xdg-config" / "claude" / "variant.json").read_text(encoding="utf-8")) == {
        "enabled": True
    }


def test_claude_prepare_runtime_uses_container_env_for_docker_backend(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "source-claude"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(source_claude_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("SECRET_SHOULD_NOT_LEAK", "1")
    monkeypatch.setenv("PATH", "/host-only/bin")

    prepared = ClaudeAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={},
        env_overrides=None,
        runtime_backend="docker",
    )

    assert prepared.env is not None
    assert "SECRET_SHOULD_NOT_LEAK" not in prepared.env
    assert "/host-only/bin" not in prepared.env["PATH"]
    assert "/usr/local/bin" in prepared.env["PATH"].split(":")
    assert "/.cache/agent-runtimes/claude/" in prepared.env["HOME"]
    assert prepared.env["HOME"].endswith("/home")
    assert json.loads((Path(prepared.env["HOME"]) / ".claude" / ".credentials.json").read_text(encoding="utf-8")) == {
        "token": "host"
    }
    assert (tmp_path / "task" / "claude.settings.json").exists()
    assert (tmp_path / "task" / "claude.mcp.json").exists()


def test_claude_prepare_runtime_accepts_configured_auth_env_without_host_auth_files(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "empty-claude"
    source_claude_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(source_claude_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    prepared = ClaudeAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={},
        env_overrides=None,
        runtime_backend="docker",
        runtime_env={"ANTHROPIC_API_KEY": "configured-key"},
    )

    assert prepared.env is not None
    assert prepared.env["ANTHROPIC_API_KEY"] == "configured-key"
    assert not (Path(prepared.env["HOME"]) / ".claude" / ".credentials.json").exists()


def test_claude_prepare_runtime_can_source_auth_from_configured_claude_config_dir(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "portable-claude-auth"
    source_claude_dir.mkdir()
    (source_claude_dir / ".credentials.json").write_text('{"token":"from-runtime-env"}', encoding="utf-8")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    prepared = ClaudeAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={},
        env_overrides=None,
        runtime_backend="docker",
        runtime_env={"CLAUDE_CONFIG_DIR": str(source_claude_dir)},
    )

    assert prepared.env is not None
    copied = Path(prepared.env["HOME"]) / ".claude" / ".credentials.json"
    assert json.loads(copied.read_text(encoding="utf-8")) == {"token": "from-runtime-env"}


def test_claude_prepare_runtime_expands_runtime_env_in_setup_files(tmp_path, monkeypatch) -> None:
    source_claude_dir = tmp_path / "empty-claude"
    source_claude_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(source_claude_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    ClaudeAdapter().prepare_runtime(
        task_dir=tmp_path / "task",
        setup={
            "files_to_materialize": [
                {
                    "path": "${VARIANT_NAME}/settings.txt",
                    "content": "${RUNTIME_VALUE}",
                    "format": "text",
                    "target_root": "task_dir",
                }
            ]
        },
        env_overrides=None,
        runtime_backend="docker",
        runtime_env={
            "ANTHROPIC_API_KEY": "configured-key",
            "VARIANT_NAME": "variant",
            "RUNTIME_VALUE": "from-runtime",
        },
    )

    assert (tmp_path / "task" / "variant" / "settings.txt").read_text(encoding="utf-8") == "from-runtime"


def test_validate_claude_auth_in_runtime_rejects_logged_out_status(tmp_path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del host_runner
            calls.append(
                {
                    "command": list(command),
                    "cwd": cwd,
                    "stdin_text": stdin_text,
                    "timeout": timeout,
                    "env": dict(env or {}),
                }
            )
            stdout_path.write_text(
                '{"loggedIn":false,"email":"person@example.com","orgId":"org-secret","orgName":"Secret Org","subscriptionType":"team"}\n',
                encoding="utf-8",
            )
            stderr_path.write_text("person@example.com orgName=SecretOrg\n", encoding="utf-8")
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    failure = validate_auth_in_runtime(
        runtime=FakeRuntime(),
        task_dir=tmp_path,
        workspace_path=workspace_path,
        timeout=300,
        env={"HOME": "/runtime-home"},
    )

    assert failure is not None
    assert failure.command_result["ok"] is False
    assert failure.command == "claude auth status --json"
    assert calls == [
        {
            "command": ["claude", "auth", "status", "--json"],
            "cwd": workspace_path,
            "stdin_text": None,
            "timeout": 30,
            "env": {"HOME": "/runtime-home"},
        }
    ]
    assert "loggedIn=false" in failure.stderr_path.read_text(encoding="utf-8")
    assert "person@example.com" not in failure.stderr_path.read_text(encoding="utf-8")
    assert "SecretOrg" not in failure.stderr_path.read_text(encoding="utf-8")
    stdout_text = failure.stdout_path.read_text(encoding="utf-8")
    assert "person@example.com" not in stdout_text
    assert "org-secret" not in stdout_text
    assert json.loads(stdout_text) == {"loggedIn": False, "status": "redacted"}


def test_validate_claude_auth_in_runtime_redacts_successful_status_artifact(tmp_path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    class FakeRuntime:
        def run_command(self, command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None, host_runner=None):
            del command, cwd, stdin_text, timeout, env, host_runner
            stdout_path.write_text(
                json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": "claude.ai",
                        "apiProvider": "firstParty",
                        "email": "person@example.com",
                        "orgId": "org-secret",
                        "orgName": "Secret Org",
                        "subscriptionType": "team",
                    }
                ),
                encoding="utf-8",
            )
            stderr_path.write_text("", encoding="utf-8")
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    failure = validate_auth_in_runtime(
        runtime=FakeRuntime(),
        task_dir=tmp_path,
        workspace_path=workspace_path,
        timeout=300,
        env={"HOME": "/runtime-home"},
    )

    assert failure is None
    stdout_text = (tmp_path / "claude-auth-status.stdout.log").read_text(encoding="utf-8")
    assert "person@example.com" not in stdout_text
    assert "org-secret" not in stdout_text
    assert "Secret Org" not in stdout_text
    assert "subscriptionType" not in stdout_text
    assert json.loads(stdout_text) == {"loggedIn": True, "status": "redacted"}
