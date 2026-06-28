
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
from contextbench.agents.adapter_base import RuntimePreflightContext
from contextbench.agents.claude.runtime import (
    runtime_root as claude_runtime_root,
    validate_auth_in_runtime,
)
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.claude_otel.adapter import ClaudeOtelAdapter
from contextbench.agents.codex.adapter import CodexAdapter
from contextbench.agents.codex_otel_v2.adapter import CodexOtelV2Adapter
from contextbench.agents.codex_otel_v2.runtime import (
    prepare_runtime_env as prepare_codex_otel_v2_runtime_env,
    runtime_root as codex_otel_v2_runtime_root,
)


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
    assert path_parts[2] == str(codex_runtime_root(tmp_path / "task") / "npm-global" / "bin")
    assert "/usr/local/bin" in path_parts
    assert "/host-only/bin" not in path_parts
    assert "SECRET_SHOULD_NOT_LEAK" not in env
    assert "/.cache/agent-runtimes/codex/" in env["HOME"]
    assert env["HOME"].endswith("/home")
    assert env["CONTEXTBENCH_RUNTIME_ROOT"] == str(codex_runtime_root(tmp_path / "task"))
    assert env["CONTEXTBENCH_RUNTIME_BIN"] == str(codex_runtime_root(tmp_path / "task") / "bin")
    assert env["NPM_CONFIG_PREFIX"] == str(codex_runtime_root(tmp_path / "task") / "npm-global")
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert json.loads((Path(env["HOME"]) / ".codex" / "auth.json").read_text(encoding="utf-8")) == {"token": "host"}


def test_prepare_codex_runtime_env_exposes_explicit_tool_bundle_for_docker(tmp_path, monkeypatch) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    bundle_root = tmp_path / "bundle"
    (bundle_root / "usr-local" / "bin").mkdir(parents=True)
    (bundle_root / "usr-local" / "lib" / "node_modules").mkdir(parents=True)
    monkeypatch.setenv("PATH", "/host-only/bin")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        include_host_env=False,
        runtime_env={"CONTEXTBENCH_CODEX_TOOL_BUNDLE": str(bundle_root)},
    )

    path_parts = env["PATH"].split(":")
    assert str(bundle_root / "usr-local" / "bin") in path_parts
    assert path_parts.index(str(codex_runtime_root(tmp_path / "task") / "npm-global" / "bin")) < path_parts.index(
        str(bundle_root / "usr-local" / "bin")
    )
    assert "/host-only/bin" not in path_parts


def test_prepare_codex_otel_v2_runtime_env_matches_codex_docker_tool_paths_without_disabling_otel(
    tmp_path,
    monkeypatch,
) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    bundle_root = tmp_path / "bundle"
    (bundle_root / "usr-local" / "bin").mkdir(parents=True)
    (bundle_root / "usr-local" / "lib" / "node_modules").mkdir(parents=True)
    monkeypatch.setenv("PATH", "/host-only/bin")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    env = prepare_codex_otel_v2_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        include_host_env=False,
        runtime_env={"CONTEXTBENCH_CODEX_TOOL_BUNDLE": str(bundle_root)},
    )

    path_parts = env["PATH"].split(":")
    assert path_parts[0] == str(Path(env["HOME"]) / ".local" / "bin")
    assert path_parts[1] == str(codex_otel_v2_runtime_root(tmp_path / "task") / "bin")
    assert path_parts[2] == str(codex_otel_v2_runtime_root(tmp_path / "task") / "npm-global" / "bin")
    assert str(bundle_root / "usr-local" / "bin") in path_parts
    assert "/host-only/bin" not in path_parts
    assert env["CONTEXTBENCH_RUNTIME_ROOT"] == str(codex_otel_v2_runtime_root(tmp_path / "task"))
    assert env["CONTEXTBENCH_RUNTIME_BIN"] == str(codex_otel_v2_runtime_root(tmp_path / "task") / "bin")
    assert env["NPM_CONFIG_PREFIX"] == str(codex_otel_v2_runtime_root(tmp_path / "task") / "npm-global")
    assert "OTEL_SDK_DISABLED" not in env


def test_codex_otel_v2_adapter_uses_codex_tool_bundle_mount_and_preflight(tmp_path, monkeypatch) -> None:
    bundle_root = tmp_path / "bundle"
    (bundle_root / "usr-local" / "bin").mkdir(parents=True)
    (bundle_root / "usr-local" / "lib" / "node_modules").mkdir(parents=True)
    adapter = CodexOtelV2Adapter()

    assert adapter.extra_runtime_readonly_mounts(
        runtime_backend="docker",
        runtime_env={"CONTEXTBENCH_CODEX_TOOL_BUNDLE": str(bundle_root)},
    ) == (bundle_root,)

    monkeypatch.setattr("contextbench.agents.codex.tool_bundle.codex_tool_bundle_root", lambda runtime_env=None: None)
    failures = adapter.runtime_preflight_failures(
        context=RuntimePreflightContext(
            variant_name="treatment",
            runtime_backend="docker",
            runtime_image_source="resolution",
            runtime_env={},
        )
    )

    assert [failure.to_json() for failure in failures] == [
        {
            "variant": "treatment",
            "agent": "codex-otel-v2",
            "error": (
                "Codex resolution-image runtimes require a repo-local Codex tool bundle. "
                "Run 'python3 -m contextbench.run_suites_setup codex-tool-bundle'."
            ),
        }
    ]


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
    assert str(codex_runtime_root(tmp_path) / "npm-global" / "bin") in prepared.env["PATH"].split(":")
    assert "/usr/local/bin" in prepared.env["PATH"].split(":")
    assert "/.cache/agent-runtimes/codex/" in prepared.env["HOME"]
    assert prepared.env["HOME"].endswith("/home")
    assert prepared.env["CONTEXTBENCH_RUNTIME_ROOT"] == str(codex_runtime_root(tmp_path))
    assert prepared.env["CONTEXTBENCH_RUNTIME_BIN"] == str(codex_runtime_root(tmp_path) / "bin")
    assert json.loads((Path(prepared.env["HOME"]) / ".codex" / "auth.json").read_text(encoding="utf-8")) == {
        "token": "host"
    }


def test_agent_docker_runtime_path_exposes_workspace_bin_without_shadowing_tools(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    (workspace_path / "bin").mkdir(parents=True)
    host_home = tmp_path / "host-home"
    source_codex_dir = host_home / ".codex"
    source_codex_dir.mkdir(parents=True)
    (source_codex_dir / "auth.json").write_text('{"token":"host"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    env_overrides = {"CONTEXTBENCH_WORKSPACE_PATH": str(workspace_path)}

    codex = CodexAdapter().prepare_runtime(
        task_dir=tmp_path / "codex-task",
        setup={},
        env_overrides=env_overrides,
        runtime_backend="docker",
    )
    codex_otel = CodexOtelV2Adapter().prepare_runtime(
        task_dir=tmp_path / "codex-otel-task",
        setup={},
        env_overrides=env_overrides,
        runtime_backend="docker",
    )
    claude = ClaudeAdapter().prepare_runtime(
        task_dir=tmp_path / "claude-task",
        setup={},
        env_overrides=env_overrides,
        runtime_backend="docker",
        runtime_env={"ANTHROPIC_API_KEY": "test-key"},
    )
    claude_otel = ClaudeOtelAdapter().prepare_runtime(
        task_dir=tmp_path / "claude-otel-task",
        setup={},
        env_overrides=env_overrides,
        runtime_backend="docker",
        runtime_env={"ANTHROPIC_API_KEY": "test-key"},
    )

    for prepared in (codex, codex_otel, claude, claude_otel):
        assert prepared.env is not None
        path_parts = prepared.env["PATH"].split(":")
        assert path_parts[-1] == str(workspace_path / "bin")
        assert path_parts.index("/usr/local/bin") < path_parts.index(str(workspace_path / "bin"))


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
