
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
import contextbench.run_suites_setup as run_suites_setup
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant, load_run_suite_config
from contextbench.run_suites import parse_args as parse_run_suite_args
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE,
    DEFAULT_CLAUDE_RUNTIME_IMAGE,
    DEFAULT_CODEX_RUNTIME_IMAGE,
    DEFAULT_POSTPROCESS_RUNTIME_IMAGE,
    DEFAULT_SUBSET_CSV,
)
from contextbench.run_suites_core.postprocess import (
    ResolutionCommandError,
    describe_resolution_backend_support,
    evaluate_resolution_for_suite,
    export_resolution_predictions,
    run_resolution_evaluation,
)
from contextbench.coding_agents.task_data import load_tasks


from .helpers import _fake_run_coding_agent_task, _make_fake_agent_record, _write_task_inputs


_MEMTRACE_NATIVE_TOOL_PATTERN = r"^mcp__memtrace__"
_MEMTRACE_RUNTIME_SETUP_COMMAND = (
    "npm install --global memtrace@0.6.46 --include=optional --no-fund --no-update-notifier --loglevel=warn "
    "&& memtrace --version"
)
_PYTHON_SOURCE_BOOTSTRAP_DIFF_EXCLUDES = [
    ".venv/**",
    ".codex-venv/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    "**/__pycache__/**",
    "build/**",
    "dist/**",
    "*.egg-info/**",
    ".eggs/**",
    "**/*.so",
    "**/*.pyd",
    "**/*.dylib",
    "**/*.dll",
    "**/*.o",
    "**/*.a",
]


def _assert_memtrace_uses_native_codex_mcp(treatment) -> None:
    prompt = treatment.setup.prompt_preamble or ""

    assert "Memtrace MCP is connected" in prompt
    assert "native Memtrace MCP tools" in prompt
    assert "mcp-tool" not in prompt
    assert "$CONTEXTBENCH_RUNTIME_ROOT/bin/mcp-tool" not in prompt
    assert "benchmark marks the run invalid" not in prompt
    assert treatment.required_tool_call_patterns_add == [_MEMTRACE_NATIVE_TOOL_PATTERN]
    assert treatment.required_command_patterns_add == []


def _assert_no_base_python_source_bootstrap(config) -> None:
    assert config.base_run.diff_exclude_paths == _PYTHON_SOURCE_BOOTSTRAP_DIFF_EXCLUDES
    assert config.base_run.runtime_setup_timeout is None
    assert config.base_run.runtime_setup_commands == []
    assert config.base_run.setup.copy_paths == []


def _assert_memtrace_resolution_runtime_config(config) -> None:
    assert config.base_run.runtime_backend == "docker"
    assert config.base_run.runtime_image_source == "resolution"
    assert config.base_run.runtime_image is None
    assert config.base_run.runtime_images_by_bench == {}
    assert config.base_run.runtime_images_by_python == {}
    assert config.base_run.runtime_platform == "linux/amd64"
    assert config.base_run.runtime_env["CONTEXTBENCH_CODEX_TOOL_BUNDLE"] == ".cache/agent-tool-bundles/codex/current"
    _assert_no_base_python_source_bootstrap(config)


def _assert_memtrace_runtime_state_is_task_local(treatment) -> None:
    setup_commands = "\n".join(treatment.runtime_setup_commands_add)

    assert treatment.runtime_env_add["MEMTRACE_DATA_DIR"] == "${CONTEXTBENCH_TASK_DIR}/memtrace-state"
    assert treatment.runtime_env_add["MEMTRACE_MEMDB_DATA_DIR"] == "${CONTEXTBENCH_TASK_DIR}/memtrace-memdb"
    assert "MEMTRACE_DATA_DIR" in setup_commands
    assert "MEMTRACE_MEMDB_DATA_DIR" in setup_commands
    assert "env_vars = " in setup_commands


def test_run_suite_cli_rejects_partial_postprocess_escape_hatch(tmp_path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text("experiment_name: suite\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        parse_run_suite_args([str(config_path), "--allow-partial-postprocess"])

    assert exc.value.code == 2


def test_superpowers_all_benches_smoke_config_selects_one_task_per_bench() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-superpowers-bootstrap-5-all-benches-smoke.json")

    tasks = load_tasks(config.base_run.task_data, subset_csv=config.base_run.task_csv, limit=config.base_run.limit)

    assert config.postprocess.resolve is True
    assert len(tasks) == 4
    assert {str(task["bench"]) for task in tasks} == {"Verified", "Pro", "Poly", "Multi"}


def test_claude_cortex_all_benches_smoke_config_selects_one_task_per_bench() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/claude-cortex-bootstrap-5-all-benches-smoke.json")

    tasks = load_tasks(config.base_run.task_data, subset_csv=config.base_run.task_csv, limit=config.base_run.limit)

    assert config.agent == "claude"
    assert config.base_run.runtime_backend == "docker"
    assert config.base_run.runtime_image == DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE
    assert config.base_run.runtime_platform == "linux/amd64"
    assert config.base_run.model == "claude-opus-4-7"
    assert config.postprocess.resolve is True
    assert config.postprocess.env_file is None
    assert config.parallelism.max_workers == 2
    treatment = next(variant for variant in config.variants if variant.name == "with-cortex-mcp")
    setup_commands = "\n".join(treatment.runtime_setup_commands_add)
    validation_commands = "\n".join(treatment.runtime_validation_commands_add)

    assert treatment.runtime_setup_timeout == 10800
    assert treatment.runtime_validation_timeout == 2400
    assert treatment.runtime_setup_cache is True
    assert "./scripts/context.sh bootstrap" in setup_commands
    assert "ryugraph/ryu_native.js" in setup_commands
    assert "@danielblomma/cortex-mcp@2.0.13" in setup_commands
    assert "@danielblomma/cortex-mcp@latest" not in setup_commands
    assert treatment.runtime_validation_commands_add
    assert "cortex-version.txt" in validation_commands
    assert "pkg.version !== '2.0.13'" in validation_commands
    assert ".context/cache/manifest.json" in validation_commands
    assert ".context/cache/graph-manifest.json" in validation_commands
    assert ".context/embeddings/manifest.json" in validation_commands
    assert "git clean -fdx --" in validation_commands
    assert treatment.agent_args_add == []
    assert treatment.diff_exclude_paths_add == [".context/**", "AGENTS.md", "CLAUDE.md"]
    assert "Before using built-in" not in (treatment.setup.prompt_preamble or "")
    assert "Use Cortex as the primary repository-navigation workflow" in (treatment.setup.prompt_preamble or "")
    assert treatment.required_available_tool_patterns_add == [
        "^mcp__cortex__context_search$",
        "^mcp__cortex__context_get_related$",
        "^mcp__cortex__context_impact$",
        "^mcp__cortex__context_get_rules$",
        "^mcp__cortex__context_reload$",
    ]
    assert len(tasks) == 4
    assert {str(task["bench"]) for task in tasks} == {"Verified", "Pro", "Poly", "Multi"}


def test_claude_cortex_main_config_bootstraps_and_validates_cortex() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/claude-cortex-bootstrap.json")
    assert config.base_run.runtime_backend == "docker"
    assert config.base_run.runtime_image == DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE
    assert config.base_run.runtime_platform == "linux/amd64"
    assert config.parallelism.max_workers == 2
    treatment = next(variant for variant in config.variants if variant.name == "with-cortex-mcp")
    setup_commands = "\n".join(treatment.runtime_setup_commands_add)
    validation_commands = "\n".join(treatment.runtime_validation_commands_add)

    assert treatment.runtime_setup_timeout == 10800
    assert treatment.runtime_validation_timeout == 2700
    assert treatment.runtime_setup_cache is True
    assert "./scripts/context.sh bootstrap" in setup_commands
    assert "ryugraph/ryu_native.js" in setup_commands
    assert "@danielblomma/cortex-mcp@2.0.13" in setup_commands
    assert "@danielblomma/cortex-mcp@latest" not in setup_commands
    assert "cortex-version.txt" in validation_commands
    assert "pkg.version !== '2.0.13'" in validation_commands
    assert ".context/cache/manifest.json" in validation_commands
    assert ".context/cache/graph-manifest.json" in validation_commands
    assert ".context/embeddings/manifest.json" in validation_commands
    assert "git clean -fdx --" in validation_commands
    assert treatment.agent_args_add == []
    assert treatment.diff_exclude_paths_add == [".context/**", "AGENTS.md", "CLAUDE.md"]
    assert "Before using built-in" not in (treatment.setup.prompt_preamble or "")
    assert "Use Cortex as the primary repository-navigation workflow" in (treatment.setup.prompt_preamble or "")
    assert treatment.required_available_tool_patterns_add == [
        "^mcp__cortex__context_search$",
        "^mcp__cortex__context_get_related$",
        "^mcp__cortex__context_impact$",
        "^mcp__cortex__context_get_rules$",
        "^mcp__cortex__context_reload$",
    ]


def test_codex_memtrace_main_config_bootstraps_and_validates_memtrace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-memtrace.json")

    assert config.agent == "codex"
    _assert_memtrace_resolution_runtime_config(config)
    assert config.base_run.runtime_env_file == Path(".env")
    assert config.base_run.model == "gpt-5.5"
    assert config.base_run.reasoning_effort == "high"
    assert config.base_run.runtime_env["MEMTRACE_TELEMETRY"] == "off"
    assert config.base_run.runtime_env["MEMTRACE_RAIL_SHADOW"] == "off"
    assert config.base_run.runtime_env["MEMTRACE_NO_REMOTE_RECEIPT"] == "1"
    assert config.base_run.runtime_env["OTEL_SDK_DISABLED"] == "true"
    assert config.parallelism.max_workers == 1
    assert config.postprocess.env_file == Path(".env")
    baseline = next(variant for variant in config.variants if variant.name == "baseline")
    treatment = next(variant for variant in config.variants if variant.name == "with-memtrace-mcp")
    effective_baseline = build_run_suite_variant(config, baseline)
    effective_treatment = build_run_suite_variant(config, treatment)
    setup_commands = "\n".join(treatment.runtime_setup_commands_add)
    validation_commands = "\n".join(treatment.runtime_validation_commands_add)

    assert baseline.model == "gpt-5.5"
    assert baseline.reasoning_effort == "high"
    assert baseline.runtime_prebuild is None
    assert treatment.model == "gpt-5.5"
    assert treatment.reasoning_effort == "high"
    assert treatment.runtime_setup_timeout == 3600
    assert treatment.runtime_validation_timeout == 600
    assert treatment.runtime_prebuild is None
    assert effective_baseline.runtime_setup_commands == []
    assert effective_treatment.runtime_setup_commands == treatment.runtime_setup_commands_add
    assert effective_treatment.setup.copy_paths == treatment.setup.copy_paths
    assert treatment.runtime_env_add["MEMTRACE_AUTH_URL"] == "http://127.0.0.1:45682"
    assert treatment.runtime_env_add["CONTEXTBENCH_PROXY_PORT"] == "45682"
    assert treatment.runtime_env_add["CONTEXTBENCH_PROXY_UPSTREAM_ORIGIN"] == "https://www.memtrace.io"
    assert "MEMTRACE_AUTH_DEVICE_ID" in treatment.runtime_env_add["CONTEXTBENCH_PROXY_REWRITE_RULES_JSON"]
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_COMMAND"] == "memtrace"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_ARGS_JSON"] == '["mcp"]'
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_CLIENT_NAME"] == "agentstackbench-memtrace"
    assert treatment.runtime_env_add["MEMTRACE_TELEMETRY"] == "off"
    assert treatment.runtime_env_add["MEMTRACE_RAIL_SHADOW"] == "off"
    assert treatment.runtime_env_add["MEMTRACE_NO_REMOTE_RECEIPT"] == "1"
    assert treatment.runtime_env_add["OTEL_SDK_DISABLED"] == "true"
    _assert_memtrace_runtime_state_is_task_local(treatment)
    assert treatment.setup.copy_paths[0].source == Path("agent-resources/http-json-rewrite-proxy.mjs")
    assert treatment.setup.copy_paths[0].destination == "http-json-rewrite-proxy.mjs"
    assert treatment.setup.copy_paths[0].target_root == "task_dir"
    assert treatment.setup.copy_paths[1].source == Path("agent-resources/prepare-local-git-worktree.sh")
    assert treatment.setup.copy_paths[1].destination == "prepare-local-git-worktree.sh"
    assert treatment.setup.copy_paths[1].target_root == "task_dir"
    assert treatment.setup.copy_paths[2].source == Path("agent-resources/mcp-tool.mjs")
    assert treatment.setup.copy_paths[2].destination == "bin/mcp-tool"
    assert treatment.setup.copy_paths[2].target_root == "runtime_root"
    assert treatment.runtime_setup_commands_add[0] == _MEMTRACE_RUNTIME_SETUP_COMMAND
    assert "memtrace@0.6.46" in setup_commands
    assert "apt-get install" not in setup_commands
    assert "MEMTRACE_LICENSE_KEY is required" in setup_commands
    assert "MEMTRACE_AUTH_DEVICE_ID is required" in setup_commands
    assert "Memtrace privacy opt-out env is not configured" in setup_commands
    assert "http-json-rewrite-proxy.mjs" in setup_commands
    assert "http-json-rewrite-proxy.log" in setup_commands
    assert "prepare-local-git-worktree.sh" in setup_commands
    assert "memtrace/installer/dist/index.js\" install --only codex --global --yes" in setup_commands
    assert "required = true" in setup_commands
    assert "startup_timeout_sec = 60" in setup_commands
    assert "tool_timeout_sec = 120" in setup_commands
    assert 'default_tools_approval_mode = "auto"' in setup_commands
    assert "env_vars = " in setup_commands
    assert "MEMTRACE_AUTH_URL" in setup_commands
    assert "MEMTRACE_NO_REMOTE_RECEIPT" in setup_commands
    assert "rm -f \"$HOME/.codex/hooks.json\"" in setup_commands
    assert setup_commands.index("prepare-local-git-worktree.sh") < setup_commands.index("memtrace index .")
    assert setup_commands.index("memtrace index .") < setup_commands.index("memtrace start --headless --force")
    assert "memtrace start --headless --force" in setup_commands
    assert "memtrace index ." in setup_commands
    assert "no git repositories found" in setup_commands
    assert "memtrace-version.txt" in validation_commands
    assert "grep -q '0.6.46'" in validation_commands
    assert "memtrace-status-final.log" in validation_commands
    assert "No data indexed yet" in validation_commands
    assert "0 nodes - 0 edges" in validation_commands
    assert "grep -q 'required = true'" in validation_commands
    assert "grep -q 'env_vars = .*MEMTRACE_AUTH_URL'" in validation_commands
    assert "codex mcp get memtrace" in validation_commands
    assert "codex-memtrace-mcp.txt" in validation_commands
    assert "enabled: true" in validation_commands
    assert "command:.*memtrace|args:.*mcp" in validation_commands
    assert "CONTEXTBENCH_RUNTIME_ROOT/mcp-tools.json" in validation_commands
    assert "CONTEXTBENCH_RUNTIME_ROOT/mcp-repos.json" in validation_commands
    assert "CONTEXTBENCH_TASK_DIR/memtrace-tools.json" not in validation_commands
    assert "CONTEXTBENCH_TASK_DIR/memtrace-repos.json" not in validation_commands
    assert "$CONTEXTBENCH_RUNTIME_ROOT/bin/mcp-tool\" --list" in validation_commands
    assert "$CONTEXTBENCH_RUNTIME_ROOT/bin/mcp-tool\" list_indexed_repositories" in validation_commands
    assert treatment.diff_exclude_paths_add == [".memdb/**", ".memtrace/**"]
    _assert_memtrace_uses_native_codex_mcp(treatment)


def test_codex_memtrace_smoke_config_selects_one_task_per_bench() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-memtrace-smoke.json")

    tasks = load_tasks(config.base_run.task_data, subset_csv=config.base_run.task_csv, limit=config.base_run.limit)
    treatment = next(variant for variant in config.variants if variant.name == "with-memtrace-mcp")

    assert config.agent == "codex"
    _assert_memtrace_resolution_runtime_config(config)
    assert config.base_run.model == "gpt-5.5"
    assert config.base_run.reasoning_effort == "high"
    assert config.base_run.runtime_env["MEMTRACE_TELEMETRY"] == "off"
    assert config.base_run.runtime_env["MEMTRACE_RAIL_SHADOW"] == "off"
    assert config.base_run.runtime_env["MEMTRACE_NO_REMOTE_RECEIPT"] == "1"
    assert config.base_run.runtime_env["OTEL_SDK_DISABLED"] == "true"
    assert config.postprocess.resolve is True
    assert config.parallelism.max_workers == 1
    assert treatment.runtime_prebuild is None
    assert treatment.runtime_env_add["MEMTRACE_AUTH_URL"] == "http://127.0.0.1:45682"
    assert treatment.runtime_env_add["CONTEXTBENCH_PROXY_PORT"] == "45682"
    assert treatment.runtime_env_add["CONTEXTBENCH_PROXY_UPSTREAM_ORIGIN"] == "https://www.memtrace.io"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_COMMAND"] == "memtrace"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_ARGS_JSON"] == '["mcp"]'
    assert treatment.runtime_env_add["MEMTRACE_TELEMETRY"] == "off"
    assert treatment.runtime_env_add["MEMTRACE_RAIL_SHADOW"] == "off"
    assert treatment.runtime_env_add["MEMTRACE_NO_REMOTE_RECEIPT"] == "1"
    assert treatment.runtime_env_add["MEMTRACE_SKIP_EMBED"] == "1"
    assert treatment.runtime_env_add["MEMTRACE_NO_REPLAY"] == "1"
    assert treatment.runtime_env_add["OTEL_SDK_DISABLED"] == "true"
    _assert_memtrace_runtime_state_is_task_local(treatment)
    _assert_memtrace_uses_native_codex_mcp(treatment)
    assert len(tasks) == 4
    assert {str(task["bench"]) for task in tasks} == {"Verified", "Pro", "Poly", "Multi"}


def test_codex_memtrace_configs_do_not_use_variant_runtime_prebuilds() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for path in sorted((repo_root / "configs/run_suites").glob("codex-memtrace*.json")):
        config = load_run_suite_config(path)
        _assert_memtrace_resolution_runtime_config(config)
        assert all(variant.runtime_prebuild is None or not variant.runtime_prebuild.enabled for variant in config.variants), path


def test_codex_memtrace_limit_15_config_uses_main_treatment_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-memtrace-limit-15.json")

    assert config.experiment_name == "codex-memtrace-limit-15"
    assert config.agent == "codex"
    assert config.base_run.limit == 15
    assert config.base_run.model == "gpt-5.5"
    assert config.base_run.reasoning_effort == "high"
    assert config.parallelism.max_workers == 1
    _assert_memtrace_resolution_runtime_config(config)
    assert [variant.name for variant in config.variants if variant.enabled] == ["baseline", "with-memtrace-mcp"]
    treatment = next(variant for variant in config.variants if variant.name == "with-memtrace-mcp")
    assert treatment.runtime_prebuild is None
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_COMMAND"] == "memtrace"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_ARGS_JSON"] == '["mcp"]'
    _assert_memtrace_runtime_state_is_task_local(treatment)
    _assert_memtrace_uses_native_codex_mcp(treatment)


def test_codex_memtrace_limit_15_noembed_config_disables_expensive_memtrace_embedding() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-memtrace-limit-15-noembed.json")

    assert config.experiment_name == "codex-memtrace-limit-15-noembed"
    assert config.base_run.limit == 15
    assert config.base_run.model == "gpt-5.5"
    assert config.base_run.reasoning_effort == "high"
    assert config.parallelism.max_workers == 1
    _assert_memtrace_resolution_runtime_config(config)
    treatment = next(variant for variant in config.variants if variant.name == "with-memtrace-mcp")
    assert treatment.runtime_env_add["MEMTRACE_SKIP_EMBED"] == "1"
    assert treatment.runtime_env_add["MEMTRACE_NO_REPLAY"] == "1"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_COMMAND"] == "memtrace"
    _assert_memtrace_runtime_state_is_task_local(treatment)
    _assert_memtrace_uses_native_codex_mcp(treatment)


def test_codex_memtrace_limit_30_noembed_config_disables_expensive_memtrace_embedding() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_run_suite_config(repo_root / "configs/run_suites/codex-memtrace-limit-30-noembed.json")

    assert config.experiment_name == "codex-memtrace-limit-30-noembed"
    assert config.base_run.limit == 30
    assert config.base_run.model == "gpt-5.5"
    assert config.base_run.reasoning_effort == "high"
    assert config.parallelism.max_workers == 1
    _assert_memtrace_resolution_runtime_config(config)
    assert [variant.name for variant in config.variants if variant.enabled] == ["baseline", "with-memtrace-mcp"]
    treatment = next(variant for variant in config.variants if variant.name == "with-memtrace-mcp")
    assert treatment.runtime_env_add["MEMTRACE_SKIP_EMBED"] == "1"
    assert treatment.runtime_env_add["MEMTRACE_NO_REPLAY"] == "1"
    assert treatment.runtime_env_add["CONTEXTBENCH_MCP_COMMAND"] == "memtrace"
    _assert_memtrace_runtime_state_is_task_local(treatment)
    _assert_memtrace_uses_native_codex_mcp(treatment)


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_agents_claude_md_exclusions_are_not_agent_defaults(tmp_path, agent) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": f"{agent}-default-diff-policy",
            "agent": agent,
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False, "runtime_backend": "host"},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.diff_exclude_paths == []


def test_codex_run_suite_rejects_claude_specific_setup_fields(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="does not support Claude-specific setup fields"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "codex-with-claude-mcp-config",
                "agent": "codex",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "setup": {
                        "claude_mcp_config": {
                            "mcpServers": {
                                "cortex": {"command": "cortex", "args": ["mcp"]},
                            },
                        },
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_build_run_suite_variant_merges_base_and_variant_overrides(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "suite-codex",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "agent_args": ["--base"],
                "env": {"BASE": "1"},
                "reasoning_effort": "medium",
                "runtime_platform": "linux/arm64",
                "runtime_setup_timeout": 120,
                "runtime_validation_timeout": 30,
                "runtime_setup_cache": True,
                "runtime_setup_cache_dir": str(tmp_path / "setup-cache"),
                "runtime_setup_commands": ["base-setup"],
                "runtime_validation_commands": ["base-validation"],
                "diff_exclude_paths": [".base-cache/**"],
                "required_tool_call_patterns": ["^base_tool$"],
                "required_command_patterns": ["^base_command$"],
                "required_available_tool_patterns": ["^base_available_tool$"],
                "setup": {
                    "copy_paths": [
                        {
                            "source": str(tmp_path),
                            "destination": "base",
                            "target_root": "task_dir",
                        }
                    ]
                },
            },
            "variants": [
                {
                    "name": "with-plugin",
                    "reasoning_effort": "high",
                    "runtime_platform": "linux/amd64",
                    "runtime_setup_timeout": 240,
                    "runtime_validation_timeout": 60,
                    "runtime_setup_cache": False,
                    "agent_args_add": ["--plugin"],
                    "env_add": {"PLUGIN": "1"},
                    "runtime_setup_commands_add": ["plugin-setup"],
                    "runtime_validation_commands_add": ["plugin-validation"],
                    "diff_exclude_paths_add": [".plugin-cache/**"],
                    "required_tool_call_patterns_add": ["^plugin_tool$"],
                    "required_command_patterns_add": ["^plugin_command$"],
                    "required_available_tool_patterns_add": ["^plugin_available_tool$"],
                    "setup": {
                        "prompt_preamble": "Enable plugin",
                        "setup_prompt": "Bootstrap tools first",
                        "setup_prompt_timeout": 90,
                        "files_to_materialize": [
                            {
                                "path": "plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "task_dir",
                            }
                        ],
                    },
                }
            ],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.agent_args == ["--base", "--plugin"]
    assert effective.env == {"BASE": "1", "PLUGIN": "1"}
    assert effective.reasoning_effort == "high"
    assert effective.runtime_platform == "linux/amd64"
    assert effective.runtime_setup_timeout == 240
    assert effective.runtime_validation_timeout == 60
    assert effective.runtime_setup_cache is False
    assert effective.runtime_setup_cache_dir == tmp_path / "setup-cache"
    assert effective.setup.prompt_preamble == "Enable plugin"
    assert effective.setup.setup_prompt == "Bootstrap tools first"
    assert effective.setup.setup_prompt_timeout == 90
    assert len(effective.setup.copy_paths) == 1
    assert len(effective.setup.files_to_materialize) == 1
    assert effective.runtime_setup_commands == ["base-setup", "plugin-setup"]
    assert effective.runtime_validation_commands == ["base-validation", "plugin-validation"]
    assert effective.diff_exclude_paths == [".base-cache/**", ".plugin-cache/**"]
    assert effective.required_tool_call_patterns == ["^base_tool$", "^plugin_tool$"]
    assert effective.required_command_patterns == ["^base_command$", "^plugin_command$"]
    assert effective.required_available_tool_patterns == ["^base_available_tool$", "^plugin_available_tool$"]


def test_run_suite_selection_kind_only_marks_default_csv_as_representative(tmp_path) -> None:
    task_data, custom_task_csv = _write_task_inputs(tmp_path, count=1)
    custom_config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "custom-subset",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(custom_task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )
    default_config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "default-subset",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(DEFAULT_SUBSET_CSV),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )
    filtered_config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "filtered-subset",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(DEFAULT_SUBSET_CSV),
                "instances": ["psf__requests-1000"],
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert RunSuiteRunner(custom_config)._task_selection_kind(source_count=1136, selected_count=500) == "configured_subset"
    assert RunSuiteRunner(default_config)._task_selection_kind(source_count=1136, selected_count=500) == "representative_subset"
    assert RunSuiteRunner(filtered_config)._task_selection_kind(source_count=1136, selected_count=1) == "filtered_selection"


def test_build_run_suite_variant_uses_pinned_runtime_and_merges_runtime_env(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=secret-token\nBASE=from-file\n", encoding="utf-8")
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "runtime-variant",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_env_file": str(env_file),
                "runtime_env": {"BASE": "1"},
            },
            "variants": [
                {
                    "name": "runtime-plugin",
                    "runtime_env_add": {"PLUGIN": "1"},
                }
            ],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.runtime_backend == "docker"
    assert effective.runtime_image == DEFAULT_CODEX_RUNTIME_IMAGE
    assert effective.runtime_env == {"HF_TOKEN": "secret-token", "BASE": "1", "PLUGIN": "1"}


def test_build_run_suite_variant_replaces_runtime_prebuild_config(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "runtime-prebuild-config",
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
                    "commands": ["base setup"],
                    "env": {"BASE_TOOL": "1"},
                    "image_tag": "base-prebuilt:latest",
                },
            },
            "variants": [
                {
                    "name": "variant-prebuild",
                    "runtime_prebuild": {
                        "enabled": True,
                        "commands": ["variant setup"],
                        "env": {"VARIANT_TOOL": "1"},
                        "pull_base_image": True,
                    },
                }
            ],
            "parallelism": {"max_workers": 4, "agent_workers": 3, "scheduler": "global"},
            "postprocess": {
                "convert": False,
                "evaluate": False,
                "prebuild_resolution_images": True,
                "prebuild_resolution_workers": 2,
            },
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert config.parallelism.agent_workers == 3
    assert config.parallelism.scheduler == "global"
    assert config.postprocess.prebuild_resolution_images is True
    assert config.postprocess.prebuild_resolution_workers == 2
    assert effective.runtime_prebuild.enabled is True
    assert effective.runtime_prebuild.commands == ["variant setup"]
    assert effective.runtime_prebuild.env == {"VARIANT_TOOL": "1"}
    assert effective.runtime_prebuild.image_tag is None
    assert effective.runtime_prebuild.pull_base_image is True


def test_build_run_suite_variant_accepts_resolution_runtime_image_source(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resolution-runtime-source",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
                "runtime_image_source": "resolution",
                "runtime_platform": "linux/amd64",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.runtime_image_source == "resolution"
    assert effective.runtime_image is None
    assert effective.runtime_platform == "linux/amd64"


def test_run_suite_config_rejects_runtime_prebuild_with_resolution_runtime_source(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime_prebuild requires runtime_image_source='configured'"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "resolution-runtime-prebuild",
                "agent": "codex",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "runtime_backend": "docker",
                    "runtime_image_source": "resolution",
                    "runtime_prebuild": {
                        "enabled": True,
                        "commands": ["echo setup"],
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_rejects_runtime_prebuild_for_host_runtime(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime_prebuild requires runtime_backend='docker'"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "host-runtime-prebuild",
                "agent": "codex",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "runtime_backend": "host",
                    "runtime_prebuild": {
                        "enabled": True,
                        "commands": ["echo shared setup"],
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_rejects_host_runtime_with_image(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime_image can only be set"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "host-with-image",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "runtime_backend": "host",
                    "runtime_image": "unsupported-host-image",
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_rejects_host_runtime_with_platform(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="runtime_platform can only be set"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "host-with-platform",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "runtime_backend": "host",
                    "runtime_platform": "linux/amd64",
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_checked_in_run_suite_configs_use_expected_base_runtimes() -> None:
    config_paths = sorted(Path("configs/run_suites").glob("*.json"))

    assert config_paths
    for config_path in config_paths:
        config = load_run_suite_config(config_path)

        if config.agent == "codex":
            assert config.base_run.runtime_backend == "docker", config_path
            if config_path.name.startswith("codex-memtrace"):
                _assert_memtrace_resolution_runtime_config(config)
                treatment = next((variant for variant in config.variants if variant.name == "with-memtrace-mcp"), None)
                if treatment is not None:
                    _assert_memtrace_runtime_state_is_task_local(treatment)
            else:
                assert config.base_run.runtime_image == DEFAULT_CODEX_RUNTIME_IMAGE, config_path
        if config.agent == "claude":
            assert config.base_run.runtime_backend == "docker", config_path
            if config.base_run.runtime_platform == "linux/amd64":
                assert config.base_run.runtime_image == DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE, config_path
            else:
                assert config.base_run.runtime_image == DEFAULT_CLAUDE_RUNTIME_IMAGE, config_path
        if config.postprocess.convert or config.postprocess.evaluate or config.postprocess.resolve:
            assert config.postprocess.runtime_backend == "docker", config_path
            assert config.postprocess.runtime_image == DEFAULT_POSTPROCESS_RUNTIME_IMAGE, config_path


def test_setup_claude_runtime_image_uses_pinned_claude_code_version(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_suites_setup, "_docker_build", lambda **kwargs: calls.append(kwargs))

    assert run_suites_setup.setup_claude_runtime_image() == 0

    assert calls == [
        {
            "image": DEFAULT_CLAUDE_RUNTIME_IMAGE,
            "dockerfile": run_suites_setup.CLAUDE_RUNTIME_DOCKERFILE,
            "force": False,
            "build_args": {"CLAUDE_CODE_VERSION": run_suites_setup.CLAUDE_CODE_VERSION},
            "platform": None,
        }
    ]


def test_setup_codex_tool_bundle_extracts_from_pinned_runtime_image(tmp_path, monkeypatch) -> None:
    bundle_root = tmp_path / "agent-tool-bundles" / "codex"
    bundle_dir = bundle_root / f"codex-cli-{run_suites_setup.CODEX_CLI_VERSION}"
    current = bundle_root / "current"
    run_calls: list[list[str]] = []
    subprocess_calls: list[list[str]] = []

    def fake_run_command(command: list[str]) -> None:
        run_calls.append(list(command))
        target = Path(command[-1])
        target.mkdir(parents=True, exist_ok=True)

    def fake_subprocess_run(command, capture_output, text, check):
        del capture_output, text, check
        subprocess_calls.append(list(command))
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, stdout="container-1\n", stderr="")
        if command[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(run_suites_setup, "CODEX_TOOL_BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(run_suites_setup, "CODEX_TOOL_BUNDLE_DIR", bundle_dir)
    monkeypatch.setattr(run_suites_setup, "CODEX_TOOL_BUNDLE_CURRENT", current)
    monkeypatch.setattr(run_suites_setup, "_docker_image_exists", lambda image: True)
    monkeypatch.setattr(run_suites_setup, "_run", fake_run_command)
    monkeypatch.setattr(run_suites_setup.subprocess, "run", fake_subprocess_run)

    assert run_suites_setup.setup_codex_tool_bundle(image="contextbench-codex-runtime:test") == 0

    tmp_bundle_dir = bundle_dir.with_name(f".{bundle_dir.name}.tmp")
    assert subprocess_calls[0] == ["docker", "create", "contextbench-codex-runtime:test"]
    assert run_calls == [
        ["docker", "cp", "container-1:/usr/local/bin", str(tmp_bundle_dir / "usr-local" / "bin")],
        [
            "docker",
            "cp",
            "container-1:/usr/local/lib/node_modules",
            str(tmp_bundle_dir / "usr-local" / "lib" / "node_modules"),
        ],
    ]
    assert current.is_symlink()
    assert current.readlink() == Path(bundle_dir.name)
    assert json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8")) == {
        "codex_cli_version": run_suites_setup.CODEX_CLI_VERSION,
        "source_image": "contextbench-codex-runtime:test",
    }


def test_setup_claude_runtime_image_accepts_explicit_platform_and_image(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_suites_setup, "_docker_build", lambda **kwargs: calls.append(kwargs))

    assert (
        run_suites_setup.setup_claude_runtime_image(
            image="contextbench-claude-runtime:test-amd64",
            platform="linux/amd64",
        )
        == 0
    )

    assert calls == [
        {
            "image": "contextbench-claude-runtime:test-amd64",
            "dockerfile": run_suites_setup.CLAUDE_RUNTIME_DOCKERFILE,
            "force": False,
            "build_args": {"CLAUDE_CODE_VERSION": run_suites_setup.CLAUDE_CODE_VERSION},
            "platform": "linux/amd64",
        }
    ]


def test_setup_claude_runtime_image_uses_native_deps_flavor_for_pinned_amd64_image(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_suites_setup, "_docker_build", lambda **kwargs: calls.append(kwargs))

    assert (
        run_suites_setup.setup_claude_runtime_image(
            image=DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE,
            platform="linux/amd64",
        )
        == 0
    )

    assert calls == [
        {
            "image": DEFAULT_CLAUDE_RUNTIME_AMD64_IMAGE,
            "dockerfile": run_suites_setup.CLAUDE_RUNTIME_NATIVE_DEPS_DOCKERFILE,
            "force": False,
            "build_args": {"CLAUDE_CODE_VERSION": run_suites_setup.CLAUDE_CODE_VERSION},
            "platform": "linux/amd64",
        }
    ]


def test_setup_claude_runtime_image_accepts_explicit_native_deps_flavor(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_suites_setup, "_docker_build", lambda **kwargs: calls.append(kwargs))

    assert (
        run_suites_setup.setup_claude_runtime_image(
            image="contextbench-claude-runtime:custom-native-deps",
            platform="linux/amd64",
            flavor="native-deps",
        )
        == 0
    )

    assert calls == [
        {
            "image": "contextbench-claude-runtime:custom-native-deps",
            "dockerfile": run_suites_setup.CLAUDE_RUNTIME_NATIVE_DEPS_DOCKERFILE,
            "force": False,
            "build_args": {"CLAUDE_CODE_VERSION": run_suites_setup.CLAUDE_CODE_VERSION},
            "platform": "linux/amd64",
        }
    ]


def test_run_suite_config_rejects_claude_only_invalid_target_roots(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="Agent 'claude' only supports setup target_root values"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "claude-invalid-root",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "setup": {
                        "files_to_materialize": [
                            {
                                "path": "settings/plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "codex_home",
                            }
                        ]
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_defaults_claude_docker_runtime_image(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-docker-default",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.base_run.runtime_backend == "docker"
    assert config.base_run.runtime_image == DEFAULT_CLAUDE_RUNTIME_IMAGE


def test_run_suite_config_defaults_codex_docker_runtime_image(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-docker-default",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "docker",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.base_run.runtime_backend == "docker"
    assert config.base_run.runtime_image == DEFAULT_CODEX_RUNTIME_IMAGE


def test_run_suite_variant_switching_to_docker_uses_agent_default_image(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-host-base-docker-variant",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "docker-baseline", "runtime_backend": "docker"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert config.base_run.runtime_image is None
    assert effective.runtime_backend == "docker"
    assert effective.runtime_image == DEFAULT_CLAUDE_RUNTIME_IMAGE


def test_run_suite_config_rejects_unsupported_claude_reasoning_effort(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="Agent 'claude' only supports reasoning_effort values"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "claude-invalid-reasoning",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "reasoning_effort": "minimal",
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_allows_codex_runtime_target_roots(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-valid-root",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [
                {
                    "name": "plugin",
                    "setup": {
                        "files_to_materialize": [
                            {
                                "path": "settings/plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "xdg_config_home",
                            }
                        ]
                    },
                }
            ],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.variants[0].setup.files_to_materialize[0].target_root == "xdg_config_home"


def test_run_suite_config_defaults_schema_path_per_agent(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    codex = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-default-schema",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-codex"),
                "repo_cache": str(tmp_path / "cache-codex"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )
    claude = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-default-schema",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-claude"),
                "repo_cache": str(tmp_path / "cache-claude"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert codex.base_run.schema_path == CODEX_OUTPUT_SCHEMA_PATH
    assert claude.base_run.schema_path == CLAUDE_OUTPUT_SCHEMA_PATH


def test_run_suite_config_normalizes_agent_aliases(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-alias-schema",
            "agent": "claude-code",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-claude"),
                "repo_cache": str(tmp_path / "cache-claude"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.agent == "claude"
    assert config.base_run.schema_path == CLAUDE_OUTPUT_SCHEMA_PATH
