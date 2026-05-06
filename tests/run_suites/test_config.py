
from __future__ import annotations

import csv
import json
import os
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
                    "agent_args_add": ["--plugin"],
                    "env_add": {"PLUGIN": "1"},
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
    assert effective.setup.prompt_preamble == "Enable plugin"
    assert effective.setup.setup_prompt == "Bootstrap tools first"
    assert effective.setup.setup_prompt_timeout == 90
    assert len(effective.setup.copy_paths) == 1
    assert len(effective.setup.files_to_materialize) == 1


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

    assert RunSuiteRunner(custom_config)._task_selection_kind(source_count=1136, selected_count=500) == "configured_subset"
    assert RunSuiteRunner(default_config)._task_selection_kind(source_count=1136, selected_count=500) == "representative_subset"


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


def test_checked_in_run_suite_configs_use_pinned_docker_runtimes() -> None:
    config_paths = sorted(Path("configs/run_suites").glob("*.json"))

    assert config_paths
    for config_path in config_paths:
        config = load_run_suite_config(config_path)

        assert config.base_run.runtime_backend == "docker", config_path
        if config.agent == "codex":
            assert config.base_run.runtime_image == DEFAULT_CODEX_RUNTIME_IMAGE, config_path
        if config.postprocess.convert or config.postprocess.evaluate or config.postprocess.resolve:
            assert config.postprocess.runtime_backend == "docker", config_path
            assert config.postprocess.runtime_image == DEFAULT_POSTPROCESS_RUNTIME_IMAGE, config_path


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
                                "target_root": "xdg_config_home",
                            }
                        ]
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


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
