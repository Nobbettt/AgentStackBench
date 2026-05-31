
"""Config loading and effective-variant expansion for run suites."""

from __future__ import annotations

import json
from pathlib import Path

from ..coding_agents.constants import DEFAULT_AGENT_RUNTIME_IMAGES
from ..coding_agents.files import safe_path_component
from .env_files import read_env_file
from .helpers import deep_merge
from .types import (
    EffectiveVariantConfig,
    RunSuiteConfig,
    VariantConfig,
    VariantSetupConfig,
)


def merge_setup_config(base: VariantSetupConfig, override: VariantSetupConfig) -> VariantSetupConfig:
    return VariantSetupConfig(
        prompt_preamble=(
            override.prompt_preamble
            if override.prompt_preamble is not None
            else base.prompt_preamble
        ),
        setup_prompt=(
            override.setup_prompt
            if override.setup_prompt is not None
            else base.setup_prompt
        ),
        setup_prompt_timeout=(
            override.setup_prompt_timeout
            if override.setup_prompt_timeout is not None
            else base.setup_prompt_timeout
        ),
        copy_paths=[*base.copy_paths, *override.copy_paths],
        files_to_materialize=[*base.files_to_materialize, *override.files_to_materialize],
        claude_settings_overrides=deep_merge(
            base.claude_settings_overrides,
            override.claude_settings_overrides,
        ),
        claude_mcp_config=deep_merge(
            base.claude_mcp_config,
            override.claude_mcp_config,
        ),
    )


def build_run_suite_variant(
    run_suite: RunSuiteConfig,
    variant: VariantConfig,
) -> EffectiveVariantConfig:
    base = run_suite.base_run
    agent_args = list(variant.agent_args_replace) if variant.agent_args_replace is not None else [
        *base.agent_args,
        *variant.agent_args_add,
    ]
    env = dict(variant.env_replace) if variant.env_replace is not None else {**base.env, **variant.env_add}
    runtime_backend = variant.runtime_backend if variant.runtime_backend is not None else base.runtime_backend
    runtime_image = variant.runtime_image if variant.runtime_image is not None else base.runtime_image
    runtime_platform = variant.runtime_platform if variant.runtime_platform is not None else base.runtime_platform
    if runtime_backend == "docker" and runtime_image is None:
        runtime_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(run_suite.agent)
    if runtime_backend == "host" and variant.runtime_backend == "host" and variant.runtime_image is None:
        runtime_image = None
        runtime_platform = None
    base_runtime_env = {**read_env_file(base.runtime_env_file), **base.runtime_env}
    runtime_env = (
        dict(variant.runtime_env_replace)
        if variant.runtime_env_replace is not None
        else {
            **base_runtime_env,
            **read_env_file(variant.runtime_env_file),
            **variant.runtime_env_add,
        }
    )
    runtime_setup_commands = (
        list(variant.runtime_setup_commands_replace)
        if variant.runtime_setup_commands_replace is not None
        else [*base.runtime_setup_commands, *variant.runtime_setup_commands_add]
    )
    runtime_validation_commands = (
        list(variant.runtime_validation_commands_replace)
        if variant.runtime_validation_commands_replace is not None
        else [*base.runtime_validation_commands, *variant.runtime_validation_commands_add]
    )
    diff_exclude_paths = (
        list(variant.diff_exclude_paths_replace)
        if variant.diff_exclude_paths_replace is not None
        else [*base.diff_exclude_paths, *variant.diff_exclude_paths_add]
    )
    required_tool_call_patterns = (
        list(variant.required_tool_call_patterns_replace)
        if variant.required_tool_call_patterns_replace is not None
        else [*base.required_tool_call_patterns, *variant.required_tool_call_patterns_add]
    )
    required_available_tool_patterns = (
        list(variant.required_available_tool_patterns_replace)
        if variant.required_available_tool_patterns_replace is not None
        else [*base.required_available_tool_patterns, *variant.required_available_tool_patterns_add]
    )
    setup = merge_setup_config(base.setup, variant.setup)
    return EffectiveVariantConfig(
        name=variant.name,
        slug=safe_path_component(variant.name),
        description=variant.description,
        labels=list(variant.labels),
        notes=variant.notes,
        agent=run_suite.agent,
        task_data=base.task_data,
        task_csv=base.task_csv,
        subset_csv=base.subset_csv,
        bench=base.bench,
        instances=base.instances,
        limit=base.limit,
        timeout=variant.timeout if variant.timeout is not None else base.timeout,
        repo_cache=base.repo_cache,
        schema_path=base.schema_path,
        model=variant.model if variant.model is not None else base.model,
        reasoning_effort=(
            variant.reasoning_effort
            if variant.reasoning_effort is not None
            else base.reasoning_effort
        ),
        env=env,
        agent_args=agent_args,
        setup=setup,
        runtime_backend=runtime_backend,
        runtime_image=runtime_image,
        runtime_platform=runtime_platform,
        runtime_env=runtime_env,
        runtime_setup_timeout=(
            variant.runtime_setup_timeout
            if variant.runtime_setup_timeout is not None
            else base.runtime_setup_timeout
        ),
        runtime_validation_timeout=(
            variant.runtime_validation_timeout
            if variant.runtime_validation_timeout is not None
            else base.runtime_validation_timeout
        ),
        runtime_setup_cache=(
            variant.runtime_setup_cache
            if variant.runtime_setup_cache is not None
            else base.runtime_setup_cache
        ),
        runtime_setup_cache_dir=(
            variant.runtime_setup_cache_dir
            if variant.runtime_setup_cache_dir is not None
            else base.runtime_setup_cache_dir
        ),
        runtime_setup_commands=runtime_setup_commands,
        runtime_validation_commands=runtime_validation_commands,
        diff_exclude_paths=diff_exclude_paths,
        required_tool_call_patterns=required_tool_call_patterns,
        required_available_tool_patterns=required_available_tool_patterns,
        runtime_keep_failed=(
            variant.runtime_keep_failed
            if variant.runtime_keep_failed is not None
            else base.runtime_keep_failed
        ),
    )


def load_run_suite_config(path: Path) -> RunSuiteConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunSuiteConfig.model_validate(payload)
