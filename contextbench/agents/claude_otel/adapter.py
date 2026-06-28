# SPDX-License-Identifier: Apache-2.0

"""Claude v2 coding-agent adapter using OpenTelemetry capture."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..adapter_base import (
    OTEL_TOOL_RESULTS_CONTEXT_SOURCE,
    BaseCodingAgentAdapter,
    CodingAgentInvocationResult,
    PreparedCodingAgentRuntime,
    expose_workspace_bin_on_path,
)
from ...coding_agents.constants import CLAUDE_OTEL_OUTPUT_SCHEMA_PATH
from ..runtime_lifecycle import RuntimeRootLifecycleMixin
from .parser import ClaudeOtelAgentParser
from .prompting import build_prompt


class ClaudeOtelAdapter(RuntimeRootLifecycleMixin, BaseCodingAgentAdapter):
    name = "claude-otel"
    aliases = ("claude-v2", "claude-otel-v2", "claude-code-otel")
    record_suffix = "claude-otel"
    output_schema_path = CLAUDE_OTEL_OUTPUT_SCHEMA_PATH
    supported_reasoning_efforts = frozenset({"low", "medium", "high", "xhigh"})
    supports_available_tools = True
    scored_context_source = OTEL_TOOL_RESULTS_CONTEXT_SOURCE
    score_inferred_context = True
    supported_runtime_target_roots = frozenset(
        {
            "task_dir",
            "runtime_root",
            "home_dir",
            "claude_home",
            "xdg_config_home",
            "xdg_data_home",
            "xdg_cache_home",
        }
    )

    def create_parser(self) -> ClaudeOtelAgentParser:
        return ClaudeOtelAgentParser()

    def build_prompt(self, task: dict[str, object]) -> str:
        return build_prompt(task)

    def runtime_root(self, task_dir: Path) -> Path:
        from .runtime import runtime_root

        return runtime_root(task_dir)

    def prepare_runtime(
        self,
        *,
        task_dir: Path,
        setup: dict[str, object],
        env_overrides: dict[str, str] | None,
        runtime_backend: str,
        runtime_env: dict[str, str] | None = None,
    ) -> PreparedCodingAgentRuntime:
        from .runtime import prepare_runtime_env, prepare_runtime_files, validate_auth
        from ...coding_agents.runtime_common import expand_runtime_templates

        auth_seed_env = {**dict(runtime_env or {}), **dict(env_overrides or {})}
        command_env = prepare_runtime_env(
            task_dir,
            include_host_env=runtime_backend != "docker",
            env_seed=auth_seed_env,
        )
        if env_overrides:
            command_env.update(env_overrides)
        expose_workspace_bin_on_path(command_env, env_overrides)
        template_env = {
            **dict(runtime_env or {}),
            **dict(env_overrides or {}),
            "CONTEXTBENCH_TASK_DIR": str(task_dir),
        }
        setup = expand_runtime_templates(setup, env=template_env)
        settings_overrides = setup.get("claude_settings_overrides")
        mcp_config_overrides = setup.get("claude_mcp_config")
        copy_paths = setup.get("copy_paths")
        materialized_files = setup.get("files_to_materialize")
        settings_path, mcp_config_path = prepare_runtime_files(
            task_dir,
            settings_overrides=settings_overrides if isinstance(settings_overrides, dict) else None,
            mcp_config_overrides=mcp_config_overrides if isinstance(mcp_config_overrides, dict) else None,
            materialized_files=materialized_files if isinstance(materialized_files, (list, tuple)) else None,
            copy_paths=copy_paths if isinstance(copy_paths, (list, tuple)) else None,
        )
        if runtime_backend != "docker":
            validation_env = {**dict(runtime_env or {}), **dict(command_env)}
            if "PATH" not in validation_env and "PATH" in os.environ:
                validation_env["PATH"] = os.environ["PATH"]
            validate_auth(env=validation_env)
        return PreparedCodingAgentRuntime(
            env=command_env,
            state={
                "settings_path": settings_path,
                "mcp_config_path": mcp_config_path,
            },
        )

    def validate_pre_invocation_runtime(
        self,
        *,
        task_dir: Path,
        workspace_path: Path,
        timeout: int,
        model: str | None = None,
        reasoning_effort: str | None = None,
        extra_args: tuple[str, ...] = (),
        prepared_runtime: PreparedCodingAgentRuntime,
    ):
        from .runtime import validate_auth_in_runtime

        del model, reasoning_effort, extra_args
        execution_backend = prepared_runtime.execution_backend
        if execution_backend is None:
            return None
        if getattr(execution_backend.config, "backend", None) != "docker":
            return None
        return validate_auth_in_runtime(
            runtime=execution_backend,
            task_dir=task_dir,
            workspace_path=workspace_path,
            timeout=timeout,
            env=prepared_runtime.env,
        )

    def run_setup_invocation(
        self,
        *,
        task_dir: Path,
        workspace_path: Path,
        prompt: str,
        timeout: int,
        model: str | None,
        reasoning_effort: str | None,
        extra_args: tuple[str, ...],
        prepared_runtime: PreparedCodingAgentRuntime,
        retry_dirty_check: Callable[[], bool] | None = None,
    ) -> CodingAgentInvocationResult:
        from .runtime import run_invocation

        return run_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="setup-prompt.txt",
            stderr_filename="setup-stderr.log",
            raw_response_filename="setup-raw-response.json",
            raw_output_filename="setup-claude-output.jsonl",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=None,
            settings_path=prepared_runtime.state["settings_path"],
            mcp_config_path=prepared_runtime.state["mcp_config_path"],
            validate_runtime_isolation=True,
            execution_backend=prepared_runtime.execution_backend,
            retry_dirty_check=retry_dirty_check,
        )

    def run_main_invocation(
        self,
        *,
        task_dir: Path,
        workspace_path: Path,
        prompt: str,
        timeout: int,
        model: str | None,
        reasoning_effort: str | None,
        extra_args: tuple[str, ...],
        schema_path: Path,
        prepared_runtime: PreparedCodingAgentRuntime,
        retry_dirty_check: Callable[[], bool] | None = None,
    ) -> CodingAgentInvocationResult:
        from .runtime import run_invocation

        return run_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="prompt.txt",
            stderr_filename="stderr.log",
            raw_response_filename="raw-response.json",
            raw_output_filename="claude-output.jsonl",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=schema_path,
            settings_path=prepared_runtime.state["settings_path"],
            mcp_config_path=prepared_runtime.state["mcp_config_path"],
            validate_runtime_isolation=True,
            execution_backend=prepared_runtime.execution_backend,
            retry_dirty_check=retry_dirty_check,
        )


CODING_AGENT_ADAPTER = ClaudeOtelAdapter()
