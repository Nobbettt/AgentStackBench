# SPDX-License-Identifier: Apache-2.0

"""Codex OTEL v2 coding-agent adapter registration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..adapter_base import (
    OTEL_TOOL_RESULTS_CONTEXT_SOURCE,
    BaseCodingAgentAdapter,
    CodingAgentInvocationResult,
    PreparedCodingAgentRuntime,
    expose_workspace_bin_on_path,
)
from ..codex.tool_bundle import CodexToolBundleSupportMixin
from ..runtime_lifecycle import RuntimeRootLifecycleMixin
from .parser import CodexOtelV2AgentParser
from .prompting import build_prompt

CODEX_OTEL_V2_OUTPUT_SCHEMA_PATH = Path(__file__).resolve().with_name("output.schema.json")

_SUPPORTED_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_SUPPORTED_RUNTIME_TARGET_ROOTS = frozenset(
    {
        "task_dir",
        "runtime_root",
        "home_dir",
        "codex_home",
        "xdg_config_home",
        "xdg_data_home",
        "xdg_cache_home",
    }
)


class CodexOtelV2Adapter(CodexToolBundleSupportMixin, RuntimeRootLifecycleMixin, BaseCodingAgentAdapter):
    name = "codex-otel-v2"
    aliases = ("codex_v2", "codex-otel")
    record_suffix = "codex-otel-v2"
    output_schema_path = CODEX_OTEL_V2_OUTPUT_SCHEMA_PATH
    supported_reasoning_efforts = _SUPPORTED_REASONING_EFFORTS
    supported_runtime_target_roots = _SUPPORTED_RUNTIME_TARGET_ROOTS
    scored_context_source = OTEL_TOOL_RESULTS_CONTEXT_SOURCE
    score_inferred_context = True

    def build_prompt(self, task: dict[str, object]) -> str:
        return build_prompt(task)

    def create_parser(self) -> CodexOtelV2AgentParser:
        return CodexOtelV2AgentParser()

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
        from .runtime import prepare_runtime_env
        from ...coding_agents.runtime_common import expand_runtime_templates

        env = prepare_runtime_env(
            task_dir,
            include_host_env=runtime_backend != "docker",
            runtime_env=runtime_env,
        )
        if env_overrides:
            env.update(env_overrides)
        expose_workspace_bin_on_path(env, env_overrides)
        env.pop("OTEL_SDK_DISABLED", None)
        template_env = {
            **dict(runtime_env or {}),
            **dict(env_overrides or {}),
            "CONTEXTBENCH_TASK_DIR": str(task_dir),
        }
        setup = expand_runtime_templates(setup, env=template_env)
        copy_paths = setup.get("copy_paths")
        materialized_files = setup.get("files_to_materialize")
        from .runtime import apply_runtime_setup_files

        apply_runtime_setup_files(
            task_dir,
            materialized_files=materialized_files if isinstance(materialized_files, (list, tuple)) else None,
            copy_paths=copy_paths if isinstance(copy_paths, (list, tuple)) else None,
        )
        return PreparedCodingAgentRuntime(env=env)

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
        from .runtime import validate_cli_in_runtime

        del model, reasoning_effort, extra_args
        execution_backend = prepared_runtime.execution_backend
        if execution_backend is None:
            return None
        if getattr(execution_backend.config, "backend", None) != "docker":
            return None
        return validate_cli_in_runtime(
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
            raw_output_filename="setup-codex-stdout.txt",
            final_output_filename="setup-last-message.txt",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=None,
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
            raw_output_filename="codex-stdout.txt",
            final_output_filename="final-output.json",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=schema_path,
            execution_backend=prepared_runtime.execution_backend,
            retry_dirty_check=retry_dirty_check,
        )


CODING_AGENT_ADAPTER = CodexOtelV2Adapter()
