
"""Claude coding-agent adapter registration."""

from __future__ import annotations

import os
from pathlib import Path

from ..adapter_base import BaseCodingAgentAdapter, CodingAgentInvocationResult, PreparedCodingAgentRuntime
from ...coding_agents.constants import CLAUDE_OUTPUT_SCHEMA_PATH
from .parser import ClaudeAgentParser
from .prompting import build_prompt


class ClaudeAdapter(BaseCodingAgentAdapter):
    name = "claude"
    aliases = ("claude-code",)
    record_suffix = "claude"
    output_schema_path = CLAUDE_OUTPUT_SCHEMA_PATH
    supported_reasoning_efforts = frozenset({"low", "medium", "high", "xhigh"})
    supported_runtime_target_roots = frozenset({"task_dir", "runtime_root"})

    def build_prompt(self, task: dict[str, object]) -> str:
        return build_prompt(task)

    def create_parser(self) -> ClaudeAgentParser:
        return ClaudeAgentParser()

    def prepare_runtime(
        self,
        *,
        task_dir: Path,
        setup: dict[str, object],
        env_overrides: dict[str, str] | None,
        runtime_backend: str,
    ) -> PreparedCodingAgentRuntime:
        from .runtime import prepare_runtime_files, validate_auth

        if runtime_backend != "host":
            raise RuntimeError("Claude Docker runtime is not supported; use runtime_backend='host'.")
        validate_auth()
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
        command_env = os.environ.copy()
        if env_overrides:
            command_env.update(env_overrides)
        return PreparedCodingAgentRuntime(
            env=command_env,
            state={
                "settings_path": settings_path,
                "mcp_config_path": mcp_config_path,
            },
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
    ) -> CodingAgentInvocationResult:
        from .runtime import run_invocation

        return run_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="setup-prompt.txt",
            stderr_filename="setup-stderr.log",
            raw_response_filename="setup-raw-response.json",
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
    ) -> CodingAgentInvocationResult:
        from .runtime import run_invocation

        return run_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="prompt.txt",
            stderr_filename="stderr.log",
            raw_response_filename="raw-response.json",
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
        )


CODING_AGENT_ADAPTER = ClaudeAdapter()
