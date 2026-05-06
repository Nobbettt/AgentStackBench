
"""Codex coding-agent adapter registration."""

from __future__ import annotations

from pathlib import Path

from ..adapter_base import BaseCodingAgentAdapter, CodingAgentInvocationResult, PreparedCodingAgentRuntime
from ...coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH
from .parser import CodexAgentParser
from .prompting import build_prompt

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


class CodexAdapter(BaseCodingAgentAdapter):
    name = "codex"
    aliases = ()
    record_suffix = "codex"
    output_schema_path = CODEX_OUTPUT_SCHEMA_PATH
    supported_reasoning_efforts = _SUPPORTED_REASONING_EFFORTS
    supported_runtime_target_roots = _SUPPORTED_RUNTIME_TARGET_ROOTS

    def build_prompt(self, task: dict[str, object]) -> str:
        return build_prompt(task)

    def create_parser(self) -> CodexAgentParser:
        return CodexAgentParser()

    def prepare_runtime(
        self,
        *,
        task_dir: Path,
        setup: dict[str, object],
        env_overrides: dict[str, str] | None,
        runtime_backend: str,
    ) -> PreparedCodingAgentRuntime:
        from .runtime import prepare_runtime_env

        copy_paths = setup.get("copy_paths")
        materialized_files = setup.get("files_to_materialize")
        env = prepare_runtime_env(
            task_dir,
            include_host_env=runtime_backend != "docker",
            materialized_files=materialized_files if isinstance(materialized_files, (list, tuple)) else None,
            copy_paths=copy_paths if isinstance(copy_paths, (list, tuple)) else None,
        )
        if env_overrides:
            env.update(env_overrides)
        return PreparedCodingAgentRuntime(env=env)

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
            raw_output_filename="setup-codex-events.jsonl",
            final_output_filename="setup-last-message.txt",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=None,
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
            raw_output_filename="codex-events.jsonl",
            final_output_filename="final-output.json",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            env=prepared_runtime.env,
            schema_path=schema_path,
            execution_backend=prepared_runtime.execution_backend,
        )


CODING_AGENT_ADAPTER = CodexAdapter()
