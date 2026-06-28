# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-19 from upstream ContextBench.
# Summary of changes: capture command executions in coding-agent invocation results.

"""Base interfaces for coding-agent adapter registration."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..coding_agents.types import CommandExecution, CommandResult, StructuredOutput, TokenUsage, ToolCall

if TYPE_CHECKING:
    from .base import BaseCodingAgentParser
    from ..coding_agents.runtime_backends import RuntimeSetupResult


@dataclass(frozen=True)
class PreparedCodingAgentRuntime:
    """Adapter-specific prepared runtime state used across setup and main phases."""

    env: dict[str, str] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    execution_backend: Any | None = None


def expose_workspace_bin_on_path(env: dict[str, str], env_overrides: dict[str, str] | None) -> None:
    """Make repo-native helper entrypoints discoverable without shadowing agent tools."""

    workspace_text = str((env_overrides or {}).get("CONTEXTBENCH_WORKSPACE_PATH") or "").strip()
    if not workspace_text:
        return
    workspace_bin_path = Path(workspace_text) / "bin"
    workspace_bin = str(workspace_bin_path)
    path_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    if not path_parts:
        if not workspace_bin_path.exists():
            return
        path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if workspace_bin in path_parts:
        return
    path_parts.append(workspace_bin)
    env["PATH"] = os.pathsep.join(path_parts)


@dataclass(frozen=True)
class CodingAgentInvocationResult:
    """Normalized result of one adapter invocation phase."""

    prompt_path: Path
    stderr_path: Path
    raw_response_path: Path
    command_result: CommandResult
    structured_output: StructuredOutput | None
    token_usage: TokenUsage | None
    tool_calls: list[ToolCall]
    command_executions: list[CommandExecution]
    available_tools: list[str]
    persisted_tool_results: list[dict[str, object]]
    diagnostic_note: str | None
    retry: dict[str, object]
    started_at: float
    completed_at: float


class BaseCodingAgentAdapter(ABC):
    """Describes the agent-specific behavior for wrapper-run coding agents."""

    name: str
    aliases: tuple[str, ...] = ()
    record_suffix: str
    output_schema_path: Path
    supported_reasoning_efforts: frozenset[str] = frozenset()
    supported_runtime_target_roots: frozenset[str] = frozenset()
    supports_available_tools: bool = False

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def matches(self, candidate: str) -> bool:
        normalized = str(candidate or "").strip().lower()
        if not normalized:
            return False
        return normalized in {name.strip().lower() for name in self.all_names}

    @abstractmethod
    def build_prompt(self, task: dict[str, object]) -> str:
        """Build the benchmark prompt for this agent."""

    @abstractmethod
    def create_parser(self) -> "BaseCodingAgentParser":
        """Create a parser for this agent's raw responses and records."""

    @abstractmethod
    def prepare_runtime(
        self,
        *,
        task_dir: Path,
        setup: dict[str, object],
        env_overrides: dict[str, str] | None,
        runtime_backend: str,
        runtime_env: dict[str, str] | None = None,
    ) -> PreparedCodingAgentRuntime:
        """Prepare adapter-specific runtime state for the task."""

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
    ) -> "RuntimeSetupResult | None":
        """Validate the selected runtime before setup prompts or scored work run."""

        del task_dir, workspace_path, timeout, model, reasoning_effort, extra_args, prepared_runtime
        return None

    @abstractmethod
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
        """Run the unscored setup phase in an already-prepared runtime."""

    @abstractmethod
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
        """Run the scored benchmark phase in an already-prepared runtime."""
