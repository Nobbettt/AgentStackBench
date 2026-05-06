
"""Base interfaces for coding-agent adapter registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..coding_agents.types import CommandResult, StructuredOutput, TokenUsage, ToolCall

if TYPE_CHECKING:
    from .base import BaseCodingAgentParser


@dataclass(frozen=True)
class PreparedCodingAgentRuntime:
    """Adapter-specific prepared runtime state used across setup and main phases."""

    env: dict[str, str] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    execution_backend: Any | None = None


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
    ) -> PreparedCodingAgentRuntime:
        """Prepare adapter-specific runtime state for the task."""

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
    ) -> CodingAgentInvocationResult:
        """Run the scored benchmark phase in an already-prepared runtime."""
