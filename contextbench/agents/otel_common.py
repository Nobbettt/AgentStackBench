# SPDX-License-Identifier: Apache-2.0

"""Shared policy helpers for OTEL-backed coding-agent adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BaseOtelAgentParser
from ..coding_agents.runtime_backends import BaseTaskRuntime
from ..coding_agents.types import CommandResult


def one_attempt_retry_metadata(*, events: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "attempts": 1,
        "max_attempts": 1,
        "retried": False,
        "suppressed": False,
        "suppression_reason": None,
        "events": list(events or []),
    }


def append_diagnostic_note(existing: str | None, note: str) -> str:
    return f"{existing} {note}" if existing else note


def force_command_failure(command_result: CommandResult) -> CommandResult:
    failed: CommandResult = dict(command_result)
    failed["ok"] = False
    if not failed.get("timeout") and failed.get("exit_code") in (None, 0):
        failed["exit_code"] = 1
    return failed


def collector_endpoint_hosts(execution_backend: BaseTaskRuntime | None) -> tuple[str, str]:
    if execution_backend is not None and getattr(execution_backend.config, "backend", None) == "docker":
        return "0.0.0.0", "host.docker.internal"
    return "127.0.0.1", "127.0.0.1"


@dataclass(frozen=True)
class OtelScoredRunValidationResult:
    command_result: CommandResult
    diagnostic_note: str | None
    failed: bool = False


@dataclass(frozen=True)
class OtelScoredRunPolicy:
    """Strict scoring policy for OTEL-backed main invocations."""

    missing_successful_tool_result_note: str
    missing_evaluable_context_note: str

    def validate(
        self,
        *,
        parser: BaseOtelAgentParser,
        raw_response: dict[str, object],
        command_result: CommandResult,
        diagnostic_note: str | None,
        workspace_path: Path,
        scored: bool,
    ) -> OtelScoredRunValidationResult:
        if not scored or not command_result.get("ok"):
            return OtelScoredRunValidationResult(command_result=command_result, diagnostic_note=diagnostic_note)

        if not parser.has_successful_tool_result_event(raw_response):
            return OtelScoredRunValidationResult(
                command_result=force_command_failure(command_result),
                diagnostic_note=append_diagnostic_note(diagnostic_note, self.missing_successful_tool_result_note),
                failed=True,
            )

        inferred_trajectory = parser.infer_trajectory_data(raw_response, record={"workspace_path": str(workspace_path)})
        if not parser.trajectory_has_evaluable_context(inferred_trajectory):
            return OtelScoredRunValidationResult(
                command_result=force_command_failure(command_result),
                diagnostic_note=append_diagnostic_note(diagnostic_note, self.missing_evaluable_context_note),
                failed=True,
            )

        return OtelScoredRunValidationResult(command_result=command_result, diagnostic_note=diagnostic_note)
