
"""Claude-specific runtime preparation and invocation helpers."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from ...coding_agents.files import read_json, write_json, usage_error
from ...coding_agents.response_parsing import build_claude_raw_response
from ...coding_agents.runtime_backends import BaseTaskRuntime
from ...coding_agents.runtime_common import apply_copy_paths, apply_materialized_files, merge_json_objects, run_command, write_prompt_file
from ...coding_agents.types import ClaudeRawResponse
from ..adapter_base import CodingAgentInvocationResult
from .parser import ClaudeAgentParser


def build_command(
    *,
    schema_path: Path | None,
    prompt: str,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    settings_path: Path,
    mcp_config_path: Path,
) -> tuple[list[str], str]:
    command = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--verbose",
        "--no-session-persistence",
        "--permission-mode",
        "auto",
        "--setting-sources",
        "",
        "--settings",
        str(settings_path),
        "--disable-slash-commands",
        "--mcp-config",
        str(mcp_config_path),
        "--strict-mcp-config",
        "--tools",
        "default",
    ]
    if schema_path is not None:
        command.extend(["--json-schema", json.dumps(read_json(schema_path), ensure_ascii=False)])
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["--effort", reasoning_effort])
    command.extend(extra_args)
    command.append(prompt)
    return command, "claude-output.json"


def validate_auth() -> None:
    import subprocess

    result = subprocess.run(
        ["claude", "auth", "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise usage_error(f"Claude auth status failed: {result.stderr or result.stdout}".strip())
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise usage_error("Claude auth status did not return valid JSON") from exc
    if payload.get("loggedIn") is not True:
        raise usage_error("Claude Code is not logged in for non-interactive use (`claude auth status` returned loggedIn=false)")


def prepare_runtime_files(
    task_dir: Path,
    *,
    settings_overrides: dict[str, object] | None = None,
    mcp_config_overrides: dict[str, object] | None = None,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    settings_path = task_dir / "claude.settings.json"
    mcp_config_path = task_dir / "claude.mcp.json"
    write_json(settings_path, merge_json_objects({}, settings_overrides or {}))
    write_json(mcp_config_path, merge_json_objects({"mcpServers": {}}, mcp_config_overrides or {}))
    runtime_roots = {
        "task_dir": task_dir,
        "runtime_root": task_dir,
    }
    apply_copy_paths(copy_paths, roots=runtime_roots)
    apply_materialized_files(materialized_files, roots=runtime_roots)
    return settings_path, mcp_config_path


def validate_isolation(raw_response: ClaudeRawResponse) -> None:
    response = raw_response.get("response")
    if not isinstance(response, list):
        raise usage_error("Claude raw response is missing the expected verbose event array")

    init_event = None
    for item in response:
        if isinstance(item, dict) and item.get("type") == "system" and item.get("subtype") == "init":
            init_event = item
            break
    if not isinstance(init_event, dict):
        raise usage_error("Claude verbose response is missing the init event needed for isolation validation")

    if init_event.get("plugins"):
        raise usage_error("Claude isolation failed: user plugins are still loaded")
    if init_event.get("mcp_servers"):
        raise usage_error("Claude isolation failed: MCP servers are still exposed")
    slash_commands = init_event.get("slash_commands")
    if isinstance(slash_commands, list) and len(slash_commands) > 0:
        raise usage_error("Claude isolation failed: slash commands are still enabled")


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    if reasoning_effort == "xhigh":
        return "max"
    if reasoning_effort in {"none", "minimal"}:
        raise usage_error(
            "Claude does not support reasoning_effort values 'none' or 'minimal'; use low, medium, high, or xhigh"
        )
    return reasoning_effort


def run_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    timeout: int,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    env: dict[str, str] | None,
    schema_path: Path | None,
    settings_path: Path,
    mcp_config_path: Path,
    validate_runtime_isolation: bool,
    execution_backend: BaseTaskRuntime | None = None,
) -> CodingAgentInvocationResult:
    parser = ClaudeAgentParser()
    prompt_path = write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    raw_response_path = task_dir / raw_response_filename
    command, _ = build_command(
        schema_path=schema_path,
        prompt=prompt,
        model=model,
        reasoning_effort=normalize_reasoning_effort(reasoning_effort),
        extra_args=extra_args,
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )
    started_at = time.time()
    if execution_backend is None:
        command_result = run_command(
            command,
            cwd=workspace_path,
            stdin_text=None,
            stdout_path=raw_response_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=env,
        )
    else:
        command_result = execution_backend.run_command(
            command,
            cwd=workspace_path,
            stdin_text=None,
            stdout_path=raw_response_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=env,
            host_runner=run_command,
        )
    completed_at = time.time()
    raw_response: ClaudeRawResponse = build_claude_raw_response(raw_response_path)
    write_json(raw_response_path, raw_response)
    if validate_runtime_isolation:
        validate_isolation(raw_response)
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    return CodingAgentInvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        started_at=started_at,
        completed_at=completed_at,
    )
