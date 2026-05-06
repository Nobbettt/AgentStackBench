
"""Codex-specific runtime preparation and invocation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from ...artifact_sanitization import SanitizationContext, sanitize_json_value, sanitize_text
from ...coding_agents.files import ensure_dir, write_json, usage_error
from ...coding_agents.response_parsing import build_codex_raw_response
from ...coding_agents.runtime_backends import BaseTaskRuntime
from ...coding_agents.runtime_common import archive_retry_artifacts, run_command, write_prompt_file
from ...coding_agents.types import CodexRawResponse, CommandResult
from ..adapter_base import CodingAgentInvocationResult
from .parser import CodexAgentParser

_RETRYABLE_ERROR_SNIPPETS = (
    "failed to connect to websocket",
    "currently experiencing high demand",
    "missing bearer or basic authentication in header",
    "falling back from websockets to https transport",
)
_RETRY_DELAYS_SECONDS = (2, 5)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def runtime_root(task_dir: Path) -> Path:
    digest = hashlib.sha1(str(task_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    return _REPO_ROOT / ".cache" / "agent-runtimes" / "codex" / digest


def build_command(
    *,
    workspace_path: Path,
    schema_path: Path | None,
    final_output_path: Path | None,
    model: str | None,
    reasoning_effort: str | None,
    sandbox_mode: str = "workspace-write",
    writable_dirs: Sequence[Path] = (),
    extra_args: Sequence[str],
) -> tuple[list[str], str]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox_mode,
        "--json",
        "--cd",
        str(workspace_path),
    ]
    if schema_path is not None:
        command.extend(["--output-schema", str(schema_path)])
    if final_output_path is not None:
        command.extend(["--output-last-message", str(final_output_path)])
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
    seen_dirs: set[str] = set()
    for path in writable_dirs:
        resolved = str(path.resolve())
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        command.extend(["--add-dir", resolved])
    command.extend(extra_args)
    command.append("-")
    return command, "codex-events.jsonl"


def prepare_runtime_env(
    task_dir: Path,
    source_codex_dir: Path | None = None,
    *,
    include_host_env: bool = True,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> dict[str, str]:
    from ...coding_agents.runtime_common import apply_copy_paths, apply_materialized_files

    source_root = source_codex_dir or (Path.home() / ".codex")
    auth_path = source_root / "auth.json"
    if not auth_path.exists():
        raise usage_error(f"Codex auth is unavailable: expected {auth_path}")

    root = runtime_root(task_dir)
    home_dir = root / "home"
    codex_home = home_dir / ".codex"
    xdg_config_home = root / "xdg-config"
    xdg_data_home = root / "xdg-data"
    xdg_cache_home = root / "xdg-cache"

    for path in (codex_home, xdg_config_home, xdg_data_home, xdg_cache_home):
        ensure_dir(path)

    shutil.copy2(auth_path, codex_home / "auth.json")
    runtime_roots = {
        "task_dir": task_dir,
        "runtime_root": root,
        "home_dir": home_dir,
        "codex_home": codex_home,
        "xdg_config_home": xdg_config_home,
        "xdg_data_home": xdg_data_home,
        "xdg_cache_home": xdg_cache_home,
    }
    apply_copy_paths(copy_paths, roots=runtime_roots)
    apply_materialized_files(materialized_files, roots=runtime_roots)
    env = os.environ.copy() if include_host_env else {}
    env.update(
        {
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_DATA_HOME": str(xdg_data_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
            "OTEL_SDK_DISABLED": "true",
        }
    )
    return env


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    return reasoning_effort


def _raw_response_text(raw_response: CodexRawResponse) -> str:
    fragments: list[str] = []
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        message = str(event.get("message") or "").strip()
        if message:
            fragments.append(message)
        error = event.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                fragments.append(message)
    final_message = raw_response.get("final_message")
    if isinstance(final_message, str) and final_message.strip():
        fragments.append(final_message.strip())
    return "\n".join(fragments)


def _sanitize_text_artifact(path: Path, *, context: SanitizationContext) -> None:
    if not path.exists():
        return
    sanitized = sanitize_text(path.read_text(encoding="utf-8", errors="replace"), context=context)
    path.write_text(sanitized, encoding="utf-8")


def should_retry_failure(
    *,
    command_result: CommandResult,
    raw_response: CodexRawResponse,
    stderr_path: Path,
) -> bool:
    if command_result["ok"] or command_result["timeout"]:
        return False
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    haystack = "\n".join((stderr_text, _raw_response_text(raw_response))).lower()
    return any(snippet in haystack for snippet in _RETRYABLE_ERROR_SNIPPETS)


def run_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    raw_output_filename: str,
    final_output_filename: str | None,
    timeout: int,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    env: dict[str, str] | None,
    schema_path: Path | None,
    execution_backend: BaseTaskRuntime | None = None,
) -> CodingAgentInvocationResult:
    parser = CodexAgentParser()
    prompt_path = write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    raw_response_path = task_dir / raw_response_filename
    final_output_path = task_dir / final_output_filename if final_output_filename else None
    writable_dirs = [runtime_root(task_dir)]
    sandbox_mode = "workspace-write"
    if execution_backend is not None and getattr(execution_backend.config, "backend", None) == "docker":
        sandbox_mode = "danger-full-access"
    command, _ = build_command(
        workspace_path=workspace_path,
        schema_path=schema_path,
        final_output_path=final_output_path,
        model=model,
        reasoning_effort=normalize_reasoning_effort(reasoning_effort),
        sandbox_mode=sandbox_mode,
        writable_dirs=writable_dirs,
        extra_args=extra_args,
    )
    raw_output_path = task_dir / raw_output_filename
    started_at = time.time()
    raw_response: CodexRawResponse = {"agent": "codex", "response_format": "jsonl-events", "events": []}
    command_result: CommandResult = {"ok": False, "exit_code": None, "signal": None, "timeout": False}
    max_attempts = len(_RETRY_DELAYS_SECONDS) + 1
    completed_at = started_at
    for attempt_index in range(1, max_attempts + 1):
        for path in (raw_output_path, stderr_path, raw_response_path, final_output_path):
            if path is not None and path.exists():
                path.unlink()
        if execution_backend is None:
            command_result = run_command(
                command,
                cwd=workspace_path,
                stdin_text=prompt,
                stdout_path=raw_output_path,
                stderr_path=stderr_path,
                timeout=timeout,
                env=env,
            )
        else:
            command_result = execution_backend.run_command(
                command,
                cwd=workspace_path,
                stdin_text=prompt,
                stdout_path=raw_output_path,
                stderr_path=stderr_path,
                timeout=timeout,
                env=env,
                host_runner=run_command,
            )
        completed_at = time.time()
        sanitize_context = SanitizationContext(
            repo_root=Path.cwd().resolve(),
            workspace_path=workspace_path,
            task_dir=task_dir,
        )
        _sanitize_text_artifact(raw_output_path, context=sanitize_context)
        _sanitize_text_artifact(stderr_path, context=sanitize_context)
        if final_output_path is not None:
            _sanitize_text_artifact(final_output_path, context=sanitize_context)
        raw_response = build_codex_raw_response(raw_output_path, final_output_path)
        raw_response = sanitize_json_value(raw_response, context=sanitize_context)
        write_json(raw_response_path, raw_response)
        if attempt_index >= max_attempts or not should_retry_failure(
            command_result=command_result,
            raw_response=raw_response,
            stderr_path=stderr_path,
        ):
            break
        archive_retry_artifacts(
            [raw_output_path, stderr_path, raw_response_path, final_output_path],
            attempt_index=attempt_index,
        )
        time.sleep(_RETRY_DELAYS_SECONDS[attempt_index - 1])
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
