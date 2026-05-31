
"""Claude-specific runtime preparation and invocation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Callable

from ...coding_agents.files import read_json, safe_path_component, write_json, usage_error
from ...coding_agents.response_parsing import build_claude_raw_response, structured_output_schema_error
from ...coding_agents.runtime_backends import BaseTaskRuntime, RuntimeSetupResult
from ...coding_agents.runtime_common import (
    apply_copy_paths,
    apply_materialized_files,
    archive_retry_artifacts,
    merge_json_objects,
    run_command,
    SHARED_RETRYABLE_ERROR_SNIPPETS,
    write_prompt_file,
)
from ...coding_agents.types import ClaudeRawResponse, CommandResult
from ..adapter_base import CodingAgentInvocationResult
from .parser import ClaudeAgentParser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RETRYABLE_ERROR_SNIPPETS = (
    *SHARED_RETRYABLE_ERROR_SNIPPETS,
    "status code: 529",
)
_RETRY_DELAYS_SECONDS = (2, 5)
_CLAUDE_CONFIG_AUTH_FILES = (
    ".credentials.json",
    "credentials.json",
    "auth.json",
)
_CLAUDE_AUTH_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
_HOST_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)
_CONTAINER_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_PERSISTED_OUTPUT_RE = re.compile(
    r"<persisted-output>\s*Output too large \((?P<label>[^)]+)\)\. "
    r"Full output saved to:\s*(?P<path>[^\r\n]+)",
    re.MULTILINE,
)


def runtime_root(task_dir: Path) -> Path:
    digest = hashlib.sha1(str(task_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    return _REPO_ROOT / ".cache" / "agent-runtimes" / "claude" / digest


def runtime_roots(task_dir: Path) -> dict[str, Path]:
    root = runtime_root(task_dir)
    home_dir = root / "home"
    return {
        "task_dir": task_dir,
        "runtime_root": root,
        "home_dir": home_dir,
        "claude_home": home_dir / ".claude",
        "xdg_config_home": root / "xdg-config",
        "xdg_data_home": root / "xdg-data",
        "xdg_cache_home": root / "xdg-cache",
    }


def claude_portable_auth_sources(
    *,
    env: Mapping[str, str] | None = None,
    source_claude_dir: Path | None = None,
) -> dict[str, object]:
    """Return non-secret evidence of Claude auth usable outside the host keychain."""

    effective_env = {str(key): str(value) for key, value in dict(os.environ if env is None else env).items()}
    config_dir_text = effective_env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
    source_config_dir = source_claude_dir or Path(config_dir_text or Path.home() / ".claude")
    auth_files = [source_config_dir / filename for filename in _CLAUDE_CONFIG_AUTH_FILES]
    return {
        "env_vars": [key for key in _CLAUDE_AUTH_ENV_KEYS if effective_env.get(key)],
        "files": [str(path) for path in auth_files if path.is_file()],
        "source_config_dir": str(source_config_dir),
        "checked_env_vars": list(_CLAUDE_AUTH_ENV_KEYS),
        "checked_files": [str(path) for path in auth_files],
    }


def claude_has_portable_auth(
    *,
    env: Mapping[str, str] | None = None,
    source_claude_dir: Path | None = None,
) -> bool:
    sources = claude_portable_auth_sources(env=env, source_claude_dir=source_claude_dir)
    return bool(sources["env_vars"] or sources["files"])


def _copy_claude_auth(
    *,
    source_config_dir: Path,
    target_claude_dir: Path,
    env: dict[str, str],
) -> None:
    candidates: list[tuple[Path, Path]] = [
        (source_config_dir / filename, target_claude_dir / filename)
        for filename in _CLAUDE_CONFIG_AUTH_FILES
    ]

    copied = False
    for source_path, target_path in candidates:
        if not source_path.is_file():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied = True
    if copied or any(env.get(key) for key in _CLAUDE_AUTH_ENV_KEYS):
        return
    expected = ", ".join(str(source_path) for source_path, _ in candidates)
    env_names = ", ".join(_CLAUDE_AUTH_ENV_KEYS)
    raise usage_error(
        "Claude portable auth is unavailable for isolated runtime execution: "
        f"expected one of [{expected}] or one of [{env_names}] in the environment. "
        "Host `claude auth status` can use local keychain state that is not available inside Docker. "
        "Run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN, or provide it through a local runtime_env_file."
    )


def prepare_runtime_env(
    task_dir: Path,
    source_claude_dir: Path | None = None,
    *,
    include_host_env: bool = True,
    env_seed: dict[str, str] | None = None,
) -> dict[str, str]:
    seed_env = {str(key): str(value) for key, value in dict(env_seed or {}).items()}
    source_config_dir = source_claude_dir or Path(
        seed_env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    )
    roots = runtime_roots(task_dir)
    local_bin = roots["home_dir"] / ".local" / "bin"
    runtime_bin = roots["runtime_root"] / "bin"
    for path in (
        roots["claude_home"],
        roots["xdg_config_home"],
        roots["xdg_data_home"],
        roots["xdg_cache_home"],
        local_bin,
        runtime_bin,
    ):
        path.mkdir(parents=True, exist_ok=True)

    env: dict[str, str] = {}
    if include_host_env:
        env.update({key: value for key, value in os.environ.items() if key in _HOST_ENV_ALLOWLIST})
        env["PATH"] = os.pathsep.join(
            [
                str(local_bin),
                str(runtime_bin),
                env.get("PATH") or os.defpath,
            ]
        )
    else:
        env["PATH"] = ":".join(
            [
                str(local_bin),
                str(runtime_bin),
                _CONTAINER_DEFAULT_PATH,
            ]
        )
    env.update({key: value for key, value in os.environ.items() if key in _CLAUDE_AUTH_ENV_KEYS})
    env.update({key: value for key, value in seed_env.items() if key in _CLAUDE_AUTH_ENV_KEYS})
    env.update(
        {
            "HOME": str(roots["home_dir"]),
            "XDG_CONFIG_HOME": str(roots["xdg_config_home"]),
            "XDG_DATA_HOME": str(roots["xdg_data_home"]),
            "XDG_CACHE_HOME": str(roots["xdg_cache_home"]),
            "OTEL_SDK_DISABLED": "true",
        }
    )
    _copy_claude_auth(
        source_config_dir=source_config_dir,
        target_claude_dir=roots["claude_home"],
        env=env,
    )
    return env


def build_command(
    *,
    schema_path: Path | None,
    prompt: str,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    settings_path: Path,
    mcp_config_path: Path,
    permission_mode: str = "acceptEdits",
) -> tuple[list[str], str]:
    del prompt
    command = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--permission-mode",
        permission_mode,
        "--setting-sources",
        "",
        "--settings",
        str(settings_path),
        "--disable-slash-commands",
        "--mcp-config",
        str(mcp_config_path),
        "--strict-mcp-config",
    ]
    if schema_path is not None:
        command.extend(["--json-schema", json.dumps(read_json(schema_path), ensure_ascii=False)])
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["--effort", reasoning_effort])
    command.extend(extra_args)
    return command, "claude-output.jsonl"


def validate_auth(*, env: dict[str, str] | None = None) -> None:
    import subprocess

    result = subprocess.run(
        ["claude", "auth", "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    if result.returncode != 0:
        raise usage_error(f"Claude auth status failed: {result.stderr or result.stdout}".strip())
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise usage_error("Claude auth status did not return valid JSON") from exc
    if payload.get("loggedIn") is not True:
        raise usage_error("Claude Code is not logged in for non-interactive use (`claude auth status` returned loggedIn=false)")


def _failed_auth_result() -> CommandResult:
    return {
        "ok": False,
        "exit_code": 1,
        "signal": None,
        "timeout": False,
    }


def _redact_auth_validation_text(text: str) -> str:
    redacted = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    for key in (
        "email",
        "orgId",
        "orgName",
        "subscriptionType",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "token",
        "authToken",
        "accessToken",
        "refreshToken",
    ):
        redacted = re.sub(
            rf'("{re.escape(key)}"\s*:\s*)"[^"]*"',
            rf'\1"[redacted]"',
            redacted,
            flags=re.IGNORECASE,
        )
        redacted = re.sub(
            rf"({re.escape(key)}\s*[=:]\s*)\S+",
            rf"\1[redacted]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def _redact_auth_validation_file(path: Path) -> None:
    if not path.exists():
        return
    redacted = _redact_auth_validation_text(path.read_text(encoding="utf-8", errors="replace"))
    path.write_text(redacted, encoding="utf-8")


def _write_auth_status_summary(stdout_path: Path, *, logged_in: bool | None) -> None:
    summary: dict[str, object] = {"status": "redacted"}
    if logged_in is not None:
        summary["loggedIn"] = logged_in
    write_json(stdout_path, summary)


def _append_auth_validation_error(stderr_path: Path, message: str) -> None:
    existing = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    existing = _redact_auth_validation_text(existing)
    separator = "\n" if existing and not existing.endswith("\n") else ""
    stderr_path.write_text(f"{existing}{separator}{message}\n", encoding="utf-8")


def validate_auth_in_runtime(
    *,
    runtime: BaseTaskRuntime,
    task_dir: Path,
    workspace_path: Path,
    timeout: int,
    env: dict[str, str] | None,
) -> RuntimeSetupResult | None:
    """Validate Claude auth in the selected execution backend."""

    command = ["claude", "auth", "status", "--json"]
    stdout_path = task_dir / "claude-auth-status.stdout.log"
    stderr_path = task_dir / "claude-auth-status.stderr.log"
    started_at = time.time()
    result = runtime.run_command(
        command,
        cwd=workspace_path,
        stdin_text=None,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=min(max(int(timeout), 1), 30),
        env=env,
    )
    completed_at = time.time()
    command_text = " ".join(command)
    if not result["ok"]:
        _write_auth_status_summary(stdout_path, logged_in=None)
        _redact_auth_validation_file(stderr_path)
        return RuntimeSetupResult(
            command_result=result,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command_text,
            started_at=started_at,
            completed_at=completed_at,
        )

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        _write_auth_status_summary(stdout_path, logged_in=None)
        _append_auth_validation_error(stderr_path, "Claude auth status did not return valid JSON")
        return RuntimeSetupResult(
            command_result=_failed_auth_result(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command_text,
            started_at=started_at,
            completed_at=completed_at,
        )
    logged_in = payload.get("loggedIn") is True
    _write_auth_status_summary(stdout_path, logged_in=logged_in)
    if not logged_in:
        _append_auth_validation_error(
            stderr_path,
            "Claude Code is not logged in for non-interactive use (`claude auth status` returned loggedIn=false)",
        )
        return RuntimeSetupResult(
            command_result=_failed_auth_result(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command_text,
            started_at=started_at,
            completed_at=completed_at,
        )
    return None


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
    roots = runtime_roots(task_dir)
    apply_copy_paths(copy_paths, roots=roots)
    apply_materialized_files(materialized_files, roots=roots)
    return settings_path, mcp_config_path


def _extract_mcp_server_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(name).strip() for name in value if str(name).strip()}
    if not isinstance(value, list):
        return set()

    names: set[str] = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            if name:
                names.add(name)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("name", "server_name", "serverName", "id"):
            name = str(item.get(key) or "").strip()
            if name:
                names.add(name)
                break
    return names


def _extract_mcp_server_statuses(value: object) -> dict[str, str | None]:
    if isinstance(value, dict):
        return {str(name).strip(): None for name in value if str(name).strip()}
    if not isinstance(value, list):
        return {}

    statuses: dict[str, str | None] = {}
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            if name:
                statuses[name] = None
            continue
        if not isinstance(item, dict):
            continue
        name = ""
        for key in ("name", "server_name", "serverName", "id"):
            name = str(item.get(key) or "").strip()
            if name:
                break
        if not name:
            continue
        status = item.get("status")
        statuses[name] = str(status).strip().lower() if status is not None else None
    return statuses


def configured_mcp_server_names(mcp_config_path: Path) -> set[str]:
    try:
        payload = read_json(mcp_config_path)
    except Exception:
        return set()
    if not isinstance(payload, dict):
        return set()
    return _extract_mcp_server_names(payload.get("mcpServers"))


def validate_isolation(
    raw_response: ClaudeRawResponse,
    *,
    allowed_mcp_servers: Collection[str] = (),
) -> None:
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
    expected_mcp_servers = {str(name).strip() for name in allowed_mcp_servers if str(name).strip()}
    active_mcp_servers = _extract_mcp_server_names(init_event.get("mcp_servers"))
    if active_mcp_servers != expected_mcp_servers:
        missing = sorted(expected_mcp_servers - active_mcp_servers)
        unexpected = sorted(active_mcp_servers - expected_mcp_servers)
        details: list[str] = []
        if missing:
            details.append(f"missing expected MCP servers: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected MCP servers: {', '.join(unexpected)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        raise usage_error(f"Claude isolation failed: MCP server set does not match configured runtime{suffix}")
    mcp_server_statuses = _extract_mcp_server_statuses(init_event.get("mcp_servers"))
    failed_mcp_servers = sorted(
        name
        for name in expected_mcp_servers
        if (status := mcp_server_statuses.get(name)) is not None and status != "connected"
    )
    if failed_mcp_servers:
        details = ", ".join(f"{name}={mcp_server_statuses.get(name)}" for name in failed_mcp_servers)
        raise usage_error(f"Claude isolation failed: configured MCP servers are not connected ({details})")
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


def _collect_response_text(value: object, *, depth: int = 0) -> list[str]:
    if depth > 8 or value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_response_text(item, depth=depth + 1))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for key in ("message", "error", "result", "content"):
            if key in value:
                fragments.extend(_collect_response_text(value.get(key), depth=depth + 1))
        return fragments
    return []


def _raw_response_text(raw_response: ClaudeRawResponse) -> str:
    return "\n".join(_collect_response_text(raw_response.get("response")))


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _archive_persisted_tool_results(
    raw_response: ClaudeRawResponse,
    *,
    task_dir: Path,
) -> list[dict[str, object]]:
    text = _raw_response_text(raw_response)
    matches = list(_PERSISTED_OUTPUT_RE.finditer(text))
    if not matches:
        return []
    output_dir = task_dir / "claude-persisted-tool-results"
    allowed_roots = (task_dir, runtime_root(task_dir))
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches, start=1):
        raw_path = match.group("path").strip()
        if raw_path in seen:
            continue
        seen.add(raw_path)
        source = Path(raw_path)
        label = match.group("label").strip() or None
        entry: dict[str, object] = {
            "source_path": raw_path,
            "artifact_path": None,
            "status": "missing",
            "size_bytes": None,
            "label": label,
        }
        if source.is_file():
            if not any(_path_within(source, root) for root in allowed_roots):
                entry["status"] = "rejected_unsafe_source"
                results.append(entry)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = source.suffix if source.suffix else ".txt"
            name = safe_path_component(source.stem or "tool-result")
            target = output_dir / f"{index:03d}-{name}{suffix}"
            shutil.copy2(source, target)
            entry.update(
                {
                    "artifact_path": str(target),
                    "status": "archived",
                    "size_bytes": source.stat().st_size,
                }
            )
        results.append(entry)

    if results:
        manifest_path = task_dir / "claude-persisted-tool-results.json"
        write_json(manifest_path, results)
    return results


def _classify_diagnostic_note(
    *,
    command_result: CommandResult,
    raw_response: ClaudeRawResponse,
    structured_output: object | None,
    schema_path: Path | None,
) -> str | None:
    try:
        response_text = json.dumps(raw_response, ensure_ascii=False)
    except TypeError:
        response_text = _raw_response_text(raw_response)
    lowered = response_text.lower()
    if "error_max_structured_output_retries" in lowered:
        return (
            "Claude Code hit error_max_structured_output_retries: "
            "structured output could not be produced within Claude Code's retry limit."
        )
    if schema_path is not None and command_result["ok"] and structured_output is None:
        return "Claude Code completed without producing valid structured output."
    return None


def should_retry_failure(
    *,
    command_result: CommandResult,
    raw_response: ClaudeRawResponse,
    stderr_path: Path,
) -> bool:
    if command_result["ok"] or command_result["timeout"]:
        return False
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    haystack = "\n".join((stderr_text, _raw_response_text(raw_response))).lower()
    return any(snippet in haystack for snippet in _RETRYABLE_ERROR_SNIPPETS)


def _build_retry_metadata(
    *,
    attempts: int,
    max_attempts: int,
    events: list[dict[str, object]],
    suppressed: bool,
    suppression_reason: str | None,
) -> dict[str, object]:
    return {
        "attempts": attempts,
        "max_attempts": max_attempts,
        "retried": attempts > 1,
        "suppressed": suppressed,
        "suppression_reason": suppression_reason,
        "events": events,
    }


def run_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    raw_output_filename: str,
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
    retry_dirty_check: Callable[[], bool] | None = None,
) -> CodingAgentInvocationResult:
    parser = ClaudeAgentParser()
    prompt_path = write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    raw_response_path = task_dir / raw_response_filename
    raw_output_path = task_dir / raw_output_filename
    permission_mode = "acceptEdits"
    if execution_backend is not None and getattr(execution_backend.config, "backend", None) == "docker":
        permission_mode = "bypassPermissions"
    command, _ = build_command(
        schema_path=schema_path,
        prompt=prompt,
        model=model,
        reasoning_effort=normalize_reasoning_effort(reasoning_effort),
        extra_args=extra_args,
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
        permission_mode=permission_mode,
    )
    started_at = time.time()
    raw_response: ClaudeRawResponse = {"agent": "claude", "response_format": "stream-json", "response": None}
    command_result: CommandResult = {"ok": False, "exit_code": None, "signal": None, "timeout": False}
    persisted_tool_results: list[dict[str, object]] = []
    max_attempts = len(_RETRY_DELAYS_SECONDS) + 1
    completed_at = started_at
    attempts_used = 0
    retry_events: list[dict[str, object]] = []
    retry_suppressed = False
    retry_suppression_reason: str | None = None
    for attempt_index in range(1, max_attempts + 1):
        attempts_used = attempt_index
        for path in (raw_output_path, stderr_path, raw_response_path):
            if path.exists():
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
        raw_response = build_claude_raw_response(raw_output_path)
        persisted_tool_results = _archive_persisted_tool_results(
            raw_response,
            task_dir=task_dir,
        )
        write_json(raw_response_path, raw_response)
        retryable_failure = should_retry_failure(
            command_result=command_result,
            raw_response=raw_response,
            stderr_path=stderr_path,
        )
        if attempt_index >= max_attempts or not retryable_failure:
            break
        if retry_dirty_check is not None:
            try:
                dirty_after_failure = retry_dirty_check()
            except Exception as exc:
                dirty_after_failure = True
                retry_suppression_reason = f"workspace_dirty_check_failed: {exc}"
            if dirty_after_failure:
                retry_suppressed = True
                retry_suppression_reason = retry_suppression_reason or "workspace_dirty_after_failed_attempt"
                retry_events.append(
                    {
                        "attempt": attempt_index,
                        "action": "suppressed",
                        "reason": retry_suppression_reason,
                    }
                )
                break
        archive_retry_artifacts(
            [raw_output_path, stderr_path, raw_response_path],
            attempt_index=attempt_index,
        )
        retry_events.append(
            {
                "attempt": attempt_index,
                "action": "retry",
                "reason": "transient_failure",
                "delay_seconds": _RETRY_DELAYS_SECONDS[attempt_index - 1],
            }
        )
        time.sleep(_RETRY_DELAYS_SECONDS[attempt_index - 1])
    isolation_diagnostic_note: str | None = None
    if command_result["ok"] and validate_runtime_isolation:
        try:
            validate_isolation(
                raw_response,
                allowed_mcp_servers=configured_mcp_server_names(mcp_config_path),
            )
        except Exception as exc:
            command_result = {
                "ok": False,
                "exit_code": command_result.get("exit_code") if command_result.get("exit_code") not in (None, 0) else 1,
                "signal": command_result.get("signal"),
                "timeout": False,
            }
            isolation_diagnostic_note = str(exc) or "Claude runtime isolation validation failed."
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    schema_validation_note = (
        structured_output_schema_error(structured_output, schema_path)
        if structured_output is not None and schema_path is not None
        else None
    )
    if schema_validation_note:
        structured_output = None
    diagnostic_note = _classify_diagnostic_note(
        command_result=command_result,
        raw_response=raw_response,
        structured_output=structured_output,
        schema_path=schema_path,
    )
    if schema_validation_note:
        diagnostic_note = f"{diagnostic_note} {schema_validation_note}" if diagnostic_note else schema_validation_note
    if isolation_diagnostic_note:
        diagnostic_note = (
            f"{diagnostic_note} {isolation_diagnostic_note}"
            if diagnostic_note
            else isolation_diagnostic_note
        )
    if retry_suppressed:
        retry_note = "Retry suppressed because the failed attempt modified the workspace; preserving the failed record."
        diagnostic_note = f"{diagnostic_note} {retry_note}" if diagnostic_note else retry_note
    return CodingAgentInvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        available_tools=parser.extract_available_tools(raw_response),
        persisted_tool_results=persisted_tool_results,
        diagnostic_note=diagnostic_note,
        retry=_build_retry_metadata(
            attempts=attempts_used,
            max_attempts=max_attempts,
            events=retry_events,
            suppressed=retry_suppressed,
            suppression_reason=retry_suppression_reason,
        ),
        started_at=started_at,
        completed_at=completed_at,
    )
