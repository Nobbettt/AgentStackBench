# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-05-20 from upstream ContextBench.
# Summary of changes: support setup prompts and capture untracked files in benchmark patches.

"""Runtime helpers for Codex and Claude CLI execution."""

from __future__ import annotations

import shutil
import subprocess
import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifact_sanitization import SanitizationContext, assert_no_private_paths, sanitize_json_value
from ..agents.claude.runtime import (
    build_command as build_claude_command,
    prepare_runtime_env as prepare_claude_runtime_env,
    prepare_runtime_files as prepare_claude_runtime_files,
    run_invocation as _run_claude_invocation,
    runtime_root as claude_runtime_root,
    validate_auth as validate_claude_auth,
    validate_isolation as validate_claude_isolation,
)
from ..agents.codex.runtime import (
    build_command as build_codex_command,
    codex_tool_bundle_root,
    prepare_runtime_env as prepare_codex_runtime_env,
    run_invocation as _run_codex_invocation,
    runtime_root as codex_runtime_root,
)
from ..agents.registry import get_coding_agent_adapter
from .constants import DEFAULT_AGENT_RUNTIME_IMAGES
from ..core import checkout
from .files import ensure_dir, safe_path_component, usage_error, write_json
from .prompting import build_prompt
from .records import build_setup_run_record, build_task_record
from .runtime_backends import (
    create_task_runtime,
    docker_checkout_tmp_root,
    normalize_runtime_backend_config,
    run_runtime_setup_commands,
)
from .runtime_common import expand_runtime_templates, run_command, write_prompt_file
from .types import SetupRunRecord, TaskRecord


def _record_path_for_task(*, task_dir: Path, task: dict[str, object], suffix: str) -> Path:
    task_key = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
    return task_dir / f"{task_key}.{suffix}-record.json"


def _public_record_path(record_path: Path) -> Path:
    return record_path.with_name(record_path.name.replace("-record.json", "-record.public.json"))


def resolve_repo_from_task(task: dict[str, object]) -> str:
    repo_url = str(task.get("repo_url") or "").strip()
    if repo_url:
        return repo_url
    return ""


def reset_workspace(workspace_path: Path) -> None:
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workspace_path), check=False, capture_output=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=str(workspace_path), check=False, capture_output=True)


def _git_pathspecs(diff_exclude_paths: Sequence[object] = ()) -> list[str]:
    excludes = []
    for raw_path in diff_exclude_paths or ():
        path = str(raw_path or "").strip()
        if not path:
            continue
        if path.startswith(":"):
            excludes.append(path)
        else:
            excludes.append(f":(exclude){path}")
    return ["--", ".", *excludes] if excludes else []


def git_diff(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> str:
    pathspecs = _git_pathspecs(diff_exclude_paths)
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", *pathspecs],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout or ""


def git_staged_diff(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> str:
    pathspecs = _git_pathspecs(diff_exclude_paths)
    result = subprocess.run(
        ["git", "diff", "--cached", "--no-ext-diff", "--binary", *pathspecs],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout or ""


def git_tracked_diff(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> str:
    if diff_exclude_paths:
        return git_staged_diff(workspace_path, diff_exclude_paths=diff_exclude_paths) + git_diff(
            workspace_path,
            diff_exclude_paths=diff_exclude_paths,
        )
    return git_staged_diff(workspace_path) + git_diff(workspace_path)


def git_untracked_files(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> list[str]:
    pathspecs = _git_pathspecs(diff_exclude_paths)
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", *pathspecs],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raw_detail = result.stderr or result.stdout or b""
        detail = (
            raw_detail.decode("utf-8", "replace")
            if isinstance(raw_detail, bytes)
            else str(raw_detail)
        ).strip()
        message = f"git ls-files failed while checking untracked files in {workspace_path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    return sorted(
        {
            path.decode("utf-8", "surrogateescape")
            for path in (result.stdout or b"").split(b"\0")
            if path
        }
    )


def _looks_like_not_git_repository(message: str) -> bool:
    return "not a git repository" in message.lower()


def git_workspace_diff(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> str:
    tracked_diff = git_tracked_diff(workspace_path, diff_exclude_paths=diff_exclude_paths)
    try:
        untracked_files = git_untracked_files(workspace_path, diff_exclude_paths=diff_exclude_paths)
    except RuntimeError as exc:
        if tracked_diff.strip() or _looks_like_not_git_repository(str(exc)):
            return tracked_diff
        raise
    if not untracked_files:
        return tracked_diff

    add_result = subprocess.run(
        ["git", "add", "--intent-to-add", "--", *untracked_files],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        detail = (add_result.stderr or add_result.stdout or "").strip()
        if _looks_like_not_git_repository(detail):
            return tracked_diff
        message = f"git add --intent-to-add failed while preparing workspace diff in {workspace_path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    diff_text = ""
    try:
        diff_text = git_tracked_diff(workspace_path, diff_exclude_paths=diff_exclude_paths)
    finally:
        reset_result = subprocess.run(
            ["git", "reset", "--", *untracked_files],
            cwd=str(workspace_path),
            check=False,
            capture_output=True,
            text=True,
        )
    if reset_result.returncode != 0:
        detail = (reset_result.stderr or reset_result.stdout or "").strip()
        message = f"git reset failed while restoring workspace diff state in {workspace_path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    return diff_text


def workspace_has_nonexcluded_changes(workspace_path: Path, *, diff_exclude_paths: Sequence[object] = ()) -> bool:
    diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=diff_exclude_paths)
    if diff_text.strip():
        return True
    return bool(git_untracked_files(workspace_path, diff_exclude_paths=diff_exclude_paths))


def summarize_tool_calls(tool_calls: Sequence[dict[str, object]] | None) -> dict[str, object]:
    by_name: dict[str, int] = {}
    by_source: dict[str, int] = {}
    mcp_by_server: dict[str, int] = {}
    mcp_by_tool: dict[str, int] = {}
    successful_by_name: dict[str, int] = {}
    mcp_successful_by_server: dict[str, int] = {}
    mcp_successful_by_tool: dict[str, int] = {}
    total = 0
    successful_total = 0
    mcp_total = 0
    mcp_successful_total = 0
    for call in tool_calls or ():
        if not isinstance(call, dict):
            continue
        total += 1
        name = str(call.get("tool_name") or "unknown")
        source = str(call.get("source") or "unknown")
        successful = tool_call_succeeded(call)
        if successful:
            successful_total += 1
            successful_by_name[name] = successful_by_name.get(name, 0) + 1
        by_name[name] = by_name.get(name, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
        payload = call.get("payload")
        if not isinstance(payload, dict):
            continue
        server = str(payload.get("mcp_server") or "").strip()
        tool = str(payload.get("mcp_tool") or "").strip()
        if not server and name.startswith("mcp__"):
            parts = name.split("__", 2)
            if len(parts) >= 3:
                server = parts[1]
                tool = parts[2]
        if not server:
            continue
        mcp_total += 1
        mcp_by_server[server] = mcp_by_server.get(server, 0) + 1
        if tool:
            key = f"{server}/{tool}"
            mcp_by_tool[key] = mcp_by_tool.get(key, 0) + 1
        if successful:
            mcp_successful_total += 1
            mcp_successful_by_server[server] = mcp_successful_by_server.get(server, 0) + 1
            if tool:
                key = f"{server}/{tool}"
                mcp_successful_by_tool[key] = mcp_successful_by_tool.get(key, 0) + 1
    return {
        "total": total,
        "successful_total": successful_total,
        "failed_total": max(total - successful_total, 0),
        "by_name": dict(sorted(by_name.items())),
        "successful_by_name": dict(sorted(successful_by_name.items())),
        "by_source": dict(sorted(by_source.items())),
        "mcp_total": mcp_total,
        "mcp_successful_total": mcp_successful_total,
        "mcp_failed_total": max(mcp_total - mcp_successful_total, 0),
        "mcp_by_server": dict(sorted(mcp_by_server.items())),
        "mcp_by_tool": dict(sorted(mcp_by_tool.items())),
        "mcp_successful_by_server": dict(sorted(mcp_successful_by_server.items())),
        "mcp_successful_by_tool": dict(sorted(mcp_successful_by_tool.items())),
    }


def tool_call_succeeded(call: dict[str, object]) -> bool:
    if call.get("ok") is False:
        return False
    payload = call.get("payload")
    if not isinstance(payload, dict):
        return True
    if payload.get("ok") is False:
        return False
    result = payload.get("result")
    if isinstance(result, dict) and result.get("ok") is False:
        return False
    if isinstance(result, dict) and "is_error" in result:
        return not bool(result.get("is_error"))
    if "is_error" in payload:
        return not bool(payload.get("is_error"))
    status = str(payload.get("status") or "").strip().lower()
    if status in {"cancelled", "canceled", "denied", "error", "failed", "failure", "rejected", "timeout"}:
        return False
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    if isinstance(exit_code, str) and exit_code.strip() and exit_code.strip() != "0":
        return False
    return True


def missing_required_tool_call_patterns(
    tool_calls: Sequence[dict[str, object]] | None,
    required_tool_call_patterns: Sequence[object] = (),
) -> list[str]:
    patterns = [str(pattern).strip() for pattern in required_tool_call_patterns or () if str(pattern).strip()]
    if not patterns:
        return []
    haystacks: list[str] = []
    for call in tool_calls or ():
        if not isinstance(call, dict):
            continue
        if not tool_call_succeeded(call):
            continue
        try:
            payload_text = json.dumps(call.get("payload") or {}, sort_keys=True)
        except TypeError:
            payload_text = str(call.get("payload") or {})
        source = str(call.get("source") or "")
        tool_name = str(call.get("tool_name") or "")
        haystacks.extend((source, tool_name, payload_text, "\n".join((source, tool_name, payload_text))))
    missing: list[str] = []
    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise usage_error(f"Invalid required tool call pattern {pattern!r}: {exc}") from exc
        if not any(compiled.search(haystack) for haystack in haystacks):
            missing.append(pattern)
    return missing


def command_execution_succeeded(execution: dict[str, object]) -> bool:
    payload = execution.get("payload")
    if not isinstance(payload, dict):
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in {"failed", "error", "cancelled", "canceled"}:
        return False
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    if isinstance(exit_code, str) and exit_code.strip() and exit_code.strip() != "0":
        return False
    return True


def missing_required_command_patterns(
    command_executions: Sequence[dict[str, object]] | None,
    required_command_patterns: Sequence[object] = (),
) -> list[str]:
    patterns = [str(pattern).strip() for pattern in required_command_patterns or () if str(pattern).strip()]
    if not patterns:
        return []
    haystacks: list[str] = []
    for execution in command_executions or ():
        if not isinstance(execution, dict):
            continue
        if not command_execution_succeeded(execution):
            continue
        try:
            payload_text = json.dumps(execution.get("payload") or {}, sort_keys=True)
        except TypeError:
            payload_text = str(execution.get("payload") or {})
        source = str(execution.get("source") or "")
        command = str(execution.get("command") or "")
        haystacks.extend((source, command, payload_text, "\n".join((source, command, payload_text))))
    missing: list[str] = []
    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise usage_error(f"Invalid required command pattern {pattern!r}: {exc}") from exc
        if not any(compiled.search(haystack) for haystack in haystacks):
            missing.append(pattern)
    return missing


def missing_required_available_tool_patterns(
    available_tools: Sequence[object] | None,
    required_available_tool_patterns: Sequence[object] = (),
) -> list[str]:
    patterns = [str(pattern).strip() for pattern in required_available_tool_patterns or () if str(pattern).strip()]
    if not patterns:
        return []
    haystacks = [str(tool or "").strip() for tool in available_tools or () if str(tool or "").strip()]
    missing: list[str] = []
    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise usage_error(f"Invalid required available tool pattern {pattern!r}: {exc}") from exc
        if not any(compiled.search(haystack) for haystack in haystacks):
            missing.append(pattern)
    return missing


def build_setup_contamination_record(
    *,
    task: dict[str, object],
    agent: str,
    workspace_path: Path,
    task_dir: Path,
    prompt_path: Path,
    adapter_record_suffix: str,
    started_at: float,
    setup_run: SetupRunRecord | None = None,
    diff_exclude_paths: Sequence[object] = (),
) -> TaskRecord | None:
    del adapter_record_suffix
    diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=diff_exclude_paths)
    untracked_files = git_untracked_files(workspace_path, diff_exclude_paths=diff_exclude_paths)
    if not diff_text.strip() and not untracked_files:
        return None
    diff_path = task_dir / "workspace.diff"
    diff_path_value: Path | None = None
    if diff_text.strip():
        diff_path.write_text(diff_text, encoding="utf-8")
        diff_path_value = diff_path
    record = build_task_record(
        task=task,
        agent=agent,
        workspace_path=workspace_path,
        task_dir=task_dir,
        prompt_path=prompt_path,
        command_result={"ok": False, "exit_code": 1, "signal": None, "timeout": False},
        structured_output=None,
        token_usage=None,
        tool_calls=[],
        raw_response_path=None,
        diff_path=diff_path_value,
        model_patch=diff_text,
        started_at=started_at,
        completed_at=time.time(),
        setup_run=setup_run,
    )
    record["status"] = "failed"
    record["notes"] = "Unscored setup modified tracked files before the scored prompt."
    record["setup_contamination"] = {
        "tracked_diff": bool(diff_text.strip()),
        "untracked_files": untracked_files,
    }
    if untracked_files and not diff_text.strip():
        record["notes"] = "Unscored setup created untracked files before the scored prompt."
    return record


def scrub_runtime_secrets(*, agent: str, task_dir: Path) -> None:
    if agent == "codex":
        shutil.rmtree(codex_runtime_root(task_dir), ignore_errors=True)
        return
    if agent == "claude":
        shutil.rmtree(claude_runtime_root(task_dir), ignore_errors=True)


def _runtime_failure_metadata(
    *,
    phase: str,
    command: str,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    return {
        "phase": phase,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


_RUNTIME_SETUP_CACHE_VERSION = 1
_RUNTIME_SETUP_CACHE_IGNORES = frozenset({".git"})
_RUNTIME_SETUP_CACHE_PROMPT_ONLY_SETUP_KEYS = frozenset(
    {
        "prompt_preamble",
        "setup_prompt",
        "setup_prompt_timeout",
    }
)


def _runtime_setup_cache_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _runtime_setup_cache_jsonable(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_runtime_setup_cache_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _runtime_setup_cache_setup_fingerprint(setup: dict[str, Any]) -> dict[str, object]:
    return {
        str(key): _runtime_setup_cache_jsonable(value)
        for key, value in sorted(setup.items(), key=lambda entry: str(entry[0]))
        if key not in _RUNTIME_SETUP_CACHE_PROMPT_ONLY_SETUP_KEYS
    }


def _runtime_setup_cache_key_inputs(
    *,
    task: dict[str, object],
    agent: str,
    runtime_config: object,
    setup: dict[str, Any],
    setup_commands: Sequence[str],
    validation_commands: Sequence[str],
    env_overrides: dict[str, str],
    schema_path: Path,
    diff_exclude_paths: Sequence[str],
) -> dict[str, object]:
    return {
        "version": _RUNTIME_SETUP_CACHE_VERSION,
        "agent": agent,
        "repo_url": task.get("repo_url"),
        "commit": task.get("commit"),
        "runtime_backend": getattr(runtime_config, "backend", None),
        "runtime_image": getattr(runtime_config, "image", None),
        "runtime_platform": getattr(runtime_config, "platform", None),
        "runtime_env": _runtime_setup_cache_jsonable(getattr(runtime_config, "env", {})),
        "env_overrides": _runtime_setup_cache_jsonable(
            {
                key: value
                for key, value in sorted(env_overrides.items())
                if key not in {"CONTEXTBENCH_TASK_DIR", "CONTEXTBENCH_WORKSPACE_PATH"}
            }
        ),
        "setup": _runtime_setup_cache_setup_fingerprint(setup),
        "setup_commands": list(setup_commands),
        "validation_commands": list(validation_commands),
        "schema_path": str(schema_path.resolve()),
        "diff_exclude_paths": list(diff_exclude_paths),
    }


def _runtime_setup_cache_key(key_inputs: dict[str, object]) -> str:
    encoded = json.dumps(key_inputs, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_setup_cache_ignore(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _RUNTIME_SETUP_CACHE_IGNORES}


def _restore_runtime_setup_cache(*, entry_dir: Path, workspace_path: Path, expected_key: str) -> bool:
    if not entry_dir.exists():
        return False
    manifest_path = entry_dir / "manifest.json"
    workspace_snapshot = entry_dir / "workspace"
    if not manifest_path.exists() or not workspace_snapshot.is_dir():
        raise RuntimeError(f"Runtime setup cache entry is incomplete: {entry_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Runtime setup cache manifest is invalid: {manifest_path}") from exc
    if manifest.get("key") != expected_key:
        raise RuntimeError(f"Runtime setup cache manifest key mismatch: {manifest_path}")
    shutil.copytree(
        workspace_snapshot,
        workspace_path,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=_runtime_setup_cache_ignore,
    )
    return True


def _save_runtime_setup_cache(
    *,
    entry_dir: Path,
    workspace_path: Path,
    key: str,
    key_inputs: dict[str, object],
) -> None:
    ensure_dir(entry_dir.parent)
    temp_dir = entry_dir.with_name(f"{entry_dir.name}.tmp-{time.time_ns()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        ensure_dir(temp_dir)
        shutil.copytree(
            workspace_path,
            temp_dir / "workspace",
            symlinks=True,
            ignore=_runtime_setup_cache_ignore,
        )
        write_json(
            temp_dir / "manifest.json",
            {
                "version": _RUNTIME_SETUP_CACHE_VERSION,
                "key": key,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "key_inputs": key_inputs,
            },
        )
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        temp_dir.rename(entry_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def run_coding_agent_task(
    *,
    task: dict[str, object],
    agent: str,
    output_dir: Path,
    cache_dir: Path,
    schema_path: Path,
    timeout: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    agent_args: Sequence[str] = (),
    env_overrides: dict[str, str] | None = None,
    prompt_preamble: str | None = None,
    setup: dict[str, object] | None = None,
    workspace_key: str | None = None,
    runtime_backend: str | None = None,
    runtime_image: str | None = None,
    runtime_platform: str | None = None,
    runtime_env: dict[str, object] | None = None,
    runtime_setup_timeout: int | None = None,
    runtime_validation_timeout: int | None = None,
    runtime_setup_cache: bool = False,
    runtime_setup_cache_dir: Path | None = None,
    runtime_setup_commands: Sequence[object] = (),
    runtime_validation_commands: Sequence[object] = (),
    diff_exclude_paths: Sequence[object] = (),
    required_tool_call_patterns: Sequence[object] = (),
    required_command_patterns: Sequence[object] = (),
    required_available_tool_patterns: Sequence[object] = (),
    runtime_keep_failed: bool = False,
) -> TaskRecord:
    try:
        adapter = get_coding_agent_adapter(agent)
    except ValueError as exc:
        raise usage_error(str(exc)) from exc
    agent = adapter.name
    if runtime_backend is None:
        raise usage_error("runtime_backend is required for coding-agent task execution.")
    if runtime_backend == "docker" and runtime_image is None:
        runtime_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(agent)
        if runtime_image is None:
            raise usage_error(f"No pinned Docker runtime image is configured for agent {agent!r}")
    runtime_config = normalize_runtime_backend_config(
        runtime_backend=runtime_backend,
        runtime_image=runtime_image,
        runtime_platform=runtime_platform,
        runtime_env=runtime_env,
        runtime_setup_commands=runtime_setup_commands,
        runtime_validation_commands=runtime_validation_commands,
        runtime_keep_failed=runtime_keep_failed,
    )
    normalized_diff_exclude_paths = tuple(str(path).strip() for path in diff_exclude_paths or () if str(path).strip())
    normalized_required_tool_call_patterns = tuple(
        str(pattern).strip() for pattern in required_tool_call_patterns or () if str(pattern).strip()
    )
    normalized_required_command_patterns = tuple(
        str(pattern).strip() for pattern in required_command_patterns or () if str(pattern).strip()
    )
    normalized_required_available_tool_patterns = tuple(
        str(pattern).strip() for pattern in required_available_tool_patterns or () if str(pattern).strip()
    )
    if runtime_setup_timeout is not None:
        if isinstance(runtime_setup_timeout, bool) or not isinstance(runtime_setup_timeout, int) or runtime_setup_timeout <= 0:
            raise usage_error("runtime_setup_timeout must be a positive integer when provided")
    if runtime_validation_timeout is not None:
        if (
            isinstance(runtime_validation_timeout, bool)
            or not isinstance(runtime_validation_timeout, int)
            or runtime_validation_timeout <= 0
        ):
            raise usage_error("runtime_validation_timeout must be a positive integer when provided")
    setup_command_timeout = runtime_setup_timeout or timeout
    validation_command_timeout = runtime_validation_timeout or timeout

    repo_url = resolve_repo_from_task(task)
    if not repo_url:
        task_id = task.get("instance_id") or task.get("original_inst_id")
        raise usage_error(
            f"Task {task_id!r} is missing required repo_url. "
            "Benchmark task execution does not guess repositories from instance ids; "
            "provide repo_url in the selected task data."
        )

    task = dict(task)
    task["repo_url"] = repo_url
    workspace = checkout(
        repo_url,
        task.get("commit") or "",
        str(cache_dir),
        verbose=True,
        workspace_key=workspace_key,
        tmp_root=str(docker_checkout_tmp_root(cache_dir)) if runtime_config.backend == "docker" else None,
    )
    if not workspace:
        raise usage_error(f"Checkout failed for {task.get('instance_id') or task.get('original_inst_id')}")

    workspace_path = Path(workspace)
    reset_workspace(workspace_path)

    task_dir = (output_dir / safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")).resolve()
    ensure_dir(task_dir)
    extra_runtime_mounts: list[Path] = []
    extra_runtime_readonly_mounts: list[Path] = []
    if agent == "codex":
        codex_runtime_dir = codex_runtime_root(task_dir)
        shutil.rmtree(codex_runtime_dir, ignore_errors=True)
        ensure_dir(codex_runtime_dir)
        extra_runtime_mounts.append(codex_runtime_dir)
    elif agent == "claude":
        claude_runtime_dir = claude_runtime_root(task_dir)
        shutil.rmtree(claude_runtime_dir, ignore_errors=True)
        ensure_dir(claude_runtime_dir)
        extra_runtime_mounts.append(claude_runtime_dir)

    runtime_env_template_env = {
        **dict(runtime_config.env or {}),
        "CONTEXTBENCH_WORKSPACE_PATH": str(workspace_path),
        "CONTEXTBENCH_TASK_DIR": str(task_dir),
    }
    expanded_runtime_env = expand_runtime_templates(dict(runtime_config.env or {}), env=runtime_env_template_env)
    runtime_config = replace(
        runtime_config,
        env={str(key): str(value) for key, value in dict(expanded_runtime_env).items()},
    )
    if agent == "codex" and runtime_config.backend == "docker":
        bundle_root = codex_tool_bundle_root(runtime_config.env)
        if bundle_root is not None:
            extra_runtime_readonly_mounts.append(bundle_root)

    prompt = build_prompt(task, agent)
    if prompt_preamble:
        prompt = prompt_preamble.rstrip() + "\n\n" + prompt
    prompt_path = write_prompt_file(task_dir, "prompt.txt", prompt)

    setup_dict: dict[str, Any] = dict(setup or {})
    copy_paths = setup_dict.get("copy_paths")
    materialized_files = setup_dict.get("files_to_materialize")
    setup_prompt = str(setup_dict.get("setup_prompt") or "").strip()
    setup_timeout_value = setup_dict.get("setup_prompt_timeout")
    setup_timeout = timeout
    if setup_prompt and setup_timeout_value is not None:
        if isinstance(setup_timeout_value, bool) or not isinstance(setup_timeout_value, int) or setup_timeout_value <= 0:
            raise usage_error("setup_prompt_timeout must be a positive integer when provided")
        setup_timeout = setup_timeout_value

    task_runtime = create_task_runtime(
        runtime_config,
        workspace_path=workspace_path,
        task_dir=task_dir,
        schema_path=schema_path,
        extra_writable_dirs=extra_runtime_mounts,
        extra_readonly_dirs=extra_runtime_readonly_mounts,
    )
    runtime_success = False
    runtime_closed = False
    runtime_metadata: dict[str, object] = {"backend": runtime_config.backend}
    runtime_setup_cache_record: dict[str, object] | None = None

    def write_record(record: TaskRecord) -> TaskRecord:
        record["runtime"] = runtime_metadata
        if runtime_setup_cache_record is not None:
            record["runtime_setup_cache"] = dict(runtime_setup_cache_record)
        record_path = _record_path_for_task(task_dir=task_dir, task=task, suffix=adapter.record_suffix)
        write_json(record_path, record)
        public_context = SanitizationContext(
            repo_root=Path.cwd().resolve(),
            suite_dir=output_dir.resolve(),
            workspace_path=workspace_path,
            task_dir=task_dir,
        )
        public_record = sanitize_json_value(record, context=public_context)
        assert_no_private_paths(public_record, label=str(_public_record_path(record_path)))
        write_json(_public_record_path(record_path), public_record)
        return record

    try:
        task_runtime.start()
        metadata_fn = getattr(task_runtime, "metadata", None)
        if callable(metadata_fn):
            runtime_metadata = metadata_fn()
        effective_env_overrides = dict(env_overrides or {})
        effective_env_overrides.setdefault("CONTEXTBENCH_WORKSPACE_PATH", str(workspace_path))
        effective_env_overrides.setdefault("CONTEXTBENCH_TASK_DIR", str(task_dir))
        template_env = {
            **dict(runtime_config.env or {}),
            **effective_env_overrides,
            "CONTEXTBENCH_WORKSPACE_PATH": str(workspace_path),
            "CONTEXTBENCH_TASK_DIR": str(task_dir),
        }
        setup_dict = expand_runtime_templates(setup_dict, env=template_env)

        prepared_runtime = adapter.prepare_runtime(
            task_dir=task_dir,
            setup=setup_dict,
            env_overrides=effective_env_overrides or None,
            runtime_backend=runtime_config.backend,
            runtime_env=runtime_config.env,
        )
        prepared_runtime = replace(prepared_runtime, execution_backend=task_runtime)
        adapter_validation_failure = adapter.validate_pre_invocation_runtime(
            task_dir=task_dir,
            workspace_path=workspace_path,
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=tuple(agent_args),
            prepared_runtime=prepared_runtime,
        )
        if adapter_validation_failure is not None:
            if adapter_validation_failure.command_result["timeout"] and runtime_config.backend == "docker":
                task_runtime.close(success=False)
                runtime_closed = True
            started_at = adapter_validation_failure.started_at
            completed_at = adapter_validation_failure.completed_at
            diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            untracked_files = git_untracked_files(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            diff_path: Path | None = None
            if diff_text.strip():
                diff_path = task_dir / "workspace.diff"
                diff_path.write_text(diff_text, encoding="utf-8")
            record = build_task_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                command_result=adapter_validation_failure.command_result,
                structured_output=None,
                token_usage=None,
                tool_calls=[],
                raw_response_path=None,
                diff_path=diff_path,
                model_patch=diff_text,
                started_at=started_at,
                completed_at=completed_at,
                setup_run=None,
            )
            record["notes"] = "Adapter runtime validation failed before setup prompts or scored work."
            record["runtime_failure"] = _runtime_failure_metadata(
                phase="adapter-validation",
                command=adapter_validation_failure.command,
                stdout_path=adapter_validation_failure.stdout_path,
                stderr_path=adapter_validation_failure.stderr_path,
            )
            record["setup_contamination"] = {
                "tracked_diff": bool(diff_text.strip()),
                "untracked_files": untracked_files,
            }
            return write_record(record)
        runtime_template_env = {
            **dict(runtime_config.env or {}),
            **effective_env_overrides,
            "CONTEXTBENCH_WORKSPACE_PATH": str(workspace_path),
            "CONTEXTBENCH_TASK_DIR": str(task_dir),
        }
        setup_commands = tuple(
            str(command)
            for command in expand_runtime_templates(list(runtime_config.setup_commands), env=runtime_template_env)
        )
        validation_commands = tuple(
            str(command)
            for command in expand_runtime_templates(list(runtime_config.validation_commands), env=runtime_template_env)
        )
        if runtime_setup_cache and (setup_commands or validation_commands):
            cache_root = (runtime_setup_cache_dir or (cache_dir.parent / "runtime-setup-cache")).resolve()
            cache_key_inputs = _runtime_setup_cache_key_inputs(
                task=task,
                agent=agent,
                runtime_config=runtime_config,
                setup=setup_dict,
                setup_commands=setup_commands,
                validation_commands=validation_commands,
                env_overrides=effective_env_overrides,
                schema_path=schema_path,
                diff_exclude_paths=normalized_diff_exclude_paths,
            )
            cache_key = _runtime_setup_cache_key(cache_key_inputs)
            cache_entry_dir = cache_root / cache_key
            cache_hit = _restore_runtime_setup_cache(
                entry_dir=cache_entry_dir,
                workspace_path=workspace_path,
                expected_key=cache_key,
            )
            runtime_setup_cache_record = {
                "enabled": True,
                "key": cache_key,
                "cache_dir": str(cache_root),
                "entry_dir": str(cache_entry_dir),
                "hit": cache_hit,
                "restored": cache_hit,
                "saved": False,
            }

        runtime_setup_failure = run_runtime_setup_commands(
            task_runtime,
            commands=setup_commands,
            workspace_path=workspace_path,
            task_dir=task_dir,
            timeout=setup_command_timeout,
            env=prepared_runtime.env,
        )
        if runtime_setup_failure is not None:
            if runtime_setup_failure.command_result["timeout"] and runtime_config.backend == "docker":
                task_runtime.close(success=False)
                runtime_closed = True
            diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            untracked_files = git_untracked_files(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            diff_path: Path | None = None
            if diff_text.strip():
                diff_path = task_dir / "workspace.diff"
                diff_path.write_text(diff_text, encoding="utf-8")
            record = build_task_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                command_result=runtime_setup_failure.command_result,
                structured_output=None,
                token_usage=None,
                tool_calls=[],
                raw_response_path=None,
                diff_path=diff_path,
                model_patch=diff_text,
                started_at=runtime_setup_failure.started_at,
                completed_at=runtime_setup_failure.completed_at,
                setup_run=None,
            )
            record["notes"] = "Runtime setup command failed before setup prompts or scored work."
            record["runtime_failure"] = _runtime_failure_metadata(
                phase="runtime-setup",
                command=runtime_setup_failure.command,
                stdout_path=runtime_setup_failure.stdout_path,
                stderr_path=runtime_setup_failure.stderr_path,
            )
            record["setup_contamination"] = {
                "tracked_diff": bool(diff_text.strip()),
                "untracked_files": untracked_files,
            }
            return write_record(record)
        runtime_validation_failure = run_runtime_setup_commands(
            task_runtime,
            commands=validation_commands,
            workspace_path=workspace_path,
            task_dir=task_dir,
            timeout=validation_command_timeout,
            env=prepared_runtime.env,
            artifact_prefix="runtime-validation",
        )
        if runtime_validation_failure is not None:
            if runtime_validation_failure.command_result["timeout"] and runtime_config.backend == "docker":
                task_runtime.close(success=False)
                runtime_closed = True
            diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            untracked_files = git_untracked_files(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
            diff_path: Path | None = None
            if diff_text.strip():
                diff_path = task_dir / "workspace.diff"
                diff_path.write_text(diff_text, encoding="utf-8")
            record = build_task_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                command_result=runtime_validation_failure.command_result,
                structured_output=None,
                token_usage=None,
                tool_calls=[],
                raw_response_path=None,
                diff_path=diff_path,
                model_patch=diff_text,
                started_at=runtime_validation_failure.started_at,
                completed_at=runtime_validation_failure.completed_at,
                setup_run=None,
            )
            record["notes"] = "Runtime validation command failed before the scored prompt."
            record["runtime_failure"] = _runtime_failure_metadata(
                phase="runtime-validation",
                command=runtime_validation_failure.command,
                stdout_path=runtime_validation_failure.stdout_path,
                stderr_path=runtime_validation_failure.stderr_path,
            )
            record["setup_contamination"] = {
                "tracked_diff": bool(diff_text.strip()),
                "untracked_files": untracked_files,
            }
            return write_record(record)
        if setup_commands or validation_commands:
            contamination_record = build_setup_contamination_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                adapter_record_suffix=adapter.record_suffix,
                started_at=time.time(),
                diff_exclude_paths=normalized_diff_exclude_paths,
            )
            if contamination_record is not None:
                return write_record(contamination_record)
            if runtime_setup_cache_record is not None:
                _save_runtime_setup_cache(
                    entry_dir=Path(str(runtime_setup_cache_record["entry_dir"])),
                    workspace_path=workspace_path,
                    key=str(runtime_setup_cache_record["key"]),
                    key_inputs=cache_key_inputs,
                )
                runtime_setup_cache_record["saved"] = True

        setup_run: SetupRunRecord | None = None
        if setup_prompt:
            setup_result = adapter.run_setup_invocation(
                task_dir=task_dir,
                workspace_path=workspace_path,
                prompt=setup_prompt,
                timeout=setup_timeout,
                model=model,
                reasoning_effort=reasoning_effort,
                extra_args=tuple(agent_args),
                prepared_runtime=prepared_runtime,
                retry_dirty_check=lambda: workspace_has_nonexcluded_changes(
                    workspace_path,
                    diff_exclude_paths=normalized_diff_exclude_paths,
                ),
            )
            setup_run = build_setup_run_record(
                prompt_path=setup_result.prompt_path,
                stderr_path=setup_result.stderr_path,
                command_result=setup_result.command_result,
                raw_response_path=setup_result.raw_response_path,
                token_usage=setup_result.token_usage,
                tool_calls=setup_result.tool_calls,
                persisted_tool_results=setup_result.persisted_tool_results,
                retry=setup_result.retry,
                started_at=setup_result.started_at,
                completed_at=setup_result.completed_at,
            )
            if not setup_result.command_result["ok"]:
                if setup_result.command_result["timeout"] and runtime_config.backend == "docker":
                    task_runtime.close(success=False)
                    runtime_closed = True
                diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
                diff_path: Path | None = None
                if diff_text.strip():
                    diff_path = task_dir / "workspace.diff"
                    diff_path.write_text(diff_text, encoding="utf-8")
                record = build_task_record(
                    task=task,
                    agent=agent,
                    workspace_path=workspace_path,
                    task_dir=task_dir,
                    prompt_path=prompt_path,
                    command_result=setup_result.command_result,
                    structured_output=None,
                    token_usage=None,
                    tool_calls=[],
                    raw_response_path=None,
                    diff_path=diff_path,
                    model_patch=diff_text,
                    started_at=setup_result.started_at,
                    completed_at=setup_result.completed_at,
                    setup_run=setup_run,
                )
                if setup_result.diagnostic_note:
                    record["notes"] = setup_result.diagnostic_note
                return write_record(record)
            contamination_record = build_setup_contamination_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                adapter_record_suffix=adapter.record_suffix,
                started_at=setup_result.started_at,
                setup_run=setup_run,
                diff_exclude_paths=normalized_diff_exclude_paths,
            )
            if contamination_record is not None:
                return write_record(contamination_record)

        main_result = adapter.run_main_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=tuple(agent_args),
            schema_path=schema_path,
            prepared_runtime=prepared_runtime,
            retry_dirty_check=lambda: workspace_has_nonexcluded_changes(
                workspace_path,
                diff_exclude_paths=normalized_diff_exclude_paths,
            ),
        )

        if main_result.command_result["timeout"] and runtime_config.backend == "docker":
            task_runtime.close(success=False)
            runtime_closed = True
        diff_text = git_workspace_diff(workspace_path, diff_exclude_paths=normalized_diff_exclude_paths)
        diff_path: Path | None = None
        if diff_text.strip():
            diff_path = task_dir / "workspace.diff"
            diff_path.write_text(diff_text, encoding="utf-8")

        record = build_task_record(
            task=task,
            agent=agent,
            workspace_path=workspace_path,
            task_dir=task_dir,
            prompt_path=main_result.prompt_path,
            command_result=main_result.command_result,
            structured_output=main_result.structured_output,
            token_usage=main_result.token_usage,
            tool_calls=main_result.tool_calls,
            raw_response_path=main_result.raw_response_path,
            diff_path=diff_path,
            model_patch=diff_text,
            started_at=main_result.started_at,
            completed_at=main_result.completed_at,
            setup_run=setup_run,
        )
        record["available_tools"] = list(main_result.available_tools)
        record["command_executions"] = list(main_result.command_executions)
        if main_result.persisted_tool_results:
            record["persisted_tool_results"] = list(main_result.persisted_tool_results)
        record["retry"] = dict(main_result.retry)
        record["tool_call_summary"] = summarize_tool_calls(main_result.tool_calls)
        if (
            schema_path is not None
            and main_result.command_result["ok"]
            and main_result.structured_output is None
        ):
            record["ok"] = False
            record["status"] = "failed"
            record["notes"] = main_result.diagnostic_note or "Agent completed without producing valid structured output."
        elif main_result.diagnostic_note:
            record["notes"] = main_result.diagnostic_note
        if normalized_required_available_tool_patterns:
            missing_patterns = missing_required_available_tool_patterns(
                main_result.available_tools,
                normalized_required_available_tool_patterns,
            )
            record["tool_availability_requirements"] = {
                "patterns": list(normalized_required_available_tool_patterns),
                "missing": missing_patterns,
                "ok": not missing_patterns,
            }
            if missing_patterns and str(record.get("status") or "") not in {"failed", "timeout"}:
                record["ok"] = False
                record["status"] = "failed"
                record["notes"] = (
                    "Required available tool patterns were not observed: "
                    + ", ".join(missing_patterns)
                )
        if normalized_required_tool_call_patterns:
            missing_patterns = missing_required_tool_call_patterns(
                main_result.tool_calls,
                normalized_required_tool_call_patterns,
            )
            record["tool_call_requirements"] = {
                "patterns": list(normalized_required_tool_call_patterns),
                "missing": missing_patterns,
                "ok": not missing_patterns,
            }
            if missing_patterns and str(record.get("status") or "") not in {"failed", "timeout"}:
                record["ok"] = False
                record["status"] = "failed"
                record["notes"] = (
                    "Required tool call patterns were not observed: "
                    + ", ".join(missing_patterns)
                )
        if normalized_required_command_patterns:
            missing_patterns = missing_required_command_patterns(
                main_result.command_executions,
                normalized_required_command_patterns,
            )
            record["command_requirements"] = {
                "patterns": list(normalized_required_command_patterns),
                "missing": missing_patterns,
                "ok": not missing_patterns,
            }
            if missing_patterns and str(record.get("status") or "") not in {"failed", "timeout"}:
                record["ok"] = False
                record["status"] = "failed"
                record["notes"] = (
                    "Required command patterns were not observed: "
                    + ", ".join(missing_patterns)
                )

        runtime_success = str(record.get("status") or "") == "completed" and not record.get("timeout")
        return write_record(record)
    finally:
        if not runtime_closed:
            task_runtime.close(success=runtime_success)
        scrub_runtime_secrets(agent=agent, task_dir=task_dir)
