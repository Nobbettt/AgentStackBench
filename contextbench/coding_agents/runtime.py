# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: support unscored setup prompts that run before the scored benchmark prompt.

"""Runtime helpers for Codex and Claude CLI execution."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifact_sanitization import SanitizationContext, assert_no_private_paths, sanitize_json_value
from ..agents.claude.runtime import (
    build_command as build_claude_command,
    prepare_runtime_files as prepare_claude_runtime_files,
    run_invocation as _run_claude_invocation,
    validate_auth as validate_claude_auth,
    validate_isolation as validate_claude_isolation,
)
from ..agents.codex.runtime import (
    build_command as build_codex_command,
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
from .runtime_common import run_command, write_prompt_file
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


def git_diff(workspace_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout or ""


def git_staged_diff(workspace_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--cached", "--no-ext-diff", "--binary"],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout or ""


def git_tracked_diff(workspace_path: Path) -> str:
    return git_staged_diff(workspace_path) + git_diff(workspace_path)


def git_untracked_files(workspace_path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        message = f"git status failed while checking setup contamination in {workspace_path}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)
    files: list[str] = []
    for line in (result.stdout or "").splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        if path:
            files.append(path)
    return sorted(set(files))


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
) -> TaskRecord | None:
    del adapter_record_suffix
    diff_text = git_tracked_diff(workspace_path)
    untracked_files = git_untracked_files(workspace_path)
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
    if agent != "codex":
        return
    shutil.rmtree(codex_runtime_root(task_dir), ignore_errors=True)


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
    runtime_env: dict[str, object] | None = None,
    runtime_setup_commands: Sequence[object] = (),
    runtime_keep_failed: bool = False,
) -> TaskRecord:
    try:
        adapter = get_coding_agent_adapter(agent)
    except ValueError as exc:
        raise usage_error(str(exc)) from exc
    agent = adapter.name
    if runtime_backend is None:
        raise usage_error("runtime_backend is required for coding-agent task execution.")
    if agent == "claude" and runtime_backend == "docker":
        raise usage_error("Claude Docker runtime is not supported; use runtime_backend='host'.")
    if runtime_backend == "docker" and runtime_image is None:
        runtime_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(agent)
        if runtime_image is None:
            raise usage_error(f"No pinned Docker runtime image is configured for agent {agent!r}")
    runtime_config = normalize_runtime_backend_config(
        runtime_backend=runtime_backend,
        runtime_image=runtime_image,
        runtime_env=runtime_env,
        runtime_setup_commands=runtime_setup_commands,
        runtime_keep_failed=runtime_keep_failed,
    )

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
    if agent == "codex":
        codex_runtime_dir = codex_runtime_root(task_dir)
        shutil.rmtree(codex_runtime_dir, ignore_errors=True)
        ensure_dir(codex_runtime_dir)
        extra_runtime_mounts.append(codex_runtime_dir)

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
    )
    runtime_success = False
    runtime_closed = False
    runtime_metadata: dict[str, object] = {"backend": runtime_config.backend}

    def write_record(record: TaskRecord) -> TaskRecord:
        record["runtime"] = runtime_metadata
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
        prepared_runtime = adapter.prepare_runtime(
            task_dir=task_dir,
            setup=setup_dict,
            env_overrides=env_overrides,
            runtime_backend=runtime_config.backend,
        )
        prepared_runtime = replace(prepared_runtime, execution_backend=task_runtime)

        runtime_setup_failure = run_runtime_setup_commands(
            task_runtime,
            commands=runtime_config.setup_commands,
            workspace_path=workspace_path,
            task_dir=task_dir,
            timeout=timeout,
            env=prepared_runtime.env,
        )
        if runtime_setup_failure is not None:
            completed_at = time.time()
            if runtime_setup_failure.command_result["timeout"] and runtime_config.backend == "docker":
                task_runtime.close(success=False)
                runtime_closed = True
            diff_text = git_tracked_diff(workspace_path)
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
                started_at=completed_at,
                completed_at=completed_at,
                setup_run=None,
            )
            return write_record(record)
        if runtime_config.setup_commands:
            contamination_record = build_setup_contamination_record(
                task=task,
                agent=agent,
                workspace_path=workspace_path,
                task_dir=task_dir,
                prompt_path=prompt_path,
                adapter_record_suffix=adapter.record_suffix,
                started_at=time.time(),
            )
            if contamination_record is not None:
                return write_record(contamination_record)

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
            )
            setup_run = build_setup_run_record(
                prompt_path=setup_result.prompt_path,
                stderr_path=setup_result.stderr_path,
                command_result=setup_result.command_result,
                raw_response_path=setup_result.raw_response_path,
                token_usage=setup_result.token_usage,
                tool_calls=setup_result.tool_calls,
                started_at=setup_result.started_at,
                completed_at=setup_result.completed_at,
            )
            if not setup_result.command_result["ok"]:
                if setup_result.command_result["timeout"] and runtime_config.backend == "docker":
                    task_runtime.close(success=False)
                    runtime_closed = True
                diff_text = git_tracked_diff(workspace_path)
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
        )

        if main_result.command_result["timeout"] and runtime_config.backend == "docker":
            task_runtime.close(success=False)
            runtime_closed = True
        diff_text = git_tracked_diff(workspace_path)
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

        runtime_success = str(record.get("status") or "") == "completed" and not record.get("timeout")
        return write_record(record)
    finally:
        if not runtime_closed:
            task_runtime.close(success=runtime_success)
        scrub_runtime_secrets(agent=agent, task_dir=task_dir)
