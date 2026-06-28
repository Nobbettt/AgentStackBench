# SPDX-License-Identifier: Apache-2.0

"""Codex OTEL v2 runtime preparation and invocation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from ...coding_agents.files import ensure_dir, read_json_or_text, write_json, usage_error
from ...coding_agents.response_parsing import structured_output_schema_error
from ...coding_agents.runtime_backends import BaseTaskRuntime, RuntimeSetupResult
from ...coding_agents.runtime_common import (
    run_command,
    write_prompt_file,
)
from ...coding_agents.types import CommandResult
from ..adapter_base import CodingAgentInvocationResult
from ..codex.runtime import codex_tool_bundle_root, validate_auth_file
from ..otel_common import (
    OtelScoredRunPolicy,
    append_diagnostic_note,
    collector_endpoint_hosts,
    force_command_failure,
    one_attempt_retry_metadata,
)
from .otel import OtlpJsonCaptureServer
from .parser import CodexOtelV2AgentParser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTAINER_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SCORED_RUN_POLICY = OtelScoredRunPolicy(
    missing_successful_tool_result_note=(
        "Codex OTEL v2 completed without successful OTEL tool-result telemetry; "
        "refusing empty-context fallback output."
    ),
    missing_evaluable_context_note=(
        "Codex OTEL v2 completed without OTEL-derived evaluable context; "
        "refusing empty-context fallback output."
    ),
)


def runtime_root(task_dir: Path) -> Path:
    digest = hashlib.sha1(str(task_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    return _REPO_ROOT / ".cache" / "agent-runtimes" / "codex-otel-v2" / digest


def runtime_roots(task_dir: Path) -> dict[str, Path]:
    root = runtime_root(task_dir)
    home_dir = root / "home"
    return {
        "task_dir": task_dir,
        "runtime_root": root,
        "home_dir": home_dir,
        "codex_home": home_dir / ".codex",
        "xdg_config_home": root / "xdg-config",
        "xdg_data_home": root / "xdg-data",
        "xdg_cache_home": root / "xdg-cache",
    }


def apply_runtime_setup_files(
    task_dir: Path,
    *,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> None:
    from ...coding_agents.runtime_common import apply_copy_paths, apply_materialized_files

    apply_copy_paths(copy_paths, roots=runtime_roots(task_dir))
    apply_materialized_files(materialized_files, roots=runtime_roots(task_dir))


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
    return command, "codex-stdout.txt"


def prepare_runtime_env(
    task_dir: Path,
    source_codex_dir: Path | None = None,
    *,
    include_host_env: bool = True,
    runtime_env: Mapping[str, str] | None = None,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> dict[str, str]:
    source_root = source_codex_dir or (Path.home() / ".codex")
    auth_path = source_root / "auth.json"
    if not auth_path.exists():
        raise usage_error(f"Codex auth is unavailable: expected {auth_path}")
    validate_auth_file(auth_path)

    roots = runtime_roots(task_dir)
    home_dir = roots["home_dir"]
    codex_home = roots["codex_home"]
    xdg_config_home = roots["xdg_config_home"]
    xdg_data_home = roots["xdg_data_home"]
    xdg_cache_home = roots["xdg_cache_home"]
    local_bin = home_dir / ".local" / "bin"
    runtime_bin = roots["runtime_root"] / "bin"
    npm_global = roots["runtime_root"] / "npm-global"
    npm_global_bin = npm_global / "bin"
    tool_bundle = codex_tool_bundle_root(runtime_env=runtime_env)

    for path in (codex_home, xdg_config_home, xdg_data_home, xdg_cache_home, local_bin, runtime_bin, npm_global_bin):
        ensure_dir(path)

    shutil.copy2(auth_path, codex_home / "auth.json")
    apply_runtime_setup_files(task_dir, materialized_files=materialized_files, copy_paths=copy_paths)
    env = os.environ.copy() if include_host_env else {}
    if not include_host_env:
        path_entries = [
            str(local_bin),
            str(runtime_bin),
            str(npm_global_bin),
        ]
        if tool_bundle is not None:
            path_entries.append(str(tool_bundle / "usr-local" / "bin"))
        path_entries.append(_CONTAINER_DEFAULT_PATH)
        env["PATH"] = ":".join(path_entries)
    env.update(
        {
            "HOME": str(home_dir),
            "CODEX_HOME": str(codex_home),
            "CONTEXTBENCH_RUNTIME_ROOT": str(roots["runtime_root"]),
            "CONTEXTBENCH_RUNTIME_BIN": str(runtime_bin),
            "NPM_CONFIG_PREFIX": str(npm_global),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_DATA_HOME": str(xdg_data_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
        }
    )
    env.pop("OTEL_SDK_DISABLED", None)
    return env


def validate_cli_in_runtime(
    *,
    runtime: BaseTaskRuntime,
    task_dir: Path,
    workspace_path: Path,
    timeout: int,
    env: dict[str, str] | None,
) -> RuntimeSetupResult | None:
    command = ["codex", "--version"]
    stdout_path = task_dir / "codex-version.stdout.log"
    stderr_path = task_dir / "codex-version.stderr.log"
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
    if result["ok"]:
        return None
    return RuntimeSetupResult(
        command_result=result,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        command=" ".join(command),
        started_at=started_at,
        completed_at=completed_at,
    )


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    return reasoning_effort


def _toml_string(value: str) -> str:
    return json.dumps(value)


def write_otel_config(*, codex_home: Path, logs_endpoint: str, traces_endpoint: str) -> Path:
    config_path = codex_home / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[otel]",
                'environment = "contextbench"',
                "log_user_prompt = false",
                (
                    "exporter = { otlp-http = { "
                    f"endpoint = {_toml_string(logs_endpoint)}, protocol = \"json\""
                    " } }"
                ),
                (
                    "trace_exporter = { otlp-http = { "
                    f"endpoint = {_toml_string(traces_endpoint)}, protocol = \"json\""
                    " } }"
                ),
                'metrics_exporter = "none"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def build_codex_otel_raw_response(
    *,
    agent_name: str,
    otel_capture: dict[str, object],
    final_output_path: Path | None,
    config_path: Path,
) -> dict[str, object]:
    raw_response: dict[str, object] = {
        "agent": agent_name,
        "response_format": "otlp-json",
        "otel": otel_capture,
        "otel_config_path": str(config_path),
    }
    if final_output_path and final_output_path.exists():
        raw_response["final_message"] = read_json_or_text(final_output_path)
    return raw_response


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
    retry_dirty_check: Callable[[], bool] | None = None,
) -> CodingAgentInvocationResult:
    parser = CodexOtelV2AgentParser()
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
    otel_artifact_path = task_dir / raw_output_filename.replace("stdout", "otel-capture").replace(".txt", ".json")
    started_at = time.time()
    raw_response: dict[str, object] = {"agent": "codex-otel-v2", "response_format": "otlp-json", "otel": {}}
    command_result: CommandResult = {"ok": False, "exit_code": None, "signal": None, "timeout": False}
    completed_at = started_at
    retry_events: list[dict[str, object]] = []
    del retry_dirty_check
    for path in (raw_output_path, stderr_path, raw_response_path, final_output_path, otel_artifact_path):
        if path is not None and path.exists():
            path.unlink()
    bind_host, endpoint_host = collector_endpoint_hosts(execution_backend)
    collector = OtlpJsonCaptureServer(bind_host=bind_host)
    collector.start()
    config_path = write_otel_config(
        codex_home=Path((env or {}).get("CODEX_HOME") or Path((env or {}).get("HOME") or "").joinpath(".codex")),
        logs_endpoint=f"http://{endpoint_host}:{collector.port}/v1/logs",
        traces_endpoint=f"http://{endpoint_host}:{collector.port}/v1/traces",
    )
    try:
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
    finally:
        completed_at = time.time()
        collector.stop()
    otel_capture = collector.export()
    write_json(otel_artifact_path, otel_capture)
    raw_response = build_codex_otel_raw_response(
        agent_name="codex-otel-v2",
        otel_capture=otel_capture,
        final_output_path=final_output_path,
        config_path=config_path,
    )
    write_json(raw_response_path, raw_response)
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    schema_validation_note = (
        structured_output_schema_error(structured_output, schema_path)
        if structured_output is not None and schema_path is not None
        else None
    )
    if schema_validation_note:
        structured_output = None
    diagnostic_note = None
    if schema_validation_note:
        diagnostic_note = append_diagnostic_note(diagnostic_note, schema_validation_note)
    otel = raw_response.get("otel")
    if isinstance(otel, dict) and not otel.get("request_count"):
        command_result = force_command_failure(command_result)
        note = "Codex OTEL v2 completed without receiving OTLP payloads; refusing non-OTEL fallback output."
        diagnostic_note = append_diagnostic_note(diagnostic_note, note)
    validation = _SCORED_RUN_POLICY.validate(
        parser=parser,
        raw_response=raw_response,
        command_result=command_result,
        diagnostic_note=diagnostic_note,
        workspace_path=workspace_path,
        scored=schema_path is not None,
    )
    command_result = validation.command_result
    diagnostic_note = validation.diagnostic_note
    return CodingAgentInvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        command_executions=parser.extract_command_executions(raw_response),
        available_tools=parser.extract_available_tools(raw_response),
        persisted_tool_results=[],
        diagnostic_note=diagnostic_note,
        retry=one_attempt_retry_metadata(events=retry_events),
        started_at=started_at,
        completed_at=completed_at,
    )
