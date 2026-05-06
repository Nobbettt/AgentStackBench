
"""Shared runtime utilities used by coding-agent adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .files import ensure_dir, usage_error, write_json
from .types import CommandResult


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdin_text: str | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> CommandResult:
    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        stdout_path.write_text(coerce_output_text(result.stdout), encoding="utf-8")
        stderr_path.write_text(coerce_output_text(result.stderr), encoding="utf-8")
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "signal": None,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(coerce_output_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(coerce_output_text(exc.stderr), encoding="utf-8")
        return {
            "ok": False,
            "exit_code": None,
            "signal": "SIGTERM",
            "timeout": True,
        }


def merge_json_objects(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = merge_json_objects(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def coerce_output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def resolve_runtime_path(roots: dict[str, Path], *, target_root: str, relative_path: str) -> Path:
    base = roots.get(target_root)
    if base is None:
        raise usage_error(f"Runtime target root '{target_root}' is not available for this agent")
    if Path(relative_path).is_absolute():
        raise usage_error(f"Runtime path must be relative: {relative_path}")
    target = (base / relative_path).resolve()
    base_real = base.resolve()
    try:
        if os.path.commonpath([str(base_real), str(target)]) != str(base_real):
            raise usage_error(f"Runtime path escapes target root '{target_root}': {relative_path}")
    except ValueError as exc:
        raise usage_error(f"Invalid runtime path for target root '{target_root}': {relative_path}") from exc
    return target


def apply_materialized_files(
    specs: Sequence[dict[str, object]] | None,
    *,
    roots: dict[str, Path],
) -> None:
    for spec in specs or ():
        path_value = str(spec.get("path") or "").strip()
        if not path_value:
            raise usage_error("Materialized runtime files require a non-empty 'path'")
        target_root = str(spec.get("target_root") or "task_dir")
        format_name = str(spec.get("format") or "text").strip().lower()
        target_path = resolve_runtime_path(roots, target_root=target_root, relative_path=path_value)
        ensure_dir(target_path.parent)
        if format_name == "json":
            write_json(target_path, spec.get("content"))
            continue
        if format_name != "text":
            raise usage_error(f"Unsupported materialized file format: {format_name}")
        content = spec.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, indent=2, ensure_ascii=False)
        target_path.write_text(text, encoding="utf-8")


def apply_copy_paths(
    specs: Sequence[dict[str, object]] | None,
    *,
    roots: dict[str, Path],
) -> None:
    for spec in specs or ():
        source_path = Path(str(spec.get("source") or "")).expanduser()
        if not source_path.exists():
            raise usage_error(f"Runtime copy source does not exist: {source_path}")
        destination = str(spec.get("destination") or "").strip()
        target_root = str(spec.get("target_root") or "task_dir")
        if source_path.is_file() and not destination:
            destination = source_path.name
        target_path = resolve_runtime_path(roots, target_root=target_root, relative_path=destination or ".")
        if source_path.is_dir():
            ensure_dir(target_path.parent)
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            continue
        if destination.endswith("/") or destination in ("", "."):
            target_path = resolve_runtime_path(
                roots,
                target_root=target_root,
                relative_path=str((Path(destination or ".") / source_path.name).as_posix()),
            )
        ensure_dir(target_path.parent)
        shutil.copy2(source_path, target_path)


def write_prompt_file(task_dir: Path, filename: str, prompt: str) -> Path:
    path = task_dir / filename
    path.write_text(prompt, encoding="utf-8")
    return path


def attempt_path(path: Path, attempt_index: int) -> Path:
    return path.with_name(f"{path.stem}.attempt{attempt_index}{path.suffix}")


def archive_retry_artifacts(paths: Sequence[Path | None], *, attempt_index: int) -> None:
    for path in paths:
        if path is None or not path.exists():
            continue
        shutil.copy2(path, attempt_path(path, attempt_index))
