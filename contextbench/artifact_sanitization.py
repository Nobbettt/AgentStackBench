
"""Sanitizers for artifacts that may be published outside the local machine."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_BOUNDARY = r"(?=$|[\s'\"`),\]}<>])"
_WORKTREE_PATTERN = re.compile(
    r"(?P<path>(?:/[^\s'\"`),\]}<>]+)?/(?:\.cache/worktrees/)?contextbench_worktrees/"
    r"(?P<slug>[^\s'\"`),\]}<>]+))"
)
_HOME_PREFIX_PATTERN = re.compile(r"(?<!\w)(?:/Users|/home)/[^/\s'\"`),\]}<>]+")
_TMP_PREFIX_PATTERN = re.compile(r"(?<!\w)(?:/var/folders|/private/var/folders|/tmp)/[^\s'\"`),\]}<>]+")
_FORBIDDEN_PATTERNS = (
    re.compile(r"(?<!\w)(?:/Users|/home)/[^/\s'\"`),\]}<>]+"),
    re.compile(r"\.cache/worktrees/contextbench_worktrees"),
    re.compile(r"(?<!\w)(?:/var/folders|/private/var/folders|/tmp)/[^\s'\"`),\]}<>]+"),
)
_JSON_SUFFIXES = {".json"}
_JSONL_SUFFIXES = {".jsonl"}
_TEXT_SUFFIXES = {
    ".csv",
    ".diff",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rst",
    ".txt",
}
_EXCLUDED_PUBLIC_ARTIFACT_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "codex-runtime",
    "image_build_dir",
    "instance_image_build_dir",
    "node_modules",
    "public-artifacts",
    "repos",
}


@dataclass(frozen=True)
class SanitizationContext:
    """Path roots that should be replaced in public artifacts."""

    repo_root: Path = _REPO_ROOT
    suite_dir: Path | None = None
    workspace_path: Path | None = None
    task_dir: Path | None = None
    extra_roots: tuple[Path, ...] = field(default_factory=tuple)


def _as_posix(path: Path | str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").rstrip("/")
    except Exception:
        return str(path).replace("\\", "/").rstrip("/")


def _replace_root(text: str, root: Path | str | None, replacement: str) -> str:
    root_text = _as_posix(root)
    if not root_text:
        return text
    return text.replace(root_text, replacement)


def sanitize_text(text: object, *, context: SanitizationContext | None = None) -> str:
    """Redact local absolute paths from a string while preserving useful context."""

    sanitized = str(text)
    ctx = context or SanitizationContext()

    def worktree_repl(match: re.Match[str]) -> str:
        path = match.group("path")
        marker = "/.cache/worktrees/contextbench_worktrees/"
        _, found, suffix = path.partition(marker)
        if not found:
            _, _, suffix = path.partition("/contextbench_worktrees/")
        return f"<worktree>/{suffix}" if suffix else "<worktree>"

    sanitized = _replace_root(sanitized, ctx.workspace_path, "<worktree>")
    sanitized = _WORKTREE_PATTERN.sub(worktree_repl, sanitized)
    sanitized = _replace_root(sanitized, ctx.task_dir, "<task-artifacts>")

    suite_dir = _as_posix(ctx.suite_dir)
    repo_root = _as_posix(ctx.repo_root)
    if suite_dir and repo_root and suite_dir.startswith(f"{repo_root}/"):
        suite_replacement = suite_dir[len(repo_root) + 1 :]
    else:
        suite_replacement = "<suite-artifacts>"
    sanitized = _replace_root(sanitized, ctx.suite_dir, suite_replacement)

    for index, root in enumerate(ctx.extra_roots, start=1):
        sanitized = _replace_root(sanitized, root, f"<artifact-root-{index}>")

    sanitized = _replace_root(sanitized, ctx.repo_root, "<contextbench-root>")
    sanitized = _HOME_PREFIX_PATTERN.sub("<home>", sanitized)
    sanitized = _TMP_PREFIX_PATTERN.sub("<tmp>", sanitized)
    return sanitized


def sanitize_json_value(value: Any, *, context: SanitizationContext | None = None) -> Any:
    """Recursively sanitize strings inside a JSON-compatible value."""

    if isinstance(value, str):
        return sanitize_text(value, context=context)
    if isinstance(value, list):
        return [sanitize_json_value(item, context=context) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item, context=context) for item in value]
    if isinstance(value, Mapping):
        return {
            sanitize_text(str(key), context=context): sanitize_json_value(item, context=context)
            for key, item in value.items()
        }
    return value


def path_for_public_artifact(path: str | Path, *, context: SanitizationContext | None = None) -> str:
    """Return a stable public reference for a path-like value."""

    ctx = context or SanitizationContext()
    raw = _as_posix(path)
    if not raw:
        return ""
    repo_root = _as_posix(ctx.repo_root)
    if repo_root and raw.startswith(f"{repo_root}/"):
        return raw[len(repo_root) + 1 :]
    return sanitize_text(raw, context=ctx)


def _stringify_for_scan(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def find_private_path_matches(value: Any) -> list[str]:
    """Return representative private/local path matches left in a value."""

    text = _stringify_for_scan(value)
    matches: list[str] = []
    seen: set[str] = set()
    for pattern in _FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            snippet = match.group(0)
            if snippet not in seen:
                seen.add(snippet)
                matches.append(snippet)
            if len(matches) >= 20:
                return matches
    return matches


def assert_no_private_paths(value: Any, *, label: str = "artifact") -> None:
    matches = find_private_path_matches(value)
    if matches:
        sample = ", ".join(matches[:5])
        raise ValueError(f"{label} contains private local paths: {sample}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_publishable_files(source_dir: Path):
    for root, dirnames, filenames in os.walk(source_dir):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in _EXCLUDED_PUBLIC_ARTIFACT_DIR_NAMES
        ]
        for filename in filenames:
            yield Path(root) / filename


def _sanitize_json_file(source: Path, target: Path, context: SanitizationContext) -> None:
    raw_text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        _sanitize_text_file(source, target, context)
        return
    sanitized = sanitize_json_value(payload, context=context)
    assert_no_private_paths(sanitized, label=str(source))
    target.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sanitize_jsonl_file(source: Path, target: Path, context: SanitizationContext) -> None:
    lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload: Any = json.loads(raw_line)
        except json.JSONDecodeError:
            sanitized_line = sanitize_text(raw_line, context=context)
            assert_no_private_paths(sanitized_line, label=str(source))
            lines.append(sanitized_line)
            continue
        sanitized = sanitize_json_value(payload, context=context)
        assert_no_private_paths(sanitized, label=str(source))
        lines.append(json.dumps(sanitized, ensure_ascii=False))
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sanitize_text_file(
    source: Path,
    target: Path,
    context: SanitizationContext,
    *,
    errors: str = "replace",
) -> None:
    sanitized = sanitize_text(source.read_text(encoding="utf-8", errors=errors), context=context)
    assert_no_private_paths(sanitized, label=str(source))
    target.write_text(sanitized, encoding="utf-8")


def sanitize_artifact_tree(
    *,
    source_dir: Path,
    output_dir: Path,
    repo_root: Path | None = None,
    overwrite: bool = True,
) -> None:
    """Write a sanitized publishable copy of a suite artifact tree."""

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Artifact source does not exist: {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    _ensure_dir(output_dir)

    context = SanitizationContext(
        repo_root=(repo_root or Path.cwd()).resolve(),
        suite_dir=source_dir,
        extra_roots=(source_dir,),
    )

    for source in _iter_publishable_files(source_dir):
        relative = source.relative_to(source_dir)
        target = output_dir / relative
        if source.is_symlink():
            _ensure_dir(target.parent)
            link_target = sanitize_text(str(source.readlink()), context=context)
            target.write_text(f"symlink_target: {link_target}\n", encoding="utf-8")
            continue
        _ensure_dir(target.parent)
        suffix = source.suffix.lower()
        try:
            if suffix in _JSON_SUFFIXES:
                _sanitize_json_file(source, target, context)
            elif suffix in _JSONL_SUFFIXES:
                _sanitize_jsonl_file(source, target, context)
            elif suffix in _TEXT_SUFFIXES:
                _sanitize_text_file(source, target, context)
            else:
                try:
                    _sanitize_text_file(source, target, context, errors="strict")
                except UnicodeDecodeError:
                    shutil.copy2(source, target)
        except UnicodeDecodeError:
            shutil.copy2(source, target)
        except FileNotFoundError:
            continue


def sanitize_artifact_tree_in_place(
    *,
    source_dir: Path,
    repo_root: Path | None = None,
) -> None:
    """Sanitize publishable text artifacts in the operational result tree."""

    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Artifact source does not exist: {source_dir}")

    context = SanitizationContext(
        repo_root=(repo_root or Path.cwd()).resolve(),
        suite_dir=source_dir,
        extra_roots=(source_dir,),
    )
    for source in _iter_publishable_files(source_dir):
        if source.is_symlink():
            continue
        suffix = source.suffix.lower()
        try:
            if suffix in _JSON_SUFFIXES:
                _sanitize_json_file(source, source, context)
            elif suffix in _JSONL_SUFFIXES:
                _sanitize_jsonl_file(source, source, context)
            elif suffix in _TEXT_SUFFIXES:
                _sanitize_text_file(source, source, context)
            else:
                try:
                    _sanitize_text_file(source, source, context, errors="strict")
                except UnicodeDecodeError:
                    continue
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            continue


def scan_paths_for_private_artifacts(paths: Sequence[Path]) -> dict[str, list[str]]:
    """Scan public artifacts and return files that still contain local paths."""

    findings: dict[str, list[str]] = {}
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files.append(path)

    for path in files:
        if path.suffix and path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        matches = find_private_path_matches(text)
        if matches:
            findings[str(path)] = matches
    return findings


def assert_paths_have_no_private_artifacts(paths: Sequence[Path]) -> None:
    findings = scan_paths_for_private_artifacts(paths)
    if not findings:
        return
    first_path, matches = next(iter(findings.items()))
    sample = ", ".join(matches[:5])
    raise ValueError(f"Public artifacts contain private local paths in {first_path}: {sample}")
