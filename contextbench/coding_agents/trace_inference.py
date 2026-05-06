# Fork note: Modified by Norbert Laszlo on 2026-04-16 from upstream ContextBench.
# Summary of changes: restore safe repo-root file inference while keeping conservative trace guards.

"""Heuristics for inferring ContextBench trajectory data from raw agent traces."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .inference_limits import (
    MAX_COMMAND_TOKENIZATION_CHARS as _MAX_COMMAND_TOKENIZATION_CHARS,
    MAX_FILE_LIST_MATCHES as _MAX_FILE_LIST_MATCHES,
    MAX_GREP_LINE_CHARS as _MAX_GREP_LINE_CHARS,
    MAX_GREP_SPAN_MATCHES as _MAX_GREP_SPAN_MATCHES,
    MAX_PLAIN_PATH_LINE_CHARS as _MAX_PLAIN_PATH_LINE_CHARS,
)
from .records import merge_span_maps
from .types import RetrievalStep, SpanMap, SymbolMap, TraceInferenceMeta, TrajectoryData

_LINE_ARROW_RE = re.compile(r"^\s*(?P<line>\d+)\s*→", re.MULTILINE)
_SED_RANGE_RE = re.compile(r"(?P<start>\d+),(?P<end>\d+)p")
_KNOWN_ROOT_FILENAMES = {
    "BUILD",
    "BUILD.bazel",
    "Brewfile",
    "CMakeLists.txt",
    "Dockerfile",
    "Gemfile",
    "Jenkinsfile",
    "LICENSE",
    "Makefile",
    "NOTICE",
    "Procfile",
    "README",
    "Rakefile",
    "Vagrantfile",
    "WORKSPACE",
    "WORKSPACE.bazel",
}
_KNOWN_ROOT_PREFIXES = ("Dockerfile.", "README.")


def normalize_workspace_path(path_value: str, workspace_path: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(workspace_path.resolve()).as_posix()
        except Exception:
            try:
                path_str = str(path.resolve())
            except Exception:
                path_str = str(path)
            try:
                workspace_str = str(workspace_path.resolve())
            except Exception:
                workspace_str = str(workspace_path)
            if path_str.startswith(workspace_str.rstrip("/") + "/"):
                return path_str[len(workspace_str.rstrip("/") + "/") :]
            return path_value
    return path.as_posix()


def infer_read_span_from_text(text: str) -> tuple[int, int] | None:
    matches = [int(match.group("line")) for match in _LINE_ARROW_RE.finditer(text)]
    if not matches:
        return None
    return min(matches), max(matches)


def infer_grep_spans_from_text(
    text: str,
    workspace_path: Path,
    *,
    meta: TraceInferenceMeta | None = None,
) -> SpanMap:
    spans: SpanMap = {}
    count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > _MAX_GREP_LINE_CHARS:
            if meta is not None and raw_line.strip() and len(line) > _MAX_GREP_LINE_CHARS:
                meta["dropped_large_grep_lines"] = int(meta.get("dropped_large_grep_lines", 0) or 0) + 1
            continue
        path_head, line_no = _parse_grep_line(line)
        if not path_head or line_no is None:
            continue
        file_path = normalize_workspace_path(path_head, workspace_path)
        spans.setdefault(file_path, []).append({"start": line_no, "end": line_no})
        count += 1
        if count >= _MAX_GREP_SPAN_MATCHES:
            if meta is not None:
                meta["grep_match_cap_hits"] = int(meta.get("grep_match_cap_hits", 0) or 0) + 1
            break
    return spans


def _parse_grep_line(line: str) -> tuple[str | None, int | None]:
    if ":" not in line:
        return None, None
    path_head, _, remainder = line.partition(":")
    if not path_head or not _looks_like_path_head(path_head):
        return None, None
    line_part, _, _ = remainder.partition(":")
    line_part = line_part.strip()
    if not line_part.isdigit():
        return None, None
    return path_head, int(line_part)


def _looks_like_repo_filename(name: str) -> bool:
    if not name:
        return False
    if name in _KNOWN_ROOT_FILENAMES or any(name.startswith(prefix) for prefix in _KNOWN_ROOT_PREFIXES):
        return True
    if "." not in name:
        return False
    ext = name.rsplit(".", 1)[-1]
    cleaned_ext = ext.replace("_", "").replace("-", "").replace("+", "")
    if not cleaned_ext.isalnum():
        return False
    return any(ch.isalpha() for ch in name)


def _looks_like_path_head(value: str) -> bool:
    if not value or "=" in value:
        return False
    name = value.rsplit("/", 1)[-1]
    return _looks_like_repo_filename(name)


def infer_file_list_from_text(
    text: str,
    workspace_path: Path,
    *,
    meta: TraceInferenceMeta | None = None,
) -> list[str]:
    files: list[str] = []
    count = 0
    for line in text.splitlines():
        raw = line.strip()
        if not _looks_like_plain_path(raw, meta=meta):
            continue
        files.append(normalize_workspace_path(raw, workspace_path))
        count += 1
        if count >= _MAX_FILE_LIST_MATCHES:
            if meta is not None:
                meta["file_list_cap_hits"] = int(meta.get("file_list_cap_hits", 0) or 0) + 1
            break
    return sorted(set(files))


def _looks_like_plain_path(value: str, *, meta: TraceInferenceMeta | None = None) -> bool:
    if not value or len(value) > _MAX_PLAIN_PATH_LINE_CHARS:
        return False
    if any(ch.isspace() for ch in value):
        return False
    # Avoid environment-variable and key=value outputs such as PATH=...
    if "=" in value:
        if meta is not None:
            meta["dropped_env_var_lines"] = int(meta.get("dropped_env_var_lines", 0) or 0) + 1
        return False
    name = value.rsplit("/", 1)[-1]
    return _looks_like_repo_filename(name)


def unwrap_shell_command(command: str) -> str:
    if len(command) > _MAX_COMMAND_TOKENIZATION_CHARS:
        return command
    try:
        outer = shlex.split(command)
    except Exception:
        return command
    if len(outer) >= 3 and outer[1] == "-lc":
        return outer[2]
    return command


def command_tokens(command: str) -> list[str]:
    if len(command) > _MAX_COMMAND_TOKENIZATION_CHARS:
        return []
    try:
        return shlex.split(unwrap_shell_command(command))
    except Exception:
        return []


def _command_has_word(command: str, word: str) -> bool:
    pattern = rf"(^|[^A-Za-z0-9_./-]){re.escape(word)}([^A-Za-z0-9_./-]|$)"
    return re.search(pattern, command) is not None


def _find_path_like_token(command: str) -> str | None:
    path_pattern = re.compile(r"(?P<path>(?:/|\.{0,2}/)?[A-Za-z0-9_.@%+=~:/-]+\.[A-Za-z0-9_+-]+)")
    for match in path_pattern.finditer(command):
        candidate = match.group("path").strip("'\"")
        if "=" in candidate:
            continue
        if candidate.endswith(".exe"):
            continue
        return candidate
    return None


def _read_like_step(tokens: list[str], output_text: str, workspace_path: Path) -> RetrievalStep | None:
    path_token = None
    for token in tokens:
        if token in {"|", "&&", "||"}:
            continue
        if token.startswith("-"):
            continue
        if "." not in token and "/" not in token:
            continue
        if token.endswith("p") and "," in token:
            continue
        path_token = token
    if not path_token:
        return None
    file_path = normalize_workspace_path(path_token, workspace_path)
    span = infer_read_span_from_text(output_text)
    spans: SpanMap = {file_path: [{"start": span[0], "end": span[1]}]} if span else {}
    return {"files": [file_path], "spans": spans, "symbols": {}}


def infer_retrieval_step_from_command(
    command: str,
    *,
    output_text: str,
    workspace_path: Path,
    meta: TraceInferenceMeta | None = None,
) -> RetrievalStep | None:
    raw_command = unwrap_shell_command(command)
    tokens = command_tokens(command)

    if "Read" in tokens or _command_has_word(raw_command, "Read"):
        return None

    if "rg" in tokens or "grep" in tokens or _command_has_word(raw_command, "rg") or _command_has_word(raw_command, "grep"):
        spans = infer_grep_spans_from_text(output_text, workspace_path, meta=meta)
        files = sorted(spans) or infer_file_list_from_text(output_text, workspace_path, meta=meta)
        if files or spans:
            return {"files": files, "spans": spans, "symbols": {}}
        return None

    if "find" in tokens or _command_has_word(raw_command, "find"):
        files = infer_file_list_from_text(output_text, workspace_path, meta=meta)
        if files:
            return {"files": files, "spans": {}, "symbols": {}}
        return None

    if any(token in {"sed", "cat", "head", "tail", "nl"} for token in tokens) or any(
        _command_has_word(raw_command, word) for word in ("sed", "cat", "head", "tail", "nl")
    ):
        if tokens:
            return _read_like_step(tokens, output_text, workspace_path)
        path_token = _find_path_like_token(raw_command)
        if not path_token:
            return None
        file_path = normalize_workspace_path(path_token, workspace_path)
        span = infer_read_span_from_text(output_text)
        spans: SpanMap = {file_path: [{"start": span[0], "end": span[1]}]} if span else {}
        return {"files": [file_path], "spans": spans, "symbols": {}}

    return None


def infer_read_step(file_path: str, *, output_text: str, workspace_path: Path) -> RetrievalStep:
    normalized = normalize_workspace_path(file_path, workspace_path)
    span = infer_read_span_from_text(output_text)
    spans: SpanMap = {normalized: [{"start": span[0], "end": span[1]}]} if span else {}
    return {"files": [normalized], "spans": spans, "symbols": {}}


def merge_retrieval_steps(*step_lists: list[RetrievalStep]) -> list[RetrievalStep]:
    merged: list[RetrievalStep] = []
    by_key: dict[tuple[str, str], RetrievalStep] = {}
    for steps in step_lists:
        for step in steps:
            key = (
                ",".join(step.get("files", [])),
                repr(step.get("spans", {})),
            )
            existing = by_key.get(key)
            if existing is not None:
                for file_path, names in step.get("symbols", {}).items():
                    bucket = existing.setdefault("symbols", {}).setdefault(file_path, [])
                    for name in names:
                        if name not in bucket:
                            bucket.append(name)
                continue
            copied: RetrievalStep = {
                "files": list(step.get("files", [])),
                "spans": {
                    file_path: [dict(span) for span in spans]
                    for file_path, spans in step.get("spans", {}).items()
                },
                "symbols": {
                    file_path: list(names)
                    for file_path, names in step.get("symbols", {}).items()
                },
            }
            by_key[key] = copied
            merged.append(copied)
    return merged


def trajectory_from_steps(steps: list[RetrievalStep]) -> TrajectoryData | None:
    grounded_files = {
        file_path
        for step in steps
        if step.get("spans") or step.get("symbols")
        for file_path in step.get("files", [])
    }
    all_step_files = {file_path for step in steps for file_path in step.get("files", [])}
    files = sorted(grounded_files or all_step_files)
    spans = merge_span_maps(*(step.get("spans") for step in steps))
    symbols: SymbolMap = {}
    for step in steps:
        for file_path, names in step.get("symbols", {}).items():
            symbols.setdefault(file_path, []).extend(names)
    symbols = {file_path: sorted(set(names)) for file_path, names in symbols.items() if names}
    if not steps and not files and not spans and not symbols:
        return None
    return {
        "pred_steps": steps,
        "pred_files": files,
        "pred_spans": spans,
        "pred_symbols": symbols,
    }
