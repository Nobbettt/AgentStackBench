# Fork note: Modified by Norbert Laszlo on 2026-04-16 from upstream ContextBench.
# Summary of changes: keep conservative trace guards and avoid treating broad file lists as retrieved context.

"""Heuristics for inferring ContextBench trajectory data from raw agent traces."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from .inference_limits import (
    MAX_COMMAND_TOKENIZATION_CHARS as _MAX_COMMAND_TOKENIZATION_CHARS,
    MAX_COMMAND_OUTPUT_CHARS as _MAX_COMMAND_OUTPUT_CHARS,
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
_PATH_KEYS = {
    "file",
    "filepath",
    "file_path",
    "filename",
    "path",
    "relative_path",
    "uri",
}
_LINE_KEYS = {"line", "line_number", "lineno", "start", "start_line"}
_END_LINE_KEYS = {"end", "end_line", "line_end"}
_TEXT_KEYS = ("content", "text", "result", "output", "stdout", "stderr", "message")
_PATH_WITH_LINE_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::.*)?$")


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


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _strip_path_decoration(value: str) -> tuple[str, int | None]:
    candidate = value.strip().strip("'\"`")
    if candidate.startswith("file://"):
        candidate = candidate[len("file://") :]
    line_no = None
    match = _PATH_WITH_LINE_RE.match(candidate)
    if match:
        maybe_path = match.group("path").strip()
        if _looks_like_path_head(maybe_path):
            candidate = maybe_path
            line_no = int(match.group("line"))
    if "#L" in candidate:
        path_part, _, line_part = candidate.partition("#L")
        parsed_line = _coerce_positive_int(line_part.split("-", 1)[0])
        if parsed_line is not None:
            candidate = path_part
            line_no = parsed_line
    return candidate, line_no


def _normalize_inferred_file_path(value: object, workspace_path: Path) -> tuple[str | None, int | None]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None, None
    candidate, line_no = _strip_path_decoration(raw_value)
    normalized = normalize_workspace_path(candidate, workspace_path).strip()
    if not normalized or normalized.startswith("/") or normalized in {".", ".."}:
        return None, None
    if normalized.startswith("../") or "/../" in normalized:
        return None, None
    if not _looks_like_plain_path(normalized) and not _looks_like_path_head(normalized):
        return None, None
    return normalized, line_no


def _merge_span(target: SpanMap, file_path: str, start: int, end: int | None = None) -> None:
    end_value = max(start, end or start)
    target.setdefault(file_path, []).append({"start": start, "end": end_value})


def tool_result_text_from_value(value: object, *, depth: int = 0) -> str:
    """Extract textual content from common tool-result shapes without using tool names."""

    if value is None or depth > 8:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [tool_result_text_from_value(item, depth=depth + 1).strip() for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in _TEXT_KEYS:
            if key in value:
                text = tool_result_text_from_value(value.get(key), depth=depth + 1).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _collect_json_path_refs(
    value: object,
    *,
    workspace_path: Path,
    files: set[str],
    spans: SpanMap,
    depth: int = 0,
) -> None:
    if value is None or depth > 8:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return
            _collect_json_path_refs(parsed, workspace_path=workspace_path, files=files, spans=spans, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value[:500]:
            _collect_json_path_refs(item, workspace_path=workspace_path, files=files, spans=spans, depth=depth + 1)
        return
    if not isinstance(value, dict):
        return

    path_value = None
    for key, item in value.items():
        if str(key).strip().lower() in _PATH_KEYS and isinstance(item, str):
            path_value = item
            break
    file_path = None
    line_from_path = None
    if path_value is not None:
        file_path, line_from_path = _normalize_inferred_file_path(path_value, workspace_path)
        if file_path:
            files.add(file_path)
            start = line_from_path
            end = None
            for key, item in value.items():
                key_text = str(key).strip().lower()
                if key_text in _LINE_KEYS and start is None:
                    start = _coerce_positive_int(item)
                elif key_text in _END_LINE_KEYS:
                    end = _coerce_positive_int(item)
            if start is not None:
                _merge_span(spans, file_path, start, end)

    for item in value.values():
        _collect_json_path_refs(item, workspace_path=workspace_path, files=files, spans=spans, depth=depth + 1)


def infer_retrieval_step_from_tool_result(
    result_value: object,
    *,
    output_text: str | None = None,
    workspace_path: Path,
    meta: TraceInferenceMeta | None = None,
) -> RetrievalStep | None:
    """Infer retrieval context from generic successful tool-result content."""

    text = output_text if output_text is not None else tool_result_text_from_value(result_value)
    if len(text) > _MAX_COMMAND_OUTPUT_CHARS:
        if meta is not None:
            meta["dropped_large_command_outputs"] = int(meta.get("dropped_large_command_outputs", 0) or 0) + 1
        text = ""

    raw_spans = infer_grep_spans_from_text(text, workspace_path, meta=meta) if text else {}
    spans: SpanMap = {}
    for file_path, file_spans in raw_spans.items():
        normalized, _ = _normalize_inferred_file_path(file_path, workspace_path)
        if not normalized:
            continue
        spans.setdefault(normalized, []).extend(file_spans)
    files = set(spans)
    if text:
        for file_path in infer_file_list_from_text(text, workspace_path, meta=meta):
            normalized, _ = _normalize_inferred_file_path(file_path, workspace_path)
            if normalized:
                files.add(normalized)

    json_files: set[str] = set()
    json_spans: SpanMap = {}
    _collect_json_path_refs(result_value, workspace_path=workspace_path, files=json_files, spans=json_spans)
    files.update(json_files)
    spans = merge_span_maps(spans, json_spans)
    if not files and not spans:
        return None
    files.update(spans)
    return {"files": sorted(files), "spans": spans, "symbols": {}}


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
        if spans:
            return {"files": sorted(spans), "spans": spans, "symbols": {}}
        infer_file_list_from_text(output_text, workspace_path, meta=meta)
        return None

    if "find" in tokens or _command_has_word(raw_command, "find"):
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
