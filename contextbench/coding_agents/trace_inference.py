# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-08 from upstream ContextBench.
# Summary of changes: keep conservative trace guards, infer command-grounded read spans, and avoid scoring broad search hits as context.

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
_NUMBERED_LINE_RE = re.compile(r"^\s*(?P<line>\d+)(?:\t| {2,})\s*(?=\S)", re.MULTILINE)
_SED_RANGE_RE = re.compile(r"(?P<start>\d+)\s*,\s*(?P<end>\d+)\s*p")
_HEAD_COUNT_RE = re.compile(r"^(?P<count>\d+)$")
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
        matches = [int(match.group("line")) for match in _NUMBERED_LINE_RE.finditer(text)]
        if not _looks_like_line_numbering(matches):
            return None
    if not matches:
        return None
    return min(matches), max(matches)


def _looks_like_line_numbering(matches: list[int]) -> bool:
    """Accept numeric prefixes only when they form a contiguous line counter.

    Timestamped logs, data tables, and other numeric-prefixed output match
    _NUMBERED_LINE_RE too; requiring consecutive +1 increments keeps those
    from being scored as fabricated read spans.
    """
    if not matches:
        return False
    if len(matches) == 1:
        return matches[0] == 1
    return all(later - earlier == 1 for earlier, later in zip(matches, matches[1:]))


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

    files: set[str] = set()
    spans: SpanMap = {}

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


def _split_command_segments(command: str) -> list[str]:
    if len(command) > _MAX_COMMAND_TOKENIZATION_CHARS:
        return [command]
    return [segment.strip() for segment in re.split(r"\n|&&|\|\||;", command) if segment.strip()]


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


def _tokens_for_segment(segment: str) -> list[str]:
    if len(segment) > _MAX_COMMAND_TOKENIZATION_CHARS:
        return []
    try:
        return shlex.split(segment)
    except Exception:
        return []


def _normalize_command_path_token(token: str, workspace_path: Path) -> str | None:
    if not token or token.startswith("-"):
        return None
    if token in {"|", "&&", "||", ";", "<", ">", ">>"}:
        return None
    if any(char in token for char in "*?[]{}\\|"):
        return None
    if token.startswith(("http://", "https://")):
        return None
    if token.endswith("p") and "," in token:
        return None
    normalized, _ = _normalize_inferred_file_path(token, workspace_path)
    return normalized


def _path_token_from_tokens(tokens: list[str], workspace_path: Path, *, reverse: bool = False) -> str | None:
    iterable = reversed(tokens) if reverse else iter(tokens)
    for token in iterable:
        normalized = _normalize_command_path_token(token, workspace_path)
        if normalized:
            return normalized
    return None


def _merge_step(target_files: set[str], target_spans: SpanMap, step: RetrievalStep | None) -> None:
    if not step:
        return
    target_files.update(step.get("files", []))
    for file_path, spans in step.get("spans", {}).items():
        target_files.add(file_path)
        target_spans.setdefault(file_path, []).extend(spans)


def _sed_range_from_tokens(tokens: list[str]) -> tuple[int, int] | None:
    for token in tokens:
        match = _SED_RANGE_RE.fullmatch(token.strip("'\""))
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start <= 0 or end <= 0:
            return None
        return min(start, end), max(start, end)
    return None


def _head_count_from_tokens(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token == "-n" and index + 1 < len(tokens):
            match = _HEAD_COUNT_RE.fullmatch(tokens[index + 1].strip())
            if match:
                return int(match.group("count"))
            return None
        if token.startswith("--lines="):
            match = _HEAD_COUNT_RE.fullmatch(token.split("=", 1)[1].strip())
            if match:
                return int(match.group("count"))
            return None
        if token.startswith("-") and len(token) > 1:
            match = _HEAD_COUNT_RE.fullmatch(token[1:])
            if match:
                return int(match.group("count"))
    return None


def _first_pipe_index_after(tokens: list[str], start: int) -> int:
    for index in range(start, len(tokens)):
        if tokens[index] == "|":
            return index
    return len(tokens)


def _has_redirection_token(tokens: list[str]) -> bool:
    return any(
        token in {"<", ">", ">>", "<<", "<<<"} or token.startswith((">", ">>")) or ">" in token or token.startswith("<")
        for token in tokens
    )


def _file_before_pipe(tokens: list[str], workspace_path: Path) -> str | None:
    before_pipe = tokens[: tokens.index("|")] if "|" in tokens else tokens
    return _path_token_from_tokens(before_pipe, workspace_path, reverse=True)


def _file_after_command(tokens: list[str], command_word: str, workspace_path: Path) -> str | None:
    try:
        start = tokens.index(command_word) + 1
    except ValueError:
        return None
    stop = _first_pipe_index_after(tokens, start)
    return _path_token_from_tokens(tokens[start:stop], workspace_path, reverse=True)


def _read_segment_step(segment: str, output_text: str, workspace_path: Path) -> RetrievalStep | None:
    if not output_text.strip():
        return None
    tokens = _tokens_for_segment(segment)
    if not tokens:
        path_token = _find_path_like_token(segment)
        if not path_token:
            return None
        file_path, _ = _normalize_inferred_file_path(path_token, workspace_path)
        if not file_path:
            return None
        span = infer_read_span_from_text(output_text)
        spans: SpanMap = {file_path: [{"start": span[0], "end": span[1]}]} if span else {}
        return {"files": [file_path], "spans": spans, "symbols": {}}

    spans: SpanMap = {}
    files: set[str] = set()

    if "sed" in tokens:
        sed_range = _sed_range_from_tokens(tokens)
        if "|" in tokens:
            # Only credit the file left of the pipe when a read-like command
            # produced the piped text; `python script.py | sed -n '1,50p'`
            # filters program output, not file content.
            before_pipe = tokens[: tokens.index("|")]
            read_like = {"sed", "cat", "head", "tail", "nl"}
            file_path = _file_before_pipe(tokens, workspace_path) if any(token in read_like for token in before_pipe) else None
        else:
            file_path = _file_after_command(tokens, "sed", workspace_path)
        if sed_range and file_path:
            files.add(file_path)
            spans.setdefault(file_path, []).append({"start": sed_range[0], "end": sed_range[1]})
            return {"files": sorted(files), "spans": spans, "symbols": {}}

    if "head" in tokens and tokens[0] == "head":
        count = _head_count_from_tokens(tokens)
        file_path = _file_after_command(tokens, "head", workspace_path)
        if count and file_path:
            files.add(file_path)
            spans.setdefault(file_path, []).append({"start": 1, "end": count})
            return {"files": sorted(files), "spans": spans, "symbols": {}}

    for command_word in ("nl", "cat", "tail"):
        if command_word not in tokens:
            continue
        if _has_redirection_token(tokens):
            continue
        file_path = _file_after_command(tokens, command_word, workspace_path)
        if not file_path:
            continue
        files.add(file_path)
        span = infer_read_span_from_text(output_text)
        if span:
            spans.setdefault(file_path, []).append({"start": span[0], "end": span[1]})
        return {"files": sorted(files), "spans": spans, "symbols": {}}

    return None


def _read_like_step(raw_command: str, output_text: str, workspace_path: Path) -> RetrievalStep | None:
    files: set[str] = set()
    spans: SpanMap = {}
    for segment in _split_command_segments(raw_command):
        _merge_step(files, spans, _read_segment_step(segment, output_text, workspace_path))
    if not files and not spans:
        return None
    return {"files": sorted(files), "spans": spans, "symbols": {}}


def _explicit_single_search_file_from_tokens(tokens: list[str], workspace_path: Path) -> str | None:
    command_indices = [index for index, token in enumerate(tokens) if token in {"grep", "rg"}]
    if not command_indices:
        return None

    start = command_indices[0] + 1
    stop = _first_pipe_index_after(tokens, start)
    candidates: list[str] = []
    for token in tokens[start:stop]:
        normalized = _normalize_command_path_token(token, workspace_path)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates[0] if len(candidates) == 1 else None


def _search_command_can_expose_file_content(tokens: list[str], output_text: str) -> bool:
    if not output_text.strip():
        return False
    if _has_redirection_token(tokens):
        return False
    content_suppressing_flags = {
        "-q",
        "--quiet",
        "-l",
        "--files-with-matches",
        "-L",
        "--files-without-match",
        "-c",
        "--count",
        "--files",
    }
    return not any(token in content_suppressing_flags for token in tokens)


def infer_search_file_step_from_command(command: str, *, output_text: str, workspace_path: Path) -> RetrievalStep | None:
    """Infer conservative retrieval context from a single-file grep/rg command."""

    files: set[str] = set()
    spans: SpanMap = {}
    for segment in _split_command_segments(unwrap_shell_command(command)):
        tokens = _tokens_for_segment(segment)
        if not tokens or not _has_grep_like_command(segment, tokens):
            continue
        if not _search_command_can_expose_file_content(tokens, output_text):
            continue
        file_path = _explicit_single_search_file_from_tokens(tokens, workspace_path)
        if file_path:
            files.add(file_path)
    if not files:
        return None
    allow_line_only_spans = len(files) == 1
    for file_path in sorted(files):
        spans = merge_span_maps(
            spans,
            _single_file_search_spans_from_text(
                output_text,
                file_path,
                workspace_path,
                allow_line_only_spans=allow_line_only_spans,
            ),
        )
    return {"files": sorted(files), "spans": spans, "symbols": {}}


def _single_file_search_spans_from_text(
    text: str,
    file_path: str,
    workspace_path: Path,
    *,
    allow_line_only_spans: bool,
) -> SpanMap:
    parsed_spans = infer_grep_spans_from_text(text, workspace_path)
    if file_spans := parsed_spans.get(file_path):
        return {file_path: file_spans}
    if not allow_line_only_spans:
        return {}

    line_spans: list[dict[str, int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > _MAX_GREP_LINE_CHARS:
            continue
        line_part, _, _ = line.partition(":")
        line_no = _coerce_positive_int(line_part)
        if line_no is not None:
            line_spans.append({"start": line_no, "end": line_no})
        if len(line_spans) >= _MAX_GREP_SPAN_MATCHES:
            break
    return {file_path: line_spans} if line_spans else {}


_GREP_NO_MATCH_MARKERS = ("no matches found", "no files found")


def grep_output_indicates_match(output_text: str) -> bool:
    text = output_text.strip()
    if not text:
        return False
    return not text.lower().startswith(_GREP_NO_MATCH_MARKERS)


def _looks_like_single_search_file(name: str) -> bool:
    if not _looks_like_repo_filename(name):
        return False
    # Hidden directories ('.github') and version-suffixed directories
    # ('jquery-3.6') pass the generic filename heuristic but are not files.
    if name.startswith(".") and "." not in name[1:]:
        return False
    if "." in name and name.rsplit(".", 1)[-1].isdigit():
        return False
    return True


def infer_search_file_step_from_path(path_value: str, *, output_text: str, workspace_path: Path) -> RetrievalStep | None:
    """Infer file-only context when a search tool was explicitly scoped to one file."""

    if not grep_output_indicates_match(output_text):
        return None
    normalized, _ = _normalize_inferred_file_path(path_value, workspace_path)
    if not normalized:
        return None
    if not _looks_like_single_search_file(normalized.rsplit("/", 1)[-1]):
        return None
    return {"files": [normalized], "spans": {}, "symbols": {}}


def _has_read_like_command(raw_command: str, tokens: list[str]) -> bool:
    return any(token in {"sed", "cat", "head", "tail", "nl"} for token in tokens) or any(
        _command_has_word(raw_command, word) for word in ("sed", "cat", "head", "tail", "nl")
    )


def _has_grep_like_command(raw_command: str, tokens: list[str]) -> bool:
    return "rg" in tokens or "grep" in tokens or _command_has_word(raw_command, "rg") or _command_has_word(raw_command, "grep")


def _has_find_command(raw_command: str, tokens: list[str]) -> bool:
    return "find" in tokens or _command_has_word(raw_command, "find")


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

    read_step = _read_like_step(raw_command, output_text, workspace_path) if _has_read_like_command(raw_command, tokens) else None
    if read_step:
        return read_step

    if _has_grep_like_command(raw_command, tokens):
        return infer_search_file_step_from_command(raw_command, output_text=output_text, workspace_path=workspace_path)

    if _has_find_command(raw_command, tokens):
        return None

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
    spans = {
        file_path: sorted(file_spans, key=lambda span: (span["start"], span["end"]))
        for file_path, file_spans in merge_span_maps(*(step.get("spans") for step in steps)).items()
    }
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
