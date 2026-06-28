# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only verification quality analysis for coding-agent run records."""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from .types import CommandExecution

_RUNTIME_VERIFICATION_CATEGORIES = frozenset({"repo_tests", "targeted_runtime"})
_STATIC_VERIFICATION_CATEGORIES = frozenset({"syntax_or_static"})
_DEPENDENCY_INSTALL_RE = re.compile(
    r"(^|\s)(python\d*(?:\.\d+)?\s+-m\s+pip|pip\d*|uv\s+pip|npm|pnpm|yarn|bundle|gem|cargo|go)\s+"
    r"(install|add|ci|sync|fetch|download|mod\s+download)\b"
)
_REPO_TEST_RE = re.compile(
    r"(^|\s)("
    r"pytest|py\.test|tox|nox|unittest|runtests?\.py|manage\.py\s+test|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|"
    r"go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|\.\/gradlew\s+test|"
    r"bundle\s+exec\s+rspec|rspec|rails\s+test|mix\s+test"
    r")(\s|$)"
)
_STATIC_CHECK_RE = re.compile(
    r"(^|\s)("
    r"py_compile|compileall|tsc(?:\s+--noEmit|\s+-p|\s|$)|mypy|pyright|ruff|flake8|pylint|"
    r"eslint|prettier|cargo\s+check|go\s+vet|javac"
    r")(\s|$)"
)
_TARGETED_RUNTIME_RE = re.compile(
    r"(^|\s)(python\d*(?:\.\d+)?|node|ruby|php|perl|Rscript|go\s+run|cargo\s+run|java)\b"
)
_ENV_LIMITATION_PATTERNS = (
    re.compile(r"\b(could not|couldn't|unable to|cannot|can't)\s+(run|execute)\s+(the\s+)?tests?\b", re.I),
    re.compile(r"\btests?\s+(could not|couldn't|cannot|can't|were not able to)\s+(run|execute)\b", re.I),
    re.compile(r"\bmissing\s+(dependency|dependencies|package|module|library)\b", re.I),
    re.compile(r"\b(module|package)\s+not\s+found\b", re.I),
    re.compile(r"\bno module named\b", re.I),
    re.compile(r"\bcommand\s+not\s+found\b", re.I),
    re.compile(r"\benvironment\s+(?:does not|doesn't|cannot|can't)\s+(?:support|provide|allow)\b", re.I),
)
_TEST_PATH_RE = re.compile(
    r"(^|/)(__tests__|tests?|specs?)(/|$)|"
    r"(^|/)(test_[^/]+|[^/]+_test)\.[A-Za-z0-9]+$|"
    r"(^|/)([^/]+)\.(test|spec)\.[A-Za-z0-9]+$|"
    r"(^|/)Test[A-Za-z0-9_]*\.(java|kt|scala)$|"
    r"(^|/)[A-Za-z0-9_]*Test\.(java|kt|scala)$"
)
_ADDED_TEST_LINE_RE = re.compile(
    r"(^|\b)(assert|expect\s*\(|should\s*\(|describe\s*\(|it\s*\(|test\s*\(|"
    r"def\s+test_|async\s+def\s+test_|class\s+Test|@Test\b|pytest|unittest|rspec|"
    r"assertEquals|assertThat|assertEqual)\b"
)
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def command_execution_succeeded(execution: dict[str, Any]) -> bool:
    payload = execution.get("payload") if isinstance(execution.get("payload"), dict) else {}
    candidates = [execution, payload]
    for source in candidates:
        if not isinstance(source, dict):
            continue
        if source.get("ok") is False:
            return False
        for key in ("exit_code", "exitCode"):
            if key in source and source.get(key) is not None:
                try:
                    return int(source.get(key)) == 0
                except (TypeError, ValueError):
                    return False
        if source.get("timeout") is True:
            return False
        if source.get("is_error") is True:
            return False
        result = source.get("result")
        if isinstance(result, dict):
            if result.get("ok") is False or result.get("is_error") is True:
                return False
            for key in ("exit_code", "exitCode"):
                if key in result and result.get(key) is not None:
                    try:
                        return int(result.get(key)) == 0
                    except (TypeError, ValueError):
                        return False
        status = str(source.get("status") or "").strip().lower()
        if status in {"cancelled", "canceled", "denied", "error", "failed", "failure", "rejected", "timeout"}:
            return False
    return True


def _command_output_text(execution: dict[str, Any], *, limit: int = 20_000) -> str:
    payload = execution.get("payload") if isinstance(execution.get("payload"), dict) else {}
    parts: list[str] = []
    for source in (execution, payload):
        if not isinstance(source, dict):
            continue
        for key in ("aggregated_output", "output", "stdout", "stderr", "result", "content"):
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                try:
                    text = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    text = str(value)
            else:
                text = str(value)
            if text.strip():
                parts.append(text)
    text = "\n".join(parts)
    return text[-limit:] if len(text) > limit else text


def classify_command(command: str) -> str:
    normalized = " ".join(str(command or "").strip().split())
    lowered = normalized.lower()
    if not lowered:
        return "other"
    if _DEPENDENCY_INSTALL_RE.search(lowered):
        return "dependency_install"
    if _REPO_TEST_RE.search(lowered):
        return "repo_tests"
    if "python" in lowered and "-m py_compile" in lowered:
        return "syntax_or_static"
    if _STATIC_CHECK_RE.search(lowered):
        return "syntax_or_static"
    if _TARGETED_RUNTIME_RE.search(lowered):
        return "targeted_runtime"
    if re.search(r"(^|\s)make\s+(test|check)\b", lowered):
        return "repo_tests"
    return "other"


def _final_output_text(record: dict[str, Any]) -> str:
    final_output = record.get("final_output") if isinstance(record.get("final_output"), dict) else {}
    parts = [
        str(final_output.get("final_answer") or ""),
        str(final_output.get("notes") or ""),
        str(record.get("notes") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _environment_limitation_matches(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in _ENV_LIMITATION_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = match.group(0).strip()
            if snippet not in matches:
                matches.append(snippet)
    return matches


def _extract_claude_bash_command_executions(raw_response: dict[str, Any]) -> list[CommandExecution]:
    response = raw_response.get("response")
    if not isinstance(response, list):
        return []
    pending: dict[str, dict[str, Any]] = {}
    executions: list[CommandExecution] = []
    for event in response:
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        content_items = message.get("content") if isinstance(message, dict) else []
        if not isinstance(content_items, list):
            continue
        if event.get("type") == "assistant":
            for content in content_items:
                if not isinstance(content, dict) or content.get("type") != "tool_use":
                    continue
                if str(content.get("name") or "") != "Bash":
                    continue
                tool_input = content.get("input") if isinstance(content.get("input"), dict) else {}
                command = str(tool_input.get("command") or "").strip()
                tool_id = str(content.get("id") or "").strip()
                if command and tool_id:
                    pending[tool_id] = {"command": command, "tool_use": dict(content)}
            continue
        if event.get("type") != "user":
            continue
        for content in content_items:
            if not isinstance(content, dict) or content.get("type") != "tool_result":
                continue
            tool_id = str(content.get("tool_use_id") or "").strip()
            pending_item = pending.pop(tool_id, None)
            if not pending_item:
                continue
            payload = {
                "status": "error" if bool(content.get("is_error")) else "completed",
                "is_error": bool(content.get("is_error")),
                "result": content.get("content"),
                "tool_use_result": event.get("tool_use_result"),
            }
            executions.append(
                {
                    "source": "claude.bash",
                    "command": str(pending_item["command"]),
                    "payload": payload,
                }
            )
    return executions


def _extract_codex_command_executions(raw_response: dict[str, Any]) -> list[CommandExecution]:
    events = raw_response.get("events")
    if not isinstance(events, list):
        return []
    executions: list[CommandExecution] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if not isinstance(item, dict) or str(item.get("type") or "") != "command_execution":
            continue
        command = str(item.get("command") or "").strip()
        if not command:
            continue
        payload = dict(item)
        payload.setdefault("event_type", event.get("type"))
        executions.append(
            {
                "source": "codex.raw_event",
                "command": command,
                "payload": payload,
            }
        )
    return executions


def command_executions_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    executions = [dict(item) for item in (record.get("command_executions") or []) if isinstance(item, dict)]
    if executions:
        return executions
    raw_response = record.get("raw_response")
    if raw_response is None and record.get("raw_response_path"):
        raw_response_path = Path(str(record.get("raw_response_path")))
        if raw_response_path.exists():
            try:
                raw_response = json.loads(raw_response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw_response = None
    if isinstance(raw_response, dict):
        return [
            dict(item)
            for item in (
                _extract_codex_command_executions(raw_response)
                or _extract_claude_bash_command_executions(raw_response)
            )
        ]
    return []


def analyze_verification_quality(record: dict[str, Any]) -> dict[str, Any]:
    executions = command_executions_from_record(record)
    categories = Counter()
    successful_categories = Counter()
    failed_commands = 0
    successful_commands = 0
    verification_commands: list[dict[str, Any]] = []
    command_text_for_environment: list[str] = []

    for execution in executions:
        command = str(execution.get("command") or "").strip()
        if not command:
            continue
        category = classify_command(command)
        succeeded = command_execution_succeeded(execution)
        categories[category] += 1
        if succeeded:
            successful_commands += 1
            successful_categories[category] += 1
        else:
            failed_commands += 1
        output_text = _command_output_text(execution)
        if output_text:
            command_text_for_environment.append(output_text)
        if category != "other":
            verification_commands.append(
                {
                    "command": command,
                    "category": category,
                    "succeeded": succeeded,
                }
            )

    successful_runtime_verification = any(successful_categories[category] for category in _RUNTIME_VERIFICATION_CATEGORIES)
    successful_static_verification = any(successful_categories[category] for category in _STATIC_VERIFICATION_CATEGORIES)
    if successful_categories["repo_tests"]:
        strongest = "repo_tests"
    elif successful_categories["targeted_runtime"]:
        strongest = "targeted_runtime"
    elif successful_static_verification:
        strongest = "syntax_or_static"
    elif successful_categories["dependency_install"]:
        strongest = "dependency_install"
    elif executions:
        strongest = "commands_without_successful_verification"
    else:
        strongest = "no_verification"

    environment_text = "\n".join([_final_output_text(record), *command_text_for_environment])
    limitation_matches = _environment_limitation_matches(environment_text)
    return {
        "schema_version": 1,
        "strongest_verification": strongest,
        "successful_runtime_verification": successful_runtime_verification,
        "successful_static_verification": successful_static_verification,
        "syntax_only": strongest == "syntax_or_static" and not successful_runtime_verification,
        "dependency_blocked": bool(limitation_matches),
        "environment_limited": bool(limitation_matches),
        "environment_limitation_matches": limitation_matches[:10],
        "commands_total": len(executions),
        "successful_commands_total": successful_commands,
        "failed_commands_total": failed_commands,
        "command_categories": dict(sorted(categories.items())),
        "successful_command_categories": dict(sorted(successful_categories.items())),
        "verification_commands": verification_commands[:25],
    }


def _diff_path_from_line(line: str) -> str:
    path = line[4:].strip() if line.startswith("+++ ") else line.strip()
    if path.startswith("b/"):
        path = path[2:]
    return "" if path == "/dev/null" else path


def _path_is_likely_test(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").strip()
    return bool(normalized and _TEST_PATH_RE.search(normalized))


def _path_overlap(left: str, right: str) -> bool:
    lhs = str(left or "").replace("\\", "/").strip().strip("/")
    rhs = str(right or "").replace("\\", "/").strip().strip("/")
    if not lhs or not rhs:
        return False
    lhs_base = lhs.rsplit("/", 1)[-1]
    rhs_base = rhs.rsplit("/", 1)[-1]
    return (
        lhs == rhs
        or lhs.endswith(f"/{rhs}")
        or rhs.endswith(f"/{lhs}")
        or lhs_base == rhs_base
    )


def detect_added_regression_tests(model_patch: str, command_executions: list[dict[str, Any]]) -> dict[str, Any]:
    current_path = ""
    candidate_paths: set[str] = set()
    test_signal_paths: set[str] = set()
    added_assertion_paths: set[str] = set()

    for line in str(model_patch or "").splitlines():
        match = _DIFF_GIT_RE.match(line)
        if match:
            current_path = match.group(2).strip()
            if current_path == "/dev/null":
                current_path = match.group(1).strip()
            if _path_is_likely_test(current_path):
                candidate_paths.add(current_path)
            continue
        if line.startswith("+++ "):
            current_path = _diff_path_from_line(line) or current_path
            if _path_is_likely_test(current_path):
                candidate_paths.add(current_path)
            continue
        if not current_path or not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:].strip()
        if not added:
            continue
        if _path_is_likely_test(current_path):
            test_signal_paths.add(current_path)
        if _ADDED_TEST_LINE_RE.search(added):
            added_assertion_paths.add(current_path)

    added_test_files = sorted(candidate_paths | test_signal_paths | added_assertion_paths)
    if not added_test_files:
        return {
            "schema_version": 1,
            "added_regression_test": False,
            "regression_tests_run": None,
            "added_test_files": [],
            "covering_commands": [],
            "reason": "no_added_regression_tests_detected",
        }

    covering_commands: list[str] = []
    successful_test_commands = [
        execution
        for execution in command_executions
        if command_execution_succeeded(execution) and classify_command(str(execution.get("command") or "")) == "repo_tests"
    ]
    for execution in successful_test_commands:
        command = str(execution.get("command") or "").strip()
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        broad_test_command = not any(
            _path_is_likely_test(token) or token.endswith((".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs"))
            for token in tokens
        )
        command_mentions_added_file = any(
            _path_overlap(path, token) or path in command
            for path in added_test_files
            for token in tokens
        )
        if broad_test_command or command_mentions_added_file:
            covering_commands.append(command)

    return {
        "schema_version": 1,
        "added_regression_test": True,
        "regression_tests_run": bool(covering_commands),
        "added_test_files": added_test_files,
        "covering_commands": covering_commands[:10],
        "reason": "covering_successful_test_command" if covering_commands else "no_successful_covering_test_command",
    }


def analyze_record_quality(record: dict[str, Any]) -> dict[str, Any]:
    commands = command_executions_from_record(record)
    return {
        "verification_quality": analyze_verification_quality({**record, "command_executions": commands}),
        "regression_test": detect_added_regression_tests(str(record.get("model_patch") or ""), commands),
    }
