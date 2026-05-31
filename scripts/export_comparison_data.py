
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from contextbench.artifact_sanitization import (
    SanitizationContext,
    assert_no_private_paths,
    sanitize_json_value,
    sanitize_text,
)
from contextbench.coding_agents.records import normalize_span_map, normalize_symbol_map
from contextbench.evaluate import aggregate_results
from contextbench.metrics.patch_editloc import compute_patch_editloc, compute_patch_to_patch_overlap
from contextbench.parsers import GoldLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_DIR = REPO_ROOT / "results" / "run_suites" / "codex-superpowers-mounted"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "site-data" / "comparison.json"
DEFAULT_DETAIL_DIR = REPO_ROOT / "site-data" / "instances"
DEFAULT_VARIANT = "with-superpowers-mounted"
DEFAULT_REPO_CACHE_DIR = REPO_ROOT / ".cache" / "repos"


class ComparisonExportError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve_path(path_like: str | Path, suite_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return REPO_ROOT / path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return suite_dir / path


def _variant_artifact_path(
    variant_manifest: dict[str, Any],
    *,
    suite_dir: Path,
    stem: str,
    artifact_suffix: str | None,
) -> Path:
    suffix = str(artifact_suffix or "").strip().strip(".")
    filename = f"{stem}.{suffix}.jsonl" if suffix else f"{stem}.jsonl"
    return _resolve_path(Path(variant_manifest["output_dir"]) / filename, suite_dir)


def _titleize(value: str | None) -> str:
    raw = str(value or "").replace("-", " ").replace("_", " ").strip()
    return raw.title() if raw else "Unknown"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_optional_percent(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    return _format_percent(float(value))


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _format_pattern_metric(value: float) -> str:
    return f"{value:.2f}"


def _format_duration_ms(value: float) -> str:
    total_seconds = int(round(value / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def _format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def _format_currency(value: float) -> str:
    return f"${value:.2f}"


def _format_rate(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return _format_percent(count / total)


def _github_repo_slug(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"github\.com[:/]([^/\s]+/[^/\s.]+)(?:\.git)?", text)
    if match:
        return match.group(1)
    if "/" in text and "://" not in text:
        owner, repo = text.split("/", 1)
        if owner and repo:
            return f"{owner}/{repo.removesuffix('.git')}"
    return None


def _repo_slug_from_instance_id(value: str | None) -> str | None:
    text = str(value or "").strip()
    if "__" not in text:
        return None
    owner_raw, repo_raw = text.split("__", 1)
    owner = owner_raw.removeprefix("instance_")
    repo = re.sub(r"-[0-9a-f]{40}(?:-v[0-9a-z]+)?$", "", repo_raw, flags=re.IGNORECASE)
    repo = re.sub(r"-v[0-9a-z]+$", "", repo, flags=re.IGNORECASE)
    repo = re.sub(r"-\d+$", "", repo)
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _repository_slug(task_row: dict[str, Any], record: dict[str, Any]) -> str | None:
    return (
        _github_repo_slug(record.get("repo_url"))
        or _github_repo_slug(task_row.get("repo_url"))
        or _github_repo_slug(task_row.get("repo"))
        or _repo_slug_from_instance_id(record.get("original_inst_id"))
        or _repo_slug_from_instance_id(task_row.get("original_inst_id"))
        or _repo_slug_from_instance_id(task_row.get("instance_id"))
    )


_REPOSITORY_SIZE_CACHE: dict[tuple[str, str, bool], dict[str, Any]] = {}


def _repository_line_count(repo_dir: Path, commit: str) -> int | None:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "grep", "-I", "-c", "-e", "^", commit, "--", "."],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0 or result.stderr.strip():
        return None

    total = 0
    for line in result.stdout.splitlines():
        _prefix, separator, count_text = line.rpartition(":")
        if not separator:
            return None
        try:
            total += int(count_text)
        except ValueError:
            return None
    return total


def _repository_size_payload(
    task_row: dict[str, Any],
    record: dict[str, Any],
    *,
    include_line_counts: bool = False,
) -> dict[str, Any] | None:
    repo = _repository_slug(task_row, record)
    if not repo:
        return None

    commit = str(record.get("commit") or task_row.get("commit") or task_row.get("base_commit") or "").strip()
    if not commit:
        return {"status": "unavailable", "reason": "missing_commit", "repo": repo}

    cache_key = (repo, commit, include_line_counts)
    if cache_key in _REPOSITORY_SIZE_CACHE:
        return _REPOSITORY_SIZE_CACHE[cache_key]

    repo_dir = DEFAULT_REPO_CACHE_DIR / f"github.com__{repo.replace('/', '__')}"
    if not repo_dir.exists():
        payload = {"status": "unavailable", "reason": "missing_repo_cache", "repo": repo, "commit": commit}
        _REPOSITORY_SIZE_CACHE[cache_key] = payload
        return payload

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "-z", commit],
        check=False,
        capture_output=True,
        text=False,
    )
    if result.returncode != 0:
        payload = {"status": "unavailable", "reason": "missing_commit", "repo": repo, "commit": commit}
        _REPOSITORY_SIZE_CACHE[cache_key] = payload
        return payload

    tracked_files = 0
    for raw_entry in result.stdout.split(b"\0"):
        if not raw_entry:
            continue
        tracked_files += 1

    payload = {
        "status": "available",
        "repo": repo,
        "trackedFiles": tracked_files,
    }
    if include_line_counts:
        tracked_lines = _repository_line_count(repo_dir, commit)
        if tracked_lines is None:
            payload["lineCountStatus"] = "unavailable"
            payload["lineCountReason"] = "missing_blob_contents"
        else:
            payload["lineCountStatus"] = "available"
            payload["trackedTextLines"] = tracked_lines
    _REPOSITORY_SIZE_CACHE[cache_key] = payload
    return payload


def _safe_mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return (sum(filtered) / len(filtered)) if filtered else None


def _coverage_precision(pred_size: int, gold_size: int, intersection: int) -> tuple[float, float]:
    coverage = (intersection / gold_size) if gold_size else 0.0
    precision = (intersection / pred_size) if pred_size else 0.0
    return coverage, precision


def _f1(coverage: float, precision: float) -> float:
    denominator = coverage + precision
    if denominator == 0:
        return 0.0
    return 2 * coverage * precision / denominator


def _unavailable_overlap(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "intersection": 0,
        "gold_size": 0,
        "pred_size": 0,
        "recall": None,
        "precision": None,
        "f1": None,
    }


def _fix_overlap_vs_gold_payload(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": metric.get("status") or "unavailable",
        "reason": metric.get("reason"),
        "recall": metric.get("recall"),
        "precision": metric.get("precision"),
        "f1": metric.get("f1"),
        "intersection": int(metric.get("intersection") or 0),
        "goldSize": int(metric.get("gold_size") or 0),
        "predSize": int(metric.get("pred_size") or 0),
    }


def _fix_overlap_pair_payload(
    metric: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    return {
        "status": metric.get("status") or "unavailable",
        "reason": metric.get("reason"),
        "leftLabel": left_label,
        "rightLabel": right_label,
        "leftCoveredByRight": metric.get("recall"),
        "rightCoveredByLeft": metric.get("precision"),
        "f1": metric.get("f1"),
        "intersection": int(metric.get("intersection") or 0),
        "leftSize": int(metric.get("gold_size") or 0),
        "rightSize": int(metric.get("pred_size") or 0),
    }


def _aggregate_overlap_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    available = [metric for metric in metrics if metric.get("status") == "available"]
    if not metrics:
        return {"status": "unavailable", "reason": "no_instances"}
    if not available:
        return {
            "status": "unavailable",
            "reason": "no_available_instances",
            "availableInstances": 0,
            "unavailableInstances": len(metrics),
        }

    intersection = sum(int(metric.get("intersection") or 0) for metric in available)
    gold_size = sum(int(metric.get("gold_size") or 0) for metric in available)
    pred_size = sum(int(metric.get("pred_size") or 0) for metric in available)
    recall = intersection / gold_size if gold_size else 0.0
    precision = intersection / pred_size if pred_size else 0.0
    f1 = _f1(recall, precision)
    return {
        "status": "available",
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "intersection": intersection,
        "goldSize": gold_size,
        "predSize": pred_size,
        "availableInstances": len(available),
        "unavailableInstances": len(metrics) - len(available),
    }


def _format_fix_overlap_summary(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": metric.get("status") or "unavailable",
        "recall": _format_optional_percent(metric.get("recall")),
        "precision": _format_optional_percent(metric.get("precision")),
        "f1": _format_optional_percent(metric.get("f1")),
        "availableInstances": int(metric.get("availableInstances") or 0),
        "unavailableInstances": int(metric.get("unavailableInstances") or 0),
    }


def _format_pair_overlap_summary(
    metric: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    return {
        "status": metric.get("status") or "unavailable",
        "leftLabel": left_label,
        "rightLabel": right_label,
        "leftCoveredByRight": _format_optional_percent(metric.get("recall")),
        "rightCoveredByLeft": _format_optional_percent(metric.get("precision")),
        "f1": _format_optional_percent(metric.get("f1")),
        "availableInstances": int(metric.get("availableInstances") or 0),
        "unavailableInstances": int(metric.get("unavailableInstances") or 0),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _mounted_resources(copy_paths: list[dict[str, Any]]) -> str:
    if not copy_paths:
        return "None"
    if any("superpowers" in str(entry.get("source", "")).lower() for entry in copy_paths):
        return "Superpowers snapshot"
    return "Mounted resources"


def _setup_parameters(effective_config: dict[str, Any]) -> list[dict[str, str]]:
    setup = effective_config.get("setup") or {}
    parameters = [
        {"label": "Model", "value": str(effective_config.get("model") or "Unknown")},
        {"label": "Reasoning Effort", "value": _titleize(str(effective_config.get("reasoning_effort") or "unknown"))},
        {"label": "Timeout", "value": f"{int(effective_config.get('timeout') or 0)}s"},
        {
            "label": "Mounted Resources",
            "value": _mounted_resources(list(setup.get("copy_paths") or [])),
        },
    ]

    prompt_preamble = str(setup.get("prompt_preamble") or "").strip()
    if prompt_preamble:
        parameters.append({"label": "Additional Prompt", "value": prompt_preamble})

    setup_prompt = str(setup.get("setup_prompt") or "").strip()
    if setup_prompt:
        parameters.append({"label": "Bootstrap Prompt", "value": setup_prompt})

    return parameters


def _task_set_payload(task_set: dict[str, Any]) -> dict[str, Any]:
    bench_counts = task_set.get("bench_counts") or {}
    normalized_bench_counts = {
        str(name): int(count)
        for name, count in bench_counts.items()
        if isinstance(name, str) and isinstance(count, (int, float))
    }
    payload: dict[str, Any] = {
        "count": int(task_set.get("count") or 0),
        "hash": str(task_set.get("hash") or ""),
    }
    if normalized_bench_counts:
        payload["benchCounts"] = normalized_bench_counts
    source_count = task_set.get("source_count")
    if isinstance(source_count, (int, float)):
        payload["sourceDatasetCount"] = int(source_count)
    selection_kind = str(task_set.get("selection_kind") or "").strip()
    if selection_kind:
        payload["selectionKind"] = selection_kind
    return payload


def _normalize_context_path(path_value: str, *, workspace_path: str | None, candidates: set[str]) -> str:
    path = str(path_value or "").replace("\\", "/").strip()
    if not path:
        return ""

    workspace = str(workspace_path or "").replace("\\", "/").rstrip("/")
    if workspace and path.startswith(f"{workspace}/"):
        path = path[len(workspace) + 1 :]

    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    if not path:
        return ""

    if path in candidates:
        return path

    matches = [candidate for candidate in candidates if path == candidate or path.endswith(f"/{candidate}")]
    if matches:
        return max(matches, key=len)

    parts = [part for part in path.split("/") if part]
    for index in range(len(parts)):
        candidate = "/".join(parts[index:])
        if candidate in candidates:
            return candidate

    return ""


def _normalize_repo_relative_path(path_value: str, *, workspace_path: str | None) -> str:
    path = str(path_value or "").replace("\\", "/").strip()
    if not path:
        return ""

    workspace = str(workspace_path or "").replace("\\", "/").rstrip("/")
    if workspace and path.startswith(f"{workspace}/"):
        path = path[len(workspace) + 1 :]
    elif Path(path).is_absolute():
        return ""

    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    if not path:
        return ""

    if path.startswith(".agents/") or path.startswith("home/.agents/") or "/.agents/" in path:
        return ""

    return path


def _filter_predicted_traj_data(
    traj_data: dict[str, Any],
    *,
    workspace_path: str | None,
) -> dict[str, Any]:
    filtered_steps: list[dict[str, Any]] = []
    for step in traj_data.get("pred_steps") or []:
        if not isinstance(step, dict):
            continue

        filtered_files = [
            normalized_file
            for raw_file in step.get("files") or []
            if (normalized_file := _normalize_repo_relative_path(str(raw_file), workspace_path=workspace_path))
        ]

        filtered_spans: dict[str, list[dict[str, int]]] = {}
        for file_path, spans in normalize_span_map(step.get("spans")).items():
            normalized_file = _normalize_repo_relative_path(file_path, workspace_path=workspace_path)
            if normalized_file:
                filtered_spans[normalized_file] = spans

        filtered_symbols: dict[str, list[str]] = {}
        for file_path, symbols in normalize_symbol_map(step.get("symbols")).items():
            normalized_file = _normalize_repo_relative_path(file_path, workspace_path=workspace_path)
            if normalized_file:
                filtered_symbols[normalized_file] = symbols

        if filtered_files or filtered_spans or filtered_symbols:
            filtered_steps.append(
                {
                    "files": sorted(set(filtered_files)),
                    "spans": filtered_spans,
                    "symbols": filtered_symbols,
                }
            )

    filtered_files = sorted(
        {
            normalized_file
            for raw_file in traj_data.get("pred_files") or []
            if (normalized_file := _normalize_repo_relative_path(str(raw_file), workspace_path=workspace_path))
        }
    )
    filtered_spans = {
        normalized_file: spans
        for file_path, spans in normalize_span_map(traj_data.get("pred_spans")).items()
        if (normalized_file := _normalize_repo_relative_path(file_path, workspace_path=workspace_path))
    }
    filtered_symbols = {
        normalized_file: symbols
        for file_path, symbols in normalize_symbol_map(traj_data.get("pred_symbols")).items()
        if (normalized_file := _normalize_repo_relative_path(file_path, workspace_path=workspace_path))
    }

    return {
        "pred_steps": filtered_steps,
        "pred_files": filtered_files,
        "pred_spans": filtered_spans,
        "pred_symbols": filtered_symbols,
    }


def _merge_line_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for current in sorted(intervals)[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def _line_total(lines_by_file: dict[str, list[tuple[int, int]]]) -> int:
    return sum(end - start + 1 for intervals in lines_by_file.values() for start, end in _merge_line_intervals(intervals))


def _line_intersection_size(
    left: dict[str, list[tuple[int, int]]],
    right: dict[str, list[tuple[int, int]]],
) -> int:
    total = 0
    for file_path in set(left) | set(right):
        left_intervals = _merge_line_intervals(left.get(file_path, []))
        right_intervals = _merge_line_intervals(right.get(file_path, []))
        left_index = 0
        right_index = 0
        while left_index < len(left_intervals) and right_index < len(right_intervals):
            left_start, left_end = left_intervals[left_index]
            right_start, right_end = right_intervals[right_index]
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_start <= overlap_end:
                total += overlap_end - overlap_start + 1
            if left_end < right_end:
                left_index += 1
            elif right_end < left_end:
                right_index += 1
            else:
                left_index += 1
                right_index += 1
    return total


def _step_line_map(step: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    line_map: dict[str, list[tuple[int, int]]] = {}
    for file_path, spans in normalize_span_map(step.get("spans")).items():
        for span in spans:
            line_map.setdefault(file_path, []).append((span["start"], span["end"]))
    return {file_path: _merge_line_intervals(intervals) for file_path, intervals in line_map.items()}


def _merge_line_maps(*maps: dict[str, list[tuple[int, int]]]) -> dict[str, list[tuple[int, int]]]:
    merged: dict[str, list[tuple[int, int]]] = {}
    for mapping in maps:
        for file_path, intervals in mapping.items():
            merged.setdefault(file_path, []).extend(intervals)
    return {file_path: _merge_line_intervals(intervals) for file_path, intervals in merged.items()}


def _extract_cost_usd(raw_value: Any) -> float | None:
    if isinstance(raw_value, dict):
        direct = raw_value.get("total_cost_usd")
        if isinstance(direct, (int, float)) and not isinstance(direct, bool):
            return float(direct)
        for value in raw_value.values():
            found = _extract_cost_usd(value)
            if found is not None:
                return found
        return None
    if isinstance(raw_value, list):
        for value in raw_value:
            found = _extract_cost_usd(value)
            if found is not None:
                return found
    return None


def _extract_skill_counts(raw_response: dict[str, Any]) -> dict[str, int]:
    skill_pattern = re.compile(r"/\.agents/skills/(?:superpowers/)?([^/]+)/SKILL\.md")
    counts: dict[str, int] = {}
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        item = event.get("item") or {}
        if not isinstance(item, dict):
            continue
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        command = str(item.get("command") or "")
        match = skill_pattern.search(command)
        if match:
            skill_name = match.group(1)
            counts[skill_name] = counts.get(skill_name, 0) + 1
    return counts


def _normalize_resolution_instance_id(task_row: dict[str, Any], record: dict[str, Any]) -> str:
    return str(
        record.get("original_inst_id")
        or task_row.get("original_inst_id")
        or record.get("instance_id")
        or task_row.get("instance_id")
        or ""
    ).strip()


def _build_resolution_status_lookup(summary: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    status_priority = {"resolved": 1, "unresolved": 2, "error": 3}

    def add_ids(values: Any, status: str) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            normalized = str(value or "").strip()
            if normalized and status_priority.get(status, 0) >= status_priority.get(lookup.get(normalized, ""), 0):
                lookup[normalized] = status

    per_bench = summary.get("per_bench")
    if isinstance(per_bench, dict):
        for payload in per_bench.values():
            if not isinstance(payload, dict):
                continue
            add_ids(payload.get("resolved_ids"), "resolved")
            add_ids(payload.get("unresolved_ids"), "unresolved")
            add_ids(payload.get("error_ids"), "error")
            add_ids(payload.get("unknown_ids"), "error")

    add_ids(summary.get("resolved_ids"), "resolved")
    add_ids(summary.get("unresolved_ids"), "unresolved")
    add_ids(summary.get("error_ids"), "error")
    add_ids(summary.get("unknown_ids"), "error")
    return lookup


def _extract_tool_counts(tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool_name") or call.get("source") or "unknown").strip()
        if not tool_name:
            continue
        counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


_CLAUDE_READ_TOOL_NAMES = {"Read", "Grep", "Glob", "LS"}
_CLAUDE_EDIT_TOOL_NAMES = {"Edit", "MultiEdit", "Write", "NotebookEdit"}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _tool_call_is_mcp(call: dict[str, Any]) -> bool:
    tool_name = str(call.get("tool_name") or "").strip()
    if tool_name.startswith("mcp__"):
        return True
    payload = call.get("payload")
    return isinstance(payload, dict) and bool(str(payload.get("mcp_server") or "").strip())


def _tool_call_succeeded(call: dict[str, Any]) -> bool:
    if call.get("ok") is False:
        return False
    payload = call.get("payload")
    if isinstance(payload, dict):
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
    status = str(call.get("status") or "").strip().lower()
    return status not in {"cancelled", "canceled", "denied", "error", "failed", "failure", "rejected", "timeout"}


def _trace_tool_name(event: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    sources: list[dict[str, Any]] = []
    if isinstance(item, dict):
        sources.append(item)
    sources.append(event)
    for source in sources:
        for key in ("tool_name", "toolName", "name"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
        payload = source.get("payload")
        if isinstance(payload, dict):
            for key in ("tool_name", "toolName", "name"):
                value = str(payload.get(key) or "").strip()
            if value:
                return value
    return ""


def _mcp_tool_name_from_sources(*sources: dict[str, Any] | None) -> str:
    usable_sources = [source for source in sources if isinstance(source, dict)]
    nested_sources: list[dict[str, Any]] = []
    for source in usable_sources:
        nested_sources.append(source)
        payload = source.get("payload")
        if isinstance(payload, dict):
            nested_sources.append(payload)

    for source in nested_sources:
        for key in ("tool_name", "toolName", "name"):
            value = str(source.get(key) or "").strip()
            if value.startswith("mcp__"):
                return value

    for source in nested_sources:
        server = str(
            source.get("mcp_server")
            or source.get("mcpServer")
            or source.get("server_name")
            or source.get("serverName")
            or source.get("server")
            or ""
        ).strip()
        tool = str(
            source.get("mcp_tool")
            or source.get("mcpTool")
            or source.get("tool_name")
            or source.get("toolName")
            or source.get("name")
            or ""
        ).strip()
        if tool.startswith("mcp__"):
            return tool
        if server and tool:
            return f"mcp__{server}__{tool}"

    return ""


def _codex_event_is_mcp(event: dict[str, Any], item: dict[str, Any] | None = None) -> bool:
    item_type = str(item.get("type") if isinstance(item, dict) else "").lower()
    event_type = str(event.get("type") or "").lower()
    if "mcp" in item_type or "mcp" in event_type:
        return True
    return bool(_mcp_tool_name_from_sources(item, event))


def _codex_tool_input(event: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
    sources = [source for source in (item, event) if isinstance(source, dict)]
    for source in sources:
        for key in ("input", "arguments", "args", "params", "parameters"):
            value = source.get(key)
            if isinstance(value, dict):
                return dict(value)
            parsed = _parse_jsonish_payload(value)
            if isinstance(parsed, dict) and parsed and "_raw" not in parsed:
                return parsed

    payload: dict[str, Any] = {}
    for source in sources:
        for key in ("query", "top_k", "topK", "path", "file_path", "filePath", "symbol"):
            if source.get(key) is not None and key not in payload:
                payload[key] = source.get(key)
    return payload


def _codex_tool_result(event: dict[str, Any], item: dict[str, Any] | None = None) -> Any:
    sources = [source for source in (item, event) if isinstance(source, dict)]
    for source in sources:
        for key in ("result", "output", "content", "tool_result", "toolResult"):
            if key in source:
                return source.get(key)
    payload = event.get("payload")
    if isinstance(payload, dict):
        for key in ("result", "output", "content"):
            if key in payload:
                return payload.get(key)
    return None


def _trace_mcp_count_from_codex(raw_response: dict[str, Any]) -> int:
    count = 0
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        item_dict = item if isinstance(item, dict) else None
        if _codex_event_is_mcp(event, item_dict):
            count += 1
    return count


def _trace_action_counts(raw_response: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, int]:
    tool_calls = [call for call in (record.get("tool_calls") or []) if isinstance(call, dict)]
    summary = record.get("tool_call_summary") if isinstance(record.get("tool_call_summary"), dict) else {}
    total_tool_calls = _int_or_none(summary.get("total")) if isinstance(summary, dict) else None
    mcp_tool_calls = _int_or_none(summary.get("mcp_total")) if isinstance(summary, dict) else None
    successful_mcp_tool_calls = _int_or_none(summary.get("mcp_successful_total")) if isinstance(summary, dict) else None

    command_executions = 0
    read_tool_calls = 0
    edit_tool_calls = 0
    trace_mcp_tool_calls = 0

    if isinstance(raw_response, dict) and isinstance(raw_response.get("response"), list):
        for event in raw_response.get("response", []):
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            message = event.get("message")
            content_items = message.get("content") if isinstance(message, dict) else []
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict) or content.get("type") != "tool_use":
                    continue
                tool_name = str(content.get("name") or "").strip()
                if tool_name == "Bash":
                    command_executions += 1
                elif tool_name in _CLAUDE_READ_TOOL_NAMES:
                    read_tool_calls += 1
                elif tool_name in _CLAUDE_EDIT_TOOL_NAMES:
                    edit_tool_calls += 1
                if tool_name.startswith("mcp__"):
                    trace_mcp_tool_calls += 1
    elif isinstance(raw_response, dict):
        for event in raw_response.get("events", []):
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            event_type = str(event.get("type") or "")
            if item_type == "command_execution" and event_type == "item.completed":
                command_executions += 1
            elif item_type == "file_change" and event_type == "item.completed":
                edit_tool_calls += 1
        trace_mcp_tool_calls = _trace_mcp_count_from_codex(raw_response)

    if total_tool_calls is None:
        total_tool_calls = len(tool_calls)
    if mcp_tool_calls is None:
        mcp_tool_calls = sum(1 for call in tool_calls if _tool_call_is_mcp(call)) or trace_mcp_tool_calls
    if successful_mcp_tool_calls is None:
        successful_mcp_tool_calls = sum(1 for call in tool_calls if _tool_call_is_mcp(call) and _tool_call_succeeded(call))

    return {
        "toolCalls": int(total_tool_calls or 0),
        "mcpToolCalls": int(mcp_tool_calls or 0),
        "successfulMcpToolCalls": int(successful_mcp_tool_calls or 0),
        "commandExecutions": command_executions,
        "readToolCalls": read_tool_calls,
        "editToolCalls": edit_tool_calls,
    }


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _patch_files_from_diff(diff_text: str) -> list[str]:
    files: list[str] = []
    for line in str(diff_text or "").splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[2]
        if path.startswith("a/"):
            path = path[2:]
        if path and path != "/dev/null":
            files.append(path)
    return sorted(set(files))


def _paths_overlap(left: str, right: str) -> bool:
    lhs = str(left or "").replace("\\", "/").strip().strip("/")
    rhs = str(right or "").replace("\\", "/").strip().strip("/")
    if not lhs or not rhs:
        return False
    return (
        lhs == rhs
        or lhs.endswith(f"/{rhs}")
        or rhs.endswith(f"/{lhs}")
        or lhs.startswith(f"{rhs}/")
        or rhs.startswith(f"{lhs}/")
    )


def _parse_jsonish_payload(value: Any) -> Any:
    if isinstance(value, dict) and "content" in value and len(value) <= 3:
        return _parse_jsonish_payload(value.get("content"))
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return _parse_jsonish_payload("\n".join(parts))
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"_raw": stripped}
    return value


def _mcp_payload_results(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    for key in ("results", "rules", "items", "matches"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalized_result_path(value: Any, *, workspace_path: str | None) -> str:
    return _normalize_repo_relative_path(str(value or ""), workspace_path=workspace_path)


def _mcp_result_paths(results: list[Any], *, workspace_path: str | None) -> list[str]:
    paths: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for key in ("path", "file", "file_path", "filepath"):
            normalized = _normalized_result_path(result.get(key), workspace_path=workspace_path)
            if normalized:
                paths.append(normalized)
                break
    return _unique_strings(paths)


def _mcp_result_symbols(results: list[Any]) -> list[str]:
    symbols: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for key in ("title", "name", "symbol"):
            value = str(result.get(key) or "").strip()
            if value:
                symbols.append(value)
                break
    return _unique_strings(symbols)


def _paths_from_tool_input(tool_name: str, tool_input: Any, *, workspace_path: str | None) -> list[str]:
    if not isinstance(tool_input, dict):
        return []
    paths: list[str] = []
    for key in ("file_path", "path"):
        normalized = _normalize_repo_relative_path(str(tool_input.get(key) or ""), workspace_path=workspace_path)
        if normalized:
            paths.append(normalized)
    command = str(tool_input.get("command") or "")
    if command:
        for raw_path in re.findall(r"/[^\s'\"`]+", command):
            normalized = _normalize_repo_relative_path(raw_path, workspace_path=workspace_path)
            if normalized:
                paths.append(normalized)
        for raw_path in re.findall(r"(?:(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z0-9_]+)?)", command):
            normalized = _normalize_repo_relative_path(raw_path, workspace_path=workspace_path)
            if normalized:
                paths.append(normalized)
    return _unique_strings(paths)


def _mcp_tool_counts(record: dict[str, Any], *, available_tools: list[str]) -> list[dict[str, Any]]:
    summary = record.get("tool_call_summary") if isinstance(record.get("tool_call_summary"), dict) else {}
    by_name = summary.get("by_name") if isinstance(summary.get("by_name"), dict) else {}
    successful_by_name = summary.get("successful_by_name") if isinstance(summary.get("successful_by_name"), dict) else {}
    names = sorted(
        {
            *available_tools,
            *[str(name) for name in by_name if str(name).startswith("mcp__")],
            *[str(name) for name in successful_by_name if str(name).startswith("mcp__")],
        }
    )
    return [
        {
            "name": name,
            "calls": int(by_name.get(name) or 0),
            "successfulCalls": int(successful_by_name.get(name) or 0),
        }
        for name in names
    ]


def _mcp_call_detail(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    result: Any,
    ordered_tool_events: list[tuple[int, str, Any]],
    event_index: int,
    final_files: set[str],
    patch_files: set[str],
    workspace_path: str | None,
    sanitize_context: SanitizationContext,
) -> dict[str, Any]:
    payload = _parse_jsonish_payload(result)
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        payload = _parse_jsonish_payload(payload.get("content"))
    results = _mcp_payload_results(payload)
    result_paths = _mcp_result_paths(results, workspace_path=workspace_path)
    result_symbols = _mcp_result_symbols(results)
    followed_paths: list[str] = []
    for index, later_tool_name, later_input in ordered_tool_events:
        if index <= event_index or str(later_tool_name).startswith("mcp__"):
            continue
        later_paths = _paths_from_tool_input(str(later_tool_name), later_input, workspace_path=workspace_path)
        for result_path in result_paths:
            if any(_paths_overlap(result_path, later_path) for later_path in later_paths):
                followed_paths.append(result_path)
    final_overlap = sorted({path for path in final_files if any(_paths_overlap(path, result_path) for result_path in result_paths)})
    patch_overlap = sorted({path for path in patch_files if any(_paths_overlap(path, result_path) for result_path in result_paths)})
    followed_paths = _unique_strings(followed_paths)
    result_count = len(results)
    meaningful = result_count > 0 and bool(final_overlap or patch_overlap or followed_paths)
    return {
        "toolName": tool_name,
        "query": sanitize_text(str(tool_input.get("query") or ""), context=sanitize_context),
        "topK": _int_or_none(tool_input.get("top_k") or tool_input.get("topK")),
        "resultCount": result_count,
        "totalCandidates": (
            _int_or_none(payload.get("total_candidates") or payload.get("totalCandidates"))
            if isinstance(payload, dict)
            else None
        ),
        "topPaths": sanitize_json_value(result_paths[:5], context=sanitize_context),
        "topSymbols": sanitize_json_value(result_symbols[:5], context=sanitize_context),
        "overlapFinalContextFiles": final_overlap,
        "overlapPatchFiles": patch_overlap,
        "followedReturnedPaths": sanitize_json_value(followed_paths[:5], context=sanitize_context),
        "meaningful": meaningful,
    }


def _claude_mcp_call_details(
    raw_response: dict[str, Any],
    *,
    final_files: set[str],
    patch_files: set[str],
    workspace_path: str | None,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    response = raw_response.get("response")
    if not isinstance(response, list):
        return []
    tool_uses: dict[str, dict[str, Any]] = {}
    ordered_tool_events: list[tuple[int, str, Any]] = []
    for index, event in enumerate(response):
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        content_items = message.get("content") if isinstance(message, dict) else []
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict) or content.get("type") != "tool_use":
                continue
            tool_id = str(content.get("id") or "").strip()
            tool_name = str(content.get("name") or "").strip()
            tool_input = content.get("input") if "input" in content else {}
            ordered_tool_events.append((index, tool_name, tool_input))
            if tool_id and tool_name.startswith("mcp__"):
                tool_uses[tool_id] = {
                    "index": index,
                    "toolName": tool_name,
                    "input": tool_input if isinstance(tool_input, dict) else {},
                    "result": None,
                }
        if event.get("type") != "user":
            continue
        for content in content_items:
            if not isinstance(content, dict) or content.get("type") != "tool_result":
                continue
            tool_id = str(content.get("tool_use_id") or "").strip()
            if tool_id in tool_uses:
                tool_uses[tool_id]["result"] = (
                    event.get("tool_use_result") if event.get("tool_use_result") is not None else content.get("content")
                )

    return [
        _mcp_call_detail(
            tool_name=str(tool_use.get("toolName") or ""),
            tool_input=tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {},
            result=tool_use.get("result"),
            ordered_tool_events=ordered_tool_events,
            event_index=int(tool_use["index"]),
            final_files=final_files,
            patch_files=patch_files,
            workspace_path=workspace_path,
            sanitize_context=sanitize_context,
        )
        for tool_use in sorted(tool_uses.values(), key=lambda item: int(item["index"]))
    ]


def _codex_mcp_call_details(
    raw_response: dict[str, Any],
    *,
    final_files: set[str],
    patch_files: set[str],
    workspace_path: str | None,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    raw_events = raw_response.get("events")
    if not isinstance(raw_events, list):
        return []
    mcp_events: list[dict[str, Any]] = []
    ordered_tool_events: list[tuple[int, str, Any]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        item_dict = item if isinstance(item, dict) else None
        if item_dict and item_dict.get("type") == "command_execution":
            ordered_tool_events.append((index, "command_execution", {"command": item_dict.get("command")}))
            continue
        if not _codex_event_is_mcp(event, item_dict):
            continue
        tool_name = _mcp_tool_name_from_sources(item_dict, event) or _trace_tool_name(event, item_dict)
        if not tool_name.startswith("mcp__"):
            continue
        tool_input = _codex_tool_input(event, item_dict)
        ordered_tool_events.append((index, tool_name, tool_input))
        mcp_events.append(
            {
                "index": index,
                "toolName": tool_name,
                "input": tool_input,
                "result": _codex_tool_result(event, item_dict),
            }
        )

    return [
        _mcp_call_detail(
            tool_name=str(tool_use.get("toolName") or ""),
            tool_input=tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {},
            result=tool_use.get("result"),
            ordered_tool_events=ordered_tool_events,
            event_index=int(tool_use["index"]),
            final_files=final_files,
            patch_files=patch_files,
            workspace_path=workspace_path,
            sanitize_context=sanitize_context,
        )
        for tool_use in mcp_events
    ]


def _record_mcp_call_details(
    record: dict[str, Any],
    *,
    final_files: set[str],
    patch_files: set[str],
    workspace_path: str | None,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(record.get("tool_calls") or []):
        if not isinstance(call, dict) or not _tool_call_is_mcp(call):
            continue
        payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
        tool_name = _mcp_tool_name_from_sources(call, payload) or str(call.get("tool_name") or "").strip()
        if not tool_name.startswith("mcp__"):
            continue
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else _codex_tool_input({}, payload)
        calls.append(
            _mcp_call_detail(
                tool_name=tool_name,
                tool_input=tool_input,
                result=_codex_tool_result({}, payload),
                ordered_tool_events=[],
                event_index=index,
                final_files=final_files,
                patch_files=patch_files,
                workspace_path=workspace_path,
                sanitize_context=sanitize_context,
            )
        )
    return calls


def _extract_mcp_usage(
    raw_response: dict[str, Any] | None,
    record: dict[str, Any],
    *,
    final_output: dict[str, Any],
    model_patch: str,
    workspace_path: str | None,
    candidates: set[str],
    sanitize_context: SanitizationContext,
) -> dict[str, Any]:
    available_tools = sorted(
        {
            str(tool).strip()
            for tool in (record.get("available_tools") or [])
            if str(tool).strip().startswith("mcp__")
        }
    )
    tool_counts = _mcp_tool_counts(record, available_tools=available_tools)
    summary = record.get("tool_call_summary") if isinstance(record.get("tool_call_summary"), dict) else {}
    total_calls = int(summary.get("mcp_total") or sum(entry["calls"] for entry in tool_counts))
    successful_calls = int(summary.get("mcp_successful_total") or sum(entry["successfulCalls"] for entry in tool_counts))

    final_files = set(
        _normalize_reported_files(
            final_output.get("retrieved_context_files"),
            workspace_path=workspace_path,
            candidates=candidates,
        )
    )
    patch_files = set(_patch_files_from_diff(model_patch))
    calls: list[dict[str, Any]] = []

    if isinstance(raw_response, dict) and isinstance(raw_response.get("response"), list):
        calls = _claude_mcp_call_details(
            raw_response,
            final_files=final_files,
            patch_files=patch_files,
            workspace_path=workspace_path,
            sanitize_context=sanitize_context,
        )
    elif isinstance(raw_response, dict):
        calls = _codex_mcp_call_details(
            raw_response,
            final_files=final_files,
            patch_files=patch_files,
            workspace_path=workspace_path,
            sanitize_context=sanitize_context,
        )
    if not calls:
        calls = _record_mcp_call_details(
            record,
            final_files=final_files,
            patch_files=patch_files,
            workspace_path=workspace_path,
            sanitize_context=sanitize_context,
        )

    calls_with_results = sum(1 for call in calls if int(call.get("resultCount") or 0) > 0)
    calls_with_final_overlap = sum(1 for call in calls if call.get("overlapFinalContextFiles"))
    calls_with_patch_overlap = sum(1 for call in calls if call.get("overlapPatchFiles"))
    calls_with_followup = sum(1 for call in calls if call.get("followedReturnedPaths"))
    meaningful_calls = sum(1 for call in calls if call.get("meaningful"))
    returned_paths = sorted({path for call in calls for path in (call.get("topPaths") or [])})

    return {
        "availableTools": available_tools,
        "toolCalls": total_calls,
        "successfulToolCalls": successful_calls,
        "callsWithResults": calls_with_results,
        "meaningfulCalls": meaningful_calls,
        "callsWithFinalContextOverlap": calls_with_final_overlap,
        "callsWithPatchOverlap": calls_with_patch_overlap,
        "callsWithFollowupOnReturnedPath": calls_with_followup,
        "returnedPathCount": len(returned_paths),
        "byTool": tool_counts,
        "calls": calls,
    }


def _normalize_retry_payload(raw_retry: Any, *, sanitize_context: SanitizationContext) -> dict[str, Any]:
    retry = raw_retry if isinstance(raw_retry, dict) else {}
    attempts = int(retry.get("attempts") or 1)
    max_attempts = int(retry.get("max_attempts") or retry.get("maxAttempts") or attempts)
    events = retry.get("events") if isinstance(retry.get("events"), list) else []
    suppression_reason = retry.get("suppression_reason")
    return {
        "attempts": attempts,
        "maxAttempts": max_attempts,
        "retried": bool(retry.get("retried")) or attempts > 1,
        "suppressed": bool(retry.get("suppressed")),
        "suppressionReason": sanitize_text(str(suppression_reason), context=sanitize_context) if suppression_reason else None,
        "events": sanitize_json_value(events, context=sanitize_context),
    }


def _truncate_text(value: str, limit: int = 4_000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def _truncate_json_strings(value: Any, *, limit: int = 1_000) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, limit=limit)
    if isinstance(value, list):
        return [_truncate_json_strings(item, limit=limit) for item in value[:50]]
    if isinstance(value, dict):
        return {str(key): _truncate_json_strings(item, limit=limit) for key, item in value.items()}
    return value


def _trace_payload(value: Any, *, sanitize_context: SanitizationContext) -> Any:
    return _truncate_json_strings(sanitize_json_value(value, context=sanitize_context))


def _json_preview(value: Any, *, sanitize_context: SanitizationContext, limit: int = 500) -> str:
    sanitized = _trace_payload(value, sanitize_context=sanitize_context)
    try:
        rendered = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = str(sanitized)
    return _truncate_text(rendered, limit=limit)


def _claude_content_to_text(value: Any, *, sanitize_context: SanitizationContext) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return sanitize_text(value, context=sanitize_context)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_claude_content_to_text(item, sanitize_context=sanitize_context).strip() for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        value_type = value.get("type")
        if value_type == "text":
            return sanitize_text(str(value.get("text") or ""), context=sanitize_context)
        if value_type == "tool_reference":
            tool_name = str(value.get("tool_name") or "unknown").strip() or "unknown"
            return f"tool_reference: {tool_name}"
        if "content" in value:
            return _claude_content_to_text(value.get("content"), sanitize_context=sanitize_context)
        if "text" in value:
            return sanitize_text(str(value.get("text") or ""), context=sanitize_context)
        if "stdout" in value or "stderr" in value:
            stdout = str(value.get("stdout") or "").strip()
            stderr = str(value.get("stderr") or "").strip()
            return sanitize_text("\n".join(part for part in (stdout, stderr) if part), context=sanitize_context)
        return _json_preview(value, sanitize_context=sanitize_context, limit=4_000)
    return sanitize_text(str(value), context=sanitize_context)


def _format_claude_tool_command(tool_name: str, tool_input: Any, *, sanitize_context: SanitizationContext) -> str:
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if tool_name == "Bash" and command:
            return _truncate_text(sanitize_text(str(command), context=sanitize_context), limit=1_000)
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if file_path and tool_name in {"Read", "Edit", "Write", "Glob", "Grep"}:
            return _truncate_text(
                sanitize_text(f"{tool_name} {file_path}", context=sanitize_context),
                limit=1_000,
            )
    preview = _json_preview(tool_input, sanitize_context=sanitize_context, limit=800)
    return f"{tool_name} {preview}" if preview and preview != "null" else tool_name


def _format_codex_tool_command(tool_name: str, tool_input: Any, *, sanitize_context: SanitizationContext) -> str:
    preview = _json_preview(tool_input, sanitize_context=sanitize_context, limit=800)
    return f"{tool_name} {preview}" if preview and preview != "null" else tool_name


def _summarize_claude_tool_result(
    *,
    content: Any,
    tool_use_result: Any,
    is_error: bool,
    sanitize_context: SanitizationContext,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "isError": is_error,
        "contentChars": len(str(content or "")),
    }
    if isinstance(tool_use_result, dict):
        summary["resultKeys"] = sorted(str(key) for key in tool_use_result.keys())
    elif tool_use_result is not None:
        summary["result"] = _truncate_text(
            sanitize_text(str(tool_use_result), context=sanitize_context),
            limit=1_000,
        )
    return summary


def _normalize_reported_files(
    raw_files: object,
    *,
    workspace_path: str | None,
    candidates: set[str],
) -> list[str]:
    normalized_files: list[str] = []
    for raw_file in raw_files or []:
        normalized_file = _normalize_context_path(str(raw_file), workspace_path=workspace_path, candidates=candidates)
        if normalized_file and normalized_file not in normalized_files:
            normalized_files.append(normalized_file)
    return normalized_files


def _normalize_reported_symbols(
    raw_symbols: object,
    *,
    workspace_path: str | None,
    candidates: set[str],
) -> list[dict[str, str]]:
    normalized_entries: list[dict[str, str]] = []
    for file_path, symbol_names in normalize_symbol_map(raw_symbols).items():
        normalized_file = _normalize_context_path(file_path, workspace_path=workspace_path, candidates=candidates)
        if not normalized_file:
            continue
        for symbol_name in symbol_names:
            normalized_entries.append({"file": normalized_file, "name": symbol_name})
    return normalized_entries


def _normalize_final_output(
    final_output: dict[str, Any],
    *,
    workspace_path: str | None,
    candidates: set[str],
    sanitize_context: SanitizationContext,
) -> dict[str, Any]:
    normalized_spans: list[dict[str, Any]] = []
    for file_path, spans in normalize_span_map(final_output.get("retrieved_context_spans")).items():
        normalized_file = _normalize_context_path(file_path, workspace_path=workspace_path, candidates=candidates)
        if not normalized_file:
            continue
        for span in spans:
            normalized_spans.append({"file": normalized_file, "start": span["start"], "end": span["end"]})

    return {
        "status": final_output.get("status"),
        "finalAnswer": sanitize_text(str(final_output.get("final_answer") or "").strip(), context=sanitize_context),
        "notes": sanitize_text(str(final_output.get("notes") or "").strip(), context=sanitize_context),
        "retrievedContextFiles": _normalize_reported_files(
            final_output.get("retrieved_context_files"),
            workspace_path=workspace_path,
            candidates=candidates,
        ),
        "retrievedContextSpans": normalized_spans,
        "retrievedContextSymbols": _normalize_reported_symbols(
            final_output.get("retrieved_context_symbols"),
            workspace_path=workspace_path,
            candidates=candidates,
        ),
    }


def _extract_trace_entries(
    raw_response: dict[str, Any],
    *,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    if isinstance(raw_response.get("response"), list):
        return _extract_claude_trace_entries(raw_response, sanitize_context=sanitize_context)
    return _extract_codex_trace_entries(raw_response, sanitize_context=sanitize_context)


def _extract_codex_trace_entries(
    raw_response: dict[str, Any],
    *,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        raw_item = event.get("item")
        item = raw_item if isinstance(raw_item, dict) else {}
        item_type = str(item.get("type") or "")
        event_type = str(event.get("type") or "")

        if item_type == "command_execution" and event_type == "item.completed":
            entries.append(
                {
                    "kind": "command_execution",
                    "status": item.get("status"),
                    "command": sanitize_text(str(item.get("command") or ""), context=sanitize_context),
                    "output": _truncate_text(
                        sanitize_text(str(item.get("aggregated_output") or ""), context=sanitize_context)
                    ),
                    "exitCode": item.get("exit_code"),
                }
            )
        elif item_type == "todo_list" and event_type in {"item.updated", "item.completed"}:
            entries.append(
                {
                    "kind": "todo_list",
                    "status": item.get("status"),
                    "payload": {
                        "items": item.get("items") or item.get("todo_list") or [],
                    },
                }
            )
        elif item_type == "file_change" and event_type == "item.completed":
            entries.append(
                {
                    "kind": "file_change",
                    "status": item.get("status"),
                    "payload": sanitize_json_value(
                        {
                            key: item.get(key)
                            for key in ("path", "kind", "change_type", "description")
                            if item.get(key) is not None
                        },
                        context=sanitize_context,
                    ),
                }
            )
        elif _codex_event_is_mcp(event, item if item else None):
            tool_name = _mcp_tool_name_from_sources(item if item else None, event) or _trace_tool_name(event, item if item else None)
            tool_input = _codex_tool_input(event, item if item else None)
            result_value = _codex_tool_result(event, item if item else None)
            result_payload = _parse_jsonish_payload(result_value)
            status = str(item.get("status") or event.get("status") or "").strip() or "completed"
            is_error = not _tool_call_succeeded({"tool_name": tool_name, "payload": {**item, **event}})
            if is_error and status == "completed":
                status = "error"
            output = ""
            if result_value is not None:
                output = _json_preview(result_payload, sanitize_context=sanitize_context, limit=4_000)
            entries.append(
                {
                    "kind": "tool_use",
                    "status": status,
                    "command": _format_codex_tool_command(
                        tool_name or "mcp_tool_call",
                        tool_input,
                        sanitize_context=sanitize_context,
                    ),
                    "output": output,
                    "payload": _trace_payload(
                        {
                            "toolName": tool_name,
                            "input": tool_input,
                            "result": result_payload,
                        },
                        sanitize_context=sanitize_context,
                    ),
                }
            )
        elif item_type == "agent_message" and event_type == "item.completed":
            text = str(item.get("text") or "").strip()
            if text:
                entries.append(
                    {
                        "kind": "assistant_message",
                        "text": _truncate_text(sanitize_text(text, context=sanitize_context), limit=8_000),
                    }
                )

        if len(entries) >= 120:
            break

    return entries


def _extract_claude_trace_entries(
    raw_response: dict[str, Any],
    *,
    sanitize_context: SanitizationContext,
) -> list[dict[str, Any]]:
    response = raw_response.get("response")
    if not isinstance(response, list):
        return []

    entries: list[dict[str, Any]] = []
    pending_entry_by_tool_id: dict[str, int] = {}

    def append_entry(entry: dict[str, Any]) -> int:
        entries.append(entry)
        return len(entries) - 1

    for event in response:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        message = event.get("message")
        content_items = message.get("content") if isinstance(message, dict) else []
        if not isinstance(content_items, list):
            content_items = []

        if event_type == "assistant":
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                content_type = str(content.get("type") or "")
                if content_type == "text":
                    text = _claude_content_to_text(content, sanitize_context=sanitize_context).strip()
                    if text:
                        append_entry(
                            {
                                "kind": "assistant_message",
                                "text": _truncate_text(text, limit=8_000),
                            }
                        )
                elif content_type == "tool_use":
                    tool_id = str(content.get("id") or "").strip()
                    tool_name = str(content.get("name") or "unknown").strip() or "unknown"
                    tool_input = content.get("input") if "input" in content else {}
                    entry_index = append_entry(
                        {
                            "kind": "command_execution" if tool_name == "Bash" else "tool_use",
                            "status": "requested",
                            "command": _format_claude_tool_command(
                                tool_name,
                                tool_input,
                                sanitize_context=sanitize_context,
                            ),
                            "payload": _trace_payload(
                                {
                                    "toolName": tool_name,
                                    "toolUseId": tool_id,
                                    "input": tool_input,
                                },
                                sanitize_context=sanitize_context,
                            ),
                        }
                    )
                    if tool_id:
                        pending_entry_by_tool_id[tool_id] = entry_index
                # Claude thinking blocks can contain private reasoning/signatures; do not export them.

        elif event_type == "user":
            for content in content_items:
                if not isinstance(content, dict) or content.get("type") != "tool_result":
                    continue
                tool_use_id = str(content.get("tool_use_id") or "").strip()
                result_content = content.get("content")
                output = _claude_content_to_text(result_content, sanitize_context=sanitize_context).strip()
                is_error = bool(content.get("is_error"))
                status = "error" if is_error else "completed"
                result_summary = _summarize_claude_tool_result(
                    content=result_content,
                    tool_use_result=event.get("tool_use_result"),
                    is_error=is_error,
                    sanitize_context=sanitize_context,
                )

                entry_index = pending_entry_by_tool_id.get(tool_use_id)
                if entry_index is None:
                    append_entry(
                        {
                            "kind": "tool_result",
                            "status": status,
                            "command": f"Result for {tool_use_id}" if tool_use_id else "Tool result",
                            "output": _truncate_text(output),
                            "payload": _trace_payload(result_summary, sanitize_context=sanitize_context),
                        }
                    )
                    continue

                entry = entries[entry_index]
                entry["status"] = status
                if output:
                    entry["output"] = _truncate_text(output)
                payload = dict(entry.get("payload") or {})
                payload["result"] = result_summary
                entry["payload"] = _trace_payload(payload, sanitize_context=sanitize_context)

        if len(entries) >= 120:
            break

    return entries[:120]


def _aggregate_pattern_metrics_from_instances(instance_rows: list[dict[str, Any]]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    step_values = [int(row.get("trajectory", {}).get("steps")) for row in instance_rows if row.get("trajectory", {}).get("steps") is not None]
    if step_values:
        metrics["averageSteps"] = _format_pattern_metric(_mean(step_values))

    total_steps = sum(step_values)
    total_lines = sum(
        float(row.get("trajectory", {}).get("linesPerStep") or 0) * int(row.get("trajectory", {}).get("steps") or 0)
        for row in instance_rows
        if row.get("trajectory", {}).get("linesPerStep") is not None and row.get("trajectory", {}).get("steps") is not None
    )
    if total_steps > 0:
        metrics["avgLinesPerStep"] = _format_pattern_metric(total_lines / total_steps)

    usage_drops = [
        float(row.get("trajectory", {}).get("usageDrop"))
        for row in instance_rows
        if row.get("trajectory", {}).get("usageDrop") is not None
    ]
    if usage_drops:
        metrics["usageDrop"] = _format_metric(_mean(usage_drops))

    return metrics


def _aggregate_fix_overlap_vs_gold_from_instances(instance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for row in instance_rows:
        payload = ((row.get("fixOverlap") or {}).get("vsGold") or {})
        if not isinstance(payload, dict):
            continue
        metrics.append(
            {
                "status": payload.get("status"),
                "reason": payload.get("reason"),
                "intersection": payload.get("intersection"),
                "gold_size": payload.get("goldSize"),
                "pred_size": payload.get("predSize"),
            }
        )
    return _aggregate_overlap_metrics(metrics)


def _build_instance_payloads(
    *,
    suite_dir: Path,
    variant_manifest: dict[str, Any],
    task_rows: list[dict[str, Any]],
    gold_loader: GoldLoader | None,
    resolution_status_lookup: dict[str, str],
    artifact_suffix: str | None = None,
    include_repo_line_counts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    effective_config_path = _resolve_path(variant_manifest["effective_config_path"], suite_dir)
    pred_path = _variant_artifact_path(
        variant_manifest,
        suite_dir=suite_dir,
        stem="pred",
        artifact_suffix=artifact_suffix,
    )
    eval_path = (
        _variant_artifact_path(
            variant_manifest,
            suite_dir=suite_dir,
            stem="eval",
            artifact_suffix=artifact_suffix,
        )
        if artifact_suffix
        else _resolve_path(
            variant_manifest.get("eval_results_path") or Path(variant_manifest["output_dir"]) / "eval.jsonl",
            suite_dir,
        )
    )
    effective_config = (_read_json(effective_config_path).get("effective_config", {}) if effective_config_path.exists() else {})

    pred_rows = _read_jsonl(pred_path) if pred_path.exists() else []
    eval_rows = _read_jsonl(eval_path) if eval_path.exists() else []

    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        for key in (row.get("instance_id"), row.get("original_inst_id")):
            normalized = str(key or "").strip()
            if normalized:
                pred_by_id[normalized] = row

    eval_by_id: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        normalized = str(row.get("instance_id") or "").strip()
        if normalized:
            eval_by_id[normalized] = row

    instance_rows: list[dict[str, Any]] = []
    instance_details: dict[str, dict[str, Any]] = {}
    for task_row in task_rows:
        record_path_value = task_row.get("record_path")
        if not record_path_value:
            continue
        record_path = _resolve_path(str(record_path_value), suite_dir)
        if not record_path.exists():
            continue

        record = _read_json(record_path)
        instance_id = str(task_row.get("instance_id") or record.get("instance_id") or record.get("original_inst_id") or "").strip()
        original_instance_id = str(record.get("original_inst_id") or "").strip() or None
        if not instance_id:
          continue
        resolution_id = _normalize_resolution_instance_id(task_row, record)

        eval_row = eval_by_id.get(instance_id) or (eval_by_id.get(original_instance_id) if original_instance_id else None)
        pred_row = pred_by_id.get(instance_id) or (pred_by_id.get(original_instance_id) if original_instance_id else None)
        evaluation_status = "missing"
        if eval_row is not None:
            evaluation_status = "error" if eval_row.get("error") else "valid"
        resolution_status = resolution_status_lookup.get(resolution_id, "missing")

        status = _classify_task_row(task_row, suite_dir)
        raw_response = None
        raw_response_path_value = record.get("raw_response_path")
        if raw_response_path_value:
            raw_response_path = _resolve_path(str(raw_response_path_value), suite_dir)
            if raw_response_path.exists():
                raw_response = _read_json(raw_response_path)

        skill_counts = _extract_skill_counts(raw_response) if isinstance(raw_response, dict) else {}
        tool_counts = _extract_tool_counts(list(record.get("tool_calls") or []))
        trace_action_counts = _trace_action_counts(raw_response if isinstance(raw_response, dict) else None, record)
        cost_usd = _extract_cost_usd(raw_response) if isinstance(raw_response, dict) else None
        has_model_patch = bool(str(record.get("model_patch") or "").strip())
        has_prediction = pred_row is not None
        repository_size = _repository_size_payload(
            task_row,
            record,
            include_line_counts=include_repo_line_counts,
        )

        steps = int((eval_row.get("num_steps") if eval_row is not None else None) or 0)
        line_steps_total = 0
        workspace_path = str(record.get("workspace_path") or "")
        sanitize_context = SanitizationContext(
            repo_root=REPO_ROOT,
            suite_dir=suite_dir,
            workspace_path=Path(workspace_path) if workspace_path else None,
            task_dir=Path(str(record.get("task_dir"))) if record.get("task_dir") else None,
        )
        traj_data = (
            _filter_predicted_traj_data((pred_row.get("traj_data") or {}), workspace_path=workspace_path)
            if pred_row is not None
            else {}
        )
        retry_payload = _normalize_retry_payload(record.get("retry"), sanitize_context=sanitize_context)
        if steps <= 0:
            steps = len(traj_data.get("pred_steps") or [])
        if pred_row is not None:
            for step in traj_data.get("pred_steps") or []:
                if not isinstance(step, dict):
                    continue
                line_steps_total += _line_total(_step_line_map(step))
        lines_per_step = (line_steps_total / steps) if steps > 0 else None

        usage_drop: float | None = None
        final_output = record.get("final_output") or {}
        candidate_files = set()
        gold = None
        if gold_loader is not None:
            gold = gold_loader.get(instance_id) or (gold_loader.get(original_instance_id) if original_instance_id else None)
        if gold is not None:
            fix_overlap_vs_gold_metric = compute_patch_editloc(
                getattr(gold, "_data", {}).get("patch", "") or "",
                record.get("model_patch") or "",
            )
        else:
            fix_overlap_vs_gold_metric = _unavailable_overlap("missing_gold")
        fix_overlap_vs_gold = _fix_overlap_vs_gold_payload(fix_overlap_vs_gold_metric)

        if gold is not None and pred_row is not None:
            seen_lines: dict[str, list[tuple[int, int]]] = {}
            for step in traj_data.get("pred_steps") or []:
                if not isinstance(step, dict):
                    continue
                seen_lines = _merge_line_maps(seen_lines, _step_line_map(step))

            candidate_files = set(gold.files()) | {
                str(file_path).strip()
                for file_path in (traj_data.get("pred_files") or [])
                if str(file_path).strip()
            }
            final_lines: dict[str, list[tuple[int, int]]] = {}
            for file_path, spans in normalize_span_map(final_output.get("retrieved_context_spans")).items():
                normalized_file = _normalize_context_path(file_path, workspace_path=workspace_path, candidates=candidate_files)
                if not normalized_file:
                    continue
                for span in spans:
                    final_lines.setdefault(normalized_file, []).append((span["start"], span["end"]))
            final_lines = {file_path: _merge_line_intervals(intervals) for file_path, intervals in final_lines.items()}

            gold_lines = gold.line_spans_init()
            seen_gold_lines = _line_intersection_size(seen_lines, gold_lines)
            if seen_gold_lines > 0:
                kept_gold_lines = _line_intersection_size(final_lines, gold_lines)
                keep_ratio = min(kept_gold_lines, seen_gold_lines) / seen_gold_lines
                usage_drop = max(0.0, min(1.0, 1 - keep_ratio))
        else:
            candidate_files = {
                str(file_path).strip()
                for file_path in (traj_data.get("pred_files") or [])
                if str(file_path).strip()
            }

        mcp_usage = _extract_mcp_usage(
            raw_response if isinstance(raw_response, dict) else None,
            record,
            final_output=final_output,
            model_patch=str(record.get("model_patch") or ""),
            workspace_path=workspace_path,
            candidates=candidate_files,
            sanitize_context=sanitize_context,
        )
        mcp_summary = {key: value for key, value in mcp_usage.items() if key != "calls"}

        instance_rows.append(
            {
                "instanceId": instance_id,
                "originalInstanceId": original_instance_id,
                "bench": str(task_row.get("bench") or record.get("bench") or "Unknown"),
                "language": str(record.get("language") or "unknown"),
                "outcome": {
                    "status": status,
                },
                "artifacts": {
                    "hasModelPatch": has_model_patch,
                    "hasPrediction": has_prediction,
                    "evaluationStatus": evaluation_status,
                    "resolutionStatus": resolution_status,
                },
                "quality": {
                    granularity: {
                        "intersection": int((((eval_row or {}).get("final") or {}).get(granularity) or {}).get("intersection") or 0),
                        "goldSize": int((((eval_row or {}).get("final") or {}).get(granularity) or {}).get("gold_size") or 0),
                        "predSize": int((((eval_row or {}).get("final") or {}).get(granularity) or {}).get("pred_size") or 0),
                    }
                    for granularity in ("file", "symbol", "span", "line")
                },
                "trajectory": {
                    "efficiency": _safe_mean(
                        [
                            (((eval_row or {}).get("trajectory") or {}).get("auc_coverage") or {}).get("file"),
                            (((eval_row or {}).get("trajectory") or {}).get("auc_coverage") or {}).get("symbol"),
                            (((eval_row or {}).get("trajectory") or {}).get("auc_coverage") or {}).get("span"),
                        ],
                    ),
                    "redundancy": _safe_mean(
                        [
                            (((eval_row or {}).get("trajectory") or {}).get("redundancy") or {}).get("file"),
                            (((eval_row or {}).get("trajectory") or {}).get("redundancy") or {}).get("symbol"),
                            (((eval_row or {}).get("trajectory") or {}).get("redundancy") or {}).get("span"),
                        ],
                    ),
                    "usageDrop": usage_drop,
                    "steps": steps,
                    "linesPerStep": lines_per_step,
                },
                "fixOverlap": {
                    "vsGold": fix_overlap_vs_gold,
                },
                "resources": {
                    "durationMs": int(record.get("duration_ms") or 0),
                    "totalTokens": int((record.get("token_usage") or {}).get("total_tokens") or 0),
                    "toolCalls": trace_action_counts["toolCalls"],
                    "mcpToolCalls": trace_action_counts["mcpToolCalls"],
                    "successfulMcpToolCalls": trace_action_counts["successfulMcpToolCalls"],
                    "commandExecutions": trace_action_counts["commandExecutions"],
                    "readToolCalls": trace_action_counts["readToolCalls"],
                    "editToolCalls": trace_action_counts["editToolCalls"],
                    "costUsd": cost_usd,
                    "retryAttempts": retry_payload["attempts"],
                    "retried": retry_payload["retried"],
                    "retrySuppressed": retry_payload["suppressed"],
                },
                **({"repositorySize": repository_size} if repository_size is not None else {}),
                "skills": {
                    "totalInvocations": sum(skill_counts.values()),
                    "byType": [
                        {
                            "name": name,
                            "count": count,
                        }
                        for name, count in sorted(skill_counts.items())
                    ],
                },
                "tools": {
                    "totalInvocations": sum(tool_counts.values()),
                    "byType": [
                        {
                            "name": name,
                            "count": count,
                        }
                        for name, count in sorted(tool_counts.items())
                    ],
                },
                "mcp": mcp_summary,
            }
        )

        instance_details[instance_id] = {
            "instanceId": instance_id,
            "originalInstanceId": original_instance_id,
            "bench": str(task_row.get("bench") or record.get("bench") or "Unknown"),
            "language": str(record.get("language") or "unknown"),
            "repoUrl": record.get("repo_url"),
            "commit": record.get("commit"),
            "variant": {
                "name": _titleize(str(effective_config.get("name") or variant_manifest["name"])),
                "model": effective_config.get("model"),
                "effort": _titleize(str(effective_config.get("reasoning_effort") or "unknown")),
                "status": status,
                "evaluationStatus": evaluation_status,
                "resolutionStatus": resolution_status,
                "startedAt": record.get("started_at"),
                "completedAt": record.get("completed_at"),
                "durationMs": int(record.get("duration_ms") or 0),
                "retry": retry_payload,
                "tokenUsage": record.get("token_usage"),
                "traceCounters": trace_action_counts,
                "mcpUse": mcp_usage,
                "persistedToolResults": sanitize_json_value(
                    record.get("persisted_tool_results") or [],
                    context=sanitize_context,
                ),
                "_rawModelPatch": str(record.get("model_patch") or ""),
                "modelPatch": sanitize_text(str(record.get("model_patch") or ""), context=sanitize_context),
                "finalOutput": _normalize_final_output(
                    final_output,
                    workspace_path=workspace_path,
                    candidates=candidate_files,
                    sanitize_context=sanitize_context,
                ),
                "predTrajectory": {
                    "predSteps": sanitize_json_value(traj_data.get("pred_steps") or [], context=sanitize_context),
                    "predFiles": sanitize_json_value(traj_data.get("pred_files") or [], context=sanitize_context),
                    "predSpans": sanitize_json_value(traj_data.get("pred_spans") or {}, context=sanitize_context),
                    "predSymbols": sanitize_json_value(traj_data.get("pred_symbols") or {}, context=sanitize_context),
                },
                "evaluatedTrajectory": {
                    "steps": (((eval_row or {}).get("trajectory") or {}).get("steps") or []),
                    "aucCoverage": (((eval_row or {}).get("trajectory") or {}).get("auc_coverage") or {}),
                    "redundancy": (((eval_row or {}).get("trajectory") or {}).get("redundancy") or {}),
                },
                "fixOverlap": {
                    "vsGold": fix_overlap_vs_gold,
                },
                "traceEntries": (
                    _extract_trace_entries(raw_response, sanitize_context=sanitize_context)
                    if isinstance(raw_response, dict)
                    else []
                ),
            },
        }

    return instance_rows, instance_details


def _aggregate_pattern_metrics(
    *,
    suite_dir: Path,
    variant_manifest: dict[str, Any],
    task_rows: list[dict[str, Any]],
    gold_loader: GoldLoader | None,
    artifact_suffix: str | None = None,
) -> dict[str, str]:
    pred_path = _variant_artifact_path(
        variant_manifest,
        suite_dir=suite_dir,
        stem="pred",
        artifact_suffix=artifact_suffix,
    )
    if not pred_path.exists():
        return {}

    pred_rows = _read_jsonl(pred_path)
    if not pred_rows:
        return {}

    task_row_by_instance: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        for key in (row.get("instance_id"), row.get("original_inst_id")):
            normalized = str(key or "").strip()
            if normalized:
                task_row_by_instance[normalized] = row

    total_steps = 0
    total_lines = 0
    step_counts: list[int] = []
    usage_drops: list[float] = []

    for pred_row in pred_rows:
        traj_data = pred_row.get("traj_data") or {}
        pred_steps = traj_data.get("pred_steps") or []
        if not isinstance(pred_steps, list):
            continue

        step_count = len(pred_steps)
        step_counts.append(step_count)
        total_steps += step_count

        seen_lines: dict[str, list[tuple[int, int]]] = {}
        for step in pred_steps:
            if not isinstance(step, dict):
                continue
            step_lines = _step_line_map(step)
            total_lines += _line_total(step_lines)
            seen_lines = _merge_line_maps(seen_lines, step_lines)

        if gold_loader is None:
            continue

        instance_id = str(pred_row.get("instance_id") or pred_row.get("original_inst_id") or "").strip()
        if not instance_id:
            continue
        gold = gold_loader.get(instance_id)
        task_row = task_row_by_instance.get(instance_id)
        if gold is None or task_row is None:
            continue

        record_path_value = task_row.get("record_path")
        if not record_path_value:
            continue
        record_path = _resolve_path(str(record_path_value), suite_dir)
        if not record_path.exists():
            continue

        record = _read_json(record_path)
        final_output = record.get("final_output") or {}
        workspace_path = str(record.get("workspace_path") or "")
        candidate_files = set(gold.files()) | {
            str(file_path).strip()
            for file_path in (traj_data.get("pred_files") or [])
            if str(file_path).strip()
        }
        final_lines: dict[str, list[tuple[int, int]]] = {}
        for file_path, spans in normalize_span_map(final_output.get("retrieved_context_spans")).items():
            normalized_file = _normalize_context_path(file_path, workspace_path=workspace_path, candidates=candidate_files)
            if not normalized_file:
                continue
            for span in spans:
                final_lines.setdefault(normalized_file, []).append((span["start"], span["end"]))
        final_lines = {file_path: _merge_line_intervals(intervals) for file_path, intervals in final_lines.items()}

        gold_lines = gold.line_spans_init()
        seen_gold_lines = _line_intersection_size(seen_lines, gold_lines)
        if seen_gold_lines <= 0:
            continue
        kept_gold_lines = _line_intersection_size(final_lines, gold_lines)
        keep_ratio = min(kept_gold_lines, seen_gold_lines) / seen_gold_lines
        usage_drops.append(max(0.0, min(1.0, 1 - keep_ratio)))

    metrics: dict[str, str] = {}
    if step_counts:
        metrics["averageSteps"] = _format_pattern_metric(_mean(step_counts))
    if total_steps > 0:
        metrics["avgLinesPerStep"] = _format_pattern_metric(total_lines / total_steps)
    if usage_drops:
        metrics["usageDrop"] = _format_metric(_mean(usage_drops))
    return metrics


def _aggregate_skill_usage_from_instances(instance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not instance_rows:
        return {"totalInvocations": 0, "averageInvocationsPerRun": 0.0, "byType": []}

    total_skill_invocations = sum(int((row.get("skills") or {}).get("totalInvocations") or 0) for row in instance_rows)
    skill_totals: dict[str, int] = {}
    for row in instance_rows:
        for entry in (row.get("skills") or {}).get("byType") or []:
            name = str(entry.get("name") or "").strip()
            count = int(entry.get("count") or 0)
            if name and count:
                skill_totals[name] = skill_totals.get(name, 0) + count

    by_type = [
        {
            "name": name,
            "averagePerRun": round(count / len(instance_rows), 2),
        }
        for name, count in sorted(skill_totals.items())
    ]

    return {
        "totalInvocations": total_skill_invocations,
        "averageInvocationsPerRun": round(total_skill_invocations / len(instance_rows), 2),
        "byType": by_type,
    }


def _aggregate_tool_usage_from_instances(instance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not instance_rows:
        return {"totalInvocations": 0, "averageInvocationsPerRun": 0.0, "byType": []}

    total_tool_invocations = sum(int((row.get("tools") or {}).get("totalInvocations") or 0) for row in instance_rows)
    tool_totals: dict[str, int] = {}
    for row in instance_rows:
        for entry in (row.get("tools") or {}).get("byType") or []:
            name = str(entry.get("name") or "").strip()
            count = int(entry.get("count") or 0)
            if name and count:
                tool_totals[name] = tool_totals.get(name, 0) + count

    by_type = [
        {
            "name": name,
            "averagePerRun": round(count / len(instance_rows), 2),
        }
        for name, count in sorted(tool_totals.items())
    ]

    return {
        "totalInvocations": total_tool_invocations,
        "averageInvocationsPerRun": round(total_tool_invocations / len(instance_rows), 2),
        "byType": by_type,
    }


def _aggregate_mcp_usage_from_instances(instance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not instance_rows:
        return {
            "availableTools": [],
            "toolCalls": 0,
            "successfulToolCalls": 0,
            "callsWithResults": 0,
            "meaningfulCalls": 0,
            "callsWithFinalContextOverlap": 0,
            "callsWithPatchOverlap": 0,
            "callsWithFollowupOnReturnedPath": 0,
            "instancesWithMcpCalls": 0,
            "instancesWithMeaningfulMcpUse": 0,
            "byTool": [],
        }

    available_tools: set[str] = set()
    by_tool: dict[str, dict[str, int]] = {}
    totals = {
        "toolCalls": 0,
        "successfulToolCalls": 0,
        "callsWithResults": 0,
        "meaningfulCalls": 0,
        "callsWithFinalContextOverlap": 0,
        "callsWithPatchOverlap": 0,
        "callsWithFollowupOnReturnedPath": 0,
    }
    instances_with_mcp_calls = 0
    instances_with_meaningful_mcp_use = 0

    for row in instance_rows:
        mcp = row.get("mcp") if isinstance(row.get("mcp"), dict) else {}
        for tool in mcp.get("availableTools") or []:
            if str(tool).strip():
                available_tools.add(str(tool).strip())
        for key in totals:
            totals[key] += int(mcp.get(key) or 0)
        if int(mcp.get("toolCalls") or 0) > 0:
            instances_with_mcp_calls += 1
        if int(mcp.get("meaningfulCalls") or 0) > 0:
            instances_with_meaningful_mcp_use += 1
        for entry in mcp.get("byTool") or []:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            aggregate = by_tool.setdefault(name, {"calls": 0, "successfulCalls": 0})
            aggregate["calls"] += int(entry.get("calls") or 0)
            aggregate["successfulCalls"] += int(entry.get("successfulCalls") or 0)

    return {
        "availableTools": sorted(available_tools),
        **totals,
        "instancesWithMcpCalls": instances_with_mcp_calls,
        "instancesWithMeaningfulMcpUse": instances_with_meaningful_mcp_use,
        "byTool": [
            {
                "name": name,
                "calls": values["calls"],
                "successfulCalls": values["successfulCalls"],
            }
            for name, values in sorted(by_tool.items())
        ],
    }


def _aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    valid = [row for row in rows if "error" not in row]
    if not valid:
        raise ComparisonExportError("Evaluation file contained no valid rows")
    summary = aggregate_results(rows)

    file_cov = float(((summary.get("final_file") or {}).get("coverage")) or 0.0)
    file_prec = float(((summary.get("final_file") or {}).get("precision")) or 0.0)
    symbol_cov = float(((summary.get("final_symbol") or {}).get("coverage")) or 0.0)
    symbol_prec = float(((summary.get("final_symbol") or {}).get("precision")) or 0.0)
    span_cov = float(((summary.get("final_span") or {}).get("coverage")) or 0.0)
    span_prec = float(((summary.get("final_span") or {}).get("precision")) or 0.0)
    line_cov = float(((summary.get("final_line") or {}).get("coverage")) or 0.0)
    line_prec = float(((summary.get("final_line") or {}).get("precision")) or 0.0)
    traj_auc_file = float(summary.get("traj_auc_file") or 0.0)
    traj_auc_symbol = float(summary.get("traj_auc_symbol") or 0.0)
    traj_auc_span = float(summary.get("traj_auc_span") or 0.0)
    traj_redundancy_file = float(summary.get("traj_redundancy_file") or 0.0)
    traj_redundancy_symbol = float(summary.get("traj_redundancy_symbol") or 0.0)
    traj_redundancy_span = float(summary.get("traj_redundancy_span") or 0.0)

    file_f1 = _f1(file_cov, file_prec)
    symbol_f1 = _f1(symbol_cov, symbol_prec)
    span_f1 = _f1(span_cov, span_prec)
    line_f1 = _f1(line_cov, line_prec)
    context_recall = (file_cov + symbol_cov + span_cov) / 3
    context_precision = (file_prec + symbol_prec + span_prec) / 3
    context_f1 = (file_f1 + symbol_f1 + span_f1) / 3
    efficiency = (traj_auc_file + traj_auc_symbol + traj_auc_span) / 3
    redundancy = (traj_redundancy_file + traj_redundancy_symbol + traj_redundancy_span) / 3

    return {
        "contextF1": _format_metric(context_f1),
        "contextRecall": _format_metric(context_recall),
        "contextPrecision": _format_metric(context_precision),
        "fileF1": _format_metric(file_f1),
        "symbolF1": _format_metric(symbol_f1),
        "spanF1": _format_metric(span_f1),
        "avgLineF1": _format_metric(line_f1),
        "contextLevels": {
            "file": {
                "recall": _format_metric(file_cov),
                "precision": _format_metric(file_prec),
                "f1": _format_metric(file_f1),
            },
            "symbol": {
                "recall": _format_metric(symbol_cov),
                "precision": _format_metric(symbol_prec),
                "f1": _format_metric(symbol_f1),
            },
            "block": {
                "recall": _format_metric(span_cov),
                "precision": _format_metric(span_prec),
                "f1": _format_metric(span_f1),
            },
            "line": {
                "recall": _format_metric(line_cov),
                "precision": _format_metric(line_prec),
                "f1": _format_metric(line_f1),
            },
        },
        "efficiency": _format_metric(efficiency),
        "redundancy": _format_metric(redundancy),
    }


def _classify_task_row(row: dict[str, Any], suite_dir: Path) -> str:
    record_path_value = row.get("record_path")
    if record_path_value:
        record_path = _resolve_path(str(record_path_value), suite_dir)
        if record_path.exists():
            record = _read_json(record_path)
            if bool(record.get("timeout")):
                return "timeout"
            if "ok" in record and not bool(record.get("ok")):
                return "failed"
            record_status = str(record.get("status") or "").strip().lower()
            if record_status == "completed":
                return "completed"
            if record_status == "partial":
                return "partial"
            return "failed"

    row_status = str(row.get("status") or "").strip().lower()
    if row_status == "completed":
        return "completed"
    if row_status == "partial":
        return "partial"
    if row_status == "timeout":
        return "timeout"
    if row_status == "skipped":
        return "skipped"
    return "failed"


def _load_variant_payload(
    *,
    suite_dir: Path,
    suite_summary: dict[str, Any],
    variant_manifest: dict[str, Any],
    gold_loader: GoldLoader | None,
    artifact_suffix: str | None = None,
    include_repo_line_counts: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    effective_config_path = _resolve_path(variant_manifest["effective_config_path"], suite_dir)
    task_results_path = _resolve_path(variant_manifest["task_results_path"], suite_dir)
    eval_path = (
        _variant_artifact_path(
            variant_manifest,
            suite_dir=suite_dir,
            stem="eval",
            artifact_suffix=artifact_suffix,
        )
        if artifact_suffix
        else _resolve_path(
            variant_manifest.get("eval_results_path") or Path(variant_manifest["output_dir"]) / "eval.jsonl",
            suite_dir,
        )
    )
    resolution_summary_path = _resolve_path(
        variant_manifest.get("resolution_summary_path") or Path(variant_manifest["output_dir"]) / "resolution-summary.json",
        suite_dir,
    )

    if not effective_config_path.exists():
        raise ComparisonExportError(f"Missing effective config for variant {variant_manifest['name']}")
    if not task_results_path.exists():
        raise ComparisonExportError(f"Missing task results for variant {variant_manifest['name']}")
    if not eval_path.exists():
        raise ComparisonExportError(f"Missing {eval_path.name} for variant {variant_manifest['name']}")
    effective_wrapper = _read_json(effective_config_path)
    effective_config = effective_wrapper.get("effective_config", {})
    task_rows = _read_jsonl(task_results_path)
    eval_rows = _read_jsonl(eval_path)
    resolution_summary = (
        _read_json(resolution_summary_path)
        if resolution_summary_path.exists()
        else {
            "status": "missing",
            "pass_at_1": None,
            "pass_at_1_on_evaluated": None,
            "resolved_count": 0,
        }
    )
    resolution_partial_from_summary = _coerce_bool(suite_summary.get("resolution_is_partial")) or _coerce_bool(
        resolution_summary.get("is_partial")
    )
    official_pass_at_1 = None if resolution_partial_from_summary else _format_optional_percent(resolution_summary.get("pass_at_1"))
    official_pass_at_1_on_evaluated = _format_optional_percent(resolution_summary.get("pass_at_1_on_evaluated"))
    resolution_status_lookup = _build_resolution_status_lookup(resolution_summary)

    classified_statuses = [_classify_task_row(row, suite_dir) for row in task_rows]
    expected_tasks = int(suite_summary["total_tasks"] or len(task_rows) or 0)
    success = sum(1 for status in classified_statuses if status == "completed")
    partial = sum(1 for status in classified_statuses if status == "partial")
    skipped = sum(1 for status in classified_statuses if status == "skipped")
    timeout = sum(1 for status in classified_statuses if status == "timeout")
    failures = max(len(task_rows) - success - partial - skipped - timeout, 0) + timeout
    completed_tasks = success + partial

    instance_rows, instance_details = _build_instance_payloads(
        suite_dir=suite_dir,
        variant_manifest=variant_manifest,
        task_rows=task_rows,
        gold_loader=gold_loader,
        resolution_status_lookup=resolution_status_lookup,
        artifact_suffix=artifact_suffix,
        include_repo_line_counts=include_repo_line_counts,
    )
    patch_producing_runs = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("hasModelPatch"))
    converted_predictions = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("hasPrediction"))
    valid_evaluations = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("evaluationStatus") == "valid")
    quality = _aggregate_eval_rows(eval_rows)
    skill_usage = _aggregate_skill_usage_from_instances(instance_rows)
    tool_usage = _aggregate_tool_usage_from_instances(instance_rows)
    mcp_usage = _aggregate_mcp_usage_from_instances(instance_rows)
    pattern_metrics = _aggregate_pattern_metrics_from_instances(instance_rows)
    fix_overlap_vs_gold_summary = _format_fix_overlap_summary(
        _aggregate_fix_overlap_vs_gold_from_instances(instance_rows)
    )
    duration_values = [
        int((row.get("resources") or {}).get("durationMs") or 0)
        for row in instance_rows
        if int((row.get("resources") or {}).get("durationMs") or 0) > 0
    ]
    total_tokens = sum(int((row.get("resources") or {}).get("totalTokens") or 0) for row in instance_rows)
    tool_calls = sum(int((row.get("resources") or {}).get("toolCalls") or 0) for row in instance_rows)
    mcp_tool_calls = sum(int((row.get("resources") or {}).get("mcpToolCalls") or 0) for row in instance_rows)
    successful_mcp_tool_calls = sum(
        int((row.get("resources") or {}).get("successfulMcpToolCalls") or 0) for row in instance_rows
    )
    command_executions = sum(int((row.get("resources") or {}).get("commandExecutions") or 0) for row in instance_rows)
    read_tool_calls = sum(int((row.get("resources") or {}).get("readToolCalls") or 0) for row in instance_rows)
    edit_tool_calls = sum(int((row.get("resources") or {}).get("editToolCalls") or 0) for row in instance_rows)
    retry_attempts = sum(int((row.get("resources") or {}).get("retryAttempts") or 1) for row in instance_rows)
    retried_runs = sum(1 for row in instance_rows if (row.get("resources") or {}).get("retried"))
    retry_suppressed_runs = sum(1 for row in instance_rows if (row.get("resources") or {}).get("retrySuppressed"))
    cost_values = [
        float((row.get("resources") or {}).get("costUsd"))
        for row in instance_rows
        if (row.get("resources") or {}).get("costUsd") is not None
    ]
    cost_metric = _format_currency(_mean(cost_values)) if instance_rows and len(cost_values) == len(instance_rows) else None
    postprocess_partial = _coerce_bool(suite_summary.get("postprocess_partial"))
    conversion_partial = _coerce_bool(suite_summary.get("conversion_is_partial"))
    evaluation_partial = _coerce_bool(suite_summary.get("evaluation_is_partial"))
    resolution_partial = resolution_partial_from_summary
    warnings_text = str(suite_summary.get("warnings") or "").strip()
    variant_notes: list[str] = []
    if postprocess_partial:
        stages = []
        if conversion_partial:
            stages.append("conversion")
        if evaluation_partial:
            stages.append("evaluation")
        if resolution_partial:
            stages.append("resolution")
        stage_text = ", ".join(stages) if stages else "postprocess"
        variant_notes.append(f"{_titleize(str(variant_manifest['name']))}: partial {stage_text} coverage across selected tasks.")
    if warnings_text:
        variant_notes.append(warnings_text)

    return {
        "slug": str(variant_manifest["name"]),
        "model": str(effective_config.get("model") or "Unknown"),
        "name": _titleize(str(effective_config.get("name") or variant_manifest["name"])),
        "effort": _titleize(str(effective_config.get("reasoning_effort") or "unknown")),
        "contextF1": quality["contextF1"],
        "notes": variant_notes,
        "parameters": _setup_parameters(effective_config),
        "results": {
            "outcome": {
                "completedRuns": success,
                "partialRuns": partial,
                "failures": failures,
                "finishedRuns": completed_tasks,
                "expectedTasks": expected_tasks,
                "attemptedTasks": len(task_rows),
                "completedRunRate": _format_rate(success, expected_tasks),
                "officialPassAt1": official_pass_at_1,
                "officialPassAt1OnEvaluated": official_pass_at_1_on_evaluated,
                "officialPassAt1Status": resolution_summary.get("status"),
                "metricType": "execution_status",
                "comparableToOfficialLeaderboard": False,
            },
            "integrity": {
                "patchProducingRuns": patch_producing_runs,
                "convertedPredictions": converted_predictions,
                "validEvaluations": valid_evaluations,
                "resolvedTasks": int(resolution_summary.get("resolved_count") or 0),
                "patchProductionRate": _format_rate(patch_producing_runs, expected_tasks),
                "convertedPredictionRate": _format_rate(converted_predictions, expected_tasks),
                "validEvaluationRate": _format_rate(valid_evaluations, expected_tasks),
                "postprocessPartial": postprocess_partial,
                "conversionPartial": conversion_partial,
                "evaluationPartial": evaluation_partial,
                "resolutionPartial": resolution_partial,
                "resolutionStatus": resolution_summary.get("status"),
            },
            "quality": {
                "contextF1": quality["contextF1"],
                "contextRecall": quality["contextRecall"],
                "contextPrecision": quality["contextPrecision"],
                "fileF1": quality["fileF1"],
                "symbolF1": quality["symbolF1"],
                "spanF1": quality["spanF1"],
                "avgLineF1": quality["avgLineF1"],
                "contextLevels": quality["contextLevels"],
                "fixOverlapVsGold": fix_overlap_vs_gold_summary,
            },
            "efficiency": {
                "efficiency": quality["efficiency"],
                "redundancy": quality["redundancy"],
                "usageDrop": pattern_metrics.get("usageDrop"),
                "averageDuration": _format_duration_ms(_mean(duration_values)),
                "averageSteps": pattern_metrics.get("averageSteps"),
                "avgLinesPerStep": pattern_metrics.get("avgLinesPerStep"),
                "totalTokens": _format_tokens(total_tokens),
                "toolCalls": str(tool_calls),
                "mcpToolCalls": str(mcp_tool_calls),
                "successfulMcpToolCalls": str(successful_mcp_tool_calls),
                "commandExecutions": str(command_executions),
                "readToolCalls": str(read_tool_calls),
                "editToolCalls": str(edit_tool_calls),
                "cost": cost_metric,
            },
            "retries": {
                "totalAttempts": retry_attempts,
                "retriedRuns": retried_runs,
                "suppressedRetries": retry_suppressed_runs,
            },
            "skills": skill_usage,
            "tools": tool_usage,
            "mcp": mcp_usage,
        },
        "instances": instance_rows,
    }, instance_details


def build_comparison_export(
    suite_dir: Path,
    *,
    variant_name: str | None = None,
    artifact_suffix: str | None = None,
    include_repo_line_counts: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    experiment_path = suite_dir / "experiment.json"
    summary_path = suite_dir / "summary.json"
    manifest_path = suite_dir / "manifest.json"

    if not experiment_path.exists() or not summary_path.exists() or not manifest_path.exists():
        raise ComparisonExportError("Suite is missing experiment.json, summary.json, or manifest.json")

    experiment = _read_json(experiment_path)
    summary_rows = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    gold_path_value = ((experiment.get("postprocess") or {}).get("gold_path")
                       or (experiment.get("base_run") or {}).get("task_data"))
    gold_path = _resolve_path(str(gold_path_value), suite_dir) if gold_path_value else None
    gold_loader = GoldLoader(str(gold_path)) if gold_path and gold_path.exists() else None

    if not isinstance(summary_rows, list) or not summary_rows:
        raise ComparisonExportError("Expected at least one variant in summary.json")

    variant_manifests = manifest.get("variants") or []
    if not isinstance(variant_manifests, list) or not variant_manifests:
        raise ComparisonExportError("Expected at least one variant in manifest.json")

    summary_by_name = {str(row["variant"]): row for row in summary_rows}
    if variant_name:
        variant_manifests = [variant for variant in variant_manifests if str(variant.get("name")) == variant_name]
        if not variant_manifests:
            raise ComparisonExportError(f"Variant {variant_name} not found in manifest.json")

    ordered_variants: list[dict[str, Any]] = []
    detail_payloads: dict[str, dict[str, Any]] = {}
    for label, manifest_variant in zip(("A", "B"), variant_manifests):
        summary_variant = summary_by_name.get(str(manifest_variant["name"]))
        if not summary_variant:
            raise ComparisonExportError(f"Variant {manifest_variant['name']} missing from summary.json")
        variant_payload, variant_details = _load_variant_payload(
            suite_dir=suite_dir,
            suite_summary=summary_variant,
            variant_manifest=manifest_variant,
            gold_loader=gold_loader,
            artifact_suffix=artifact_suffix,
            include_repo_line_counts=include_repo_line_counts,
        )
        variant_payload["label"] = label
        ordered_variants.append(variant_payload)
        for instance_id, detail_row in variant_details.items():
            payload = detail_payloads.setdefault(
                instance_id,
                {
                    "comparisonId": None,  # filled below after comparison id is known
                    "instanceId": detail_row["instanceId"],
                    "originalInstanceId": detail_row.get("originalInstanceId"),
                    "bench": detail_row.get("bench"),
                    "language": detail_row.get("language"),
                    "repoUrl": detail_row.get("repoUrl"),
                    "commit": detail_row.get("commit"),
                    "variants": [],
                },
            )
            variant_detail = dict(detail_row["variant"])
            variant_detail["label"] = label
            payload["variants"].append(variant_detail)

    pair_overlap_summary: dict[str, Any] | None = None
    if len(ordered_variants) == 2:
        pair_metrics: list[dict[str, Any]] = []
        for detail_payload in detail_payloads.values():
            variants_by_label = {
                str(variant.get("label")): variant
                for variant in detail_payload.get("variants") or []
                if isinstance(variant, dict)
            }
            left_variant = variants_by_label.get("A")
            right_variant = variants_by_label.get("B")
            if left_variant is None or right_variant is None:
                continue
            metric = compute_patch_to_patch_overlap(
                str(left_variant.get("_rawModelPatch") or ""),
                str(right_variant.get("_rawModelPatch") or ""),
            )
            pair_metrics.append(metric)
            detail_payload["fixOverlapBetweenVariants"] = _fix_overlap_pair_payload(
                metric,
                left_label="A",
                right_label="B",
            )
        pair_overlap_summary = _format_pair_overlap_summary(
            _aggregate_overlap_metrics(pair_metrics),
            left_label="A",
            right_label="B",
        )

    if not ordered_variants:
        raise ComparisonExportError("No variants selected for export")

    agent = str(experiment.get("agent") or "codex")
    base_reasoning = _titleize(str((experiment.get("base_run") or {}).get("reasoning_effort") or "unknown"))
    if len(ordered_variants) == 1:
        title = ordered_variants[0]["name"]
    else:
        title = f"{ordered_variants[0]['name']} vs {ordered_variants[1]['name']}"
    top_score = max(ordered_variants, key=lambda variant: float(str(variant["contextF1"])))["contextF1"]
    task_count = int((manifest.get("task_set") or {}).get("count") or summary_rows[0].get("total_tasks") or 0)

    comparison_card = {
        "id": f"{suite_dir.name}-{ordered_variants[0]['slug']}" if len(ordered_variants) == 1 else suite_dir.name,
        "agent": agent,
        "title": title,
        "summary": str(experiment.get("description") or ""),
        "suite": str(experiment.get("experiment_name") or suite_dir.name),
        "startedAt": manifest.get("started_at"),
        "completedAt": manifest.get("completed_at"),
        "taskSet": _task_set_payload(manifest.get("task_set") or {}),
        "effort": base_reasoning,
        "tasks": task_count,
        "contextF1": top_score,
        "variants": ordered_variants,
        "notes": [],
    }
    if pair_overlap_summary is not None:
        comparison_card["fixOverlapBetweenVariants"] = pair_overlap_summary
    comparison_card["notes"].append(
        "Pass@1 is computed via the SWE-bench harness on generated patches. Completed Run Rate remains a separate fork-specific execution-status metric."
    )
    if artifact_suffix:
        comparison_card["notes"].append(
            f"Context retrieval metrics are exported from {artifact_suffix.strip().strip('.')} postprocess artifacts."
        )
    for variant in ordered_variants:
        for note in variant.get("notes") or []:
            if note not in comparison_card["notes"]:
                comparison_card["notes"].append(note)

    leaderboard_rows = [
        {
            "agent": agent,
            "model": variant["model"],
            "suite": variant["name"],
            "effort": variant["effort"],
            "tasks": task_count,
            "completedRunRate": variant["results"]["outcome"]["completedRunRate"],
            "officialPassAt1": variant["results"]["outcome"]["officialPassAt1"],
            "passAt1": variant["results"]["outcome"]["officialPassAt1"],
            "contextF1": variant["contextF1"],
        }
        for variant in ordered_variants
    ]

    payload = {
        "filterOrder": ["all", agent],
        "comparisonCards": [comparison_card],
        "leaderboardRows": leaderboard_rows,
    }
    comparison_id = comparison_card["id"]
    for detail_payload in detail_payloads.values():
        detail_payload["comparisonId"] = comparison_id
        detail_payload["variants"] = sorted(detail_payload["variants"], key=lambda variant: variant["label"])
        for variant in detail_payload["variants"]:
            variant.pop("_rawModelPatch", None)
    export_context = SanitizationContext(repo_root=REPO_ROOT, suite_dir=suite_dir)
    payload = sanitize_json_value(payload, context=export_context)
    detail_payloads = {
        instance_id: sanitize_json_value(detail_payload, context=export_context)
        for instance_id, detail_payload in detail_payloads.items()
    }
    try:
        assert_no_private_paths(payload, label="comparison payload")
        assert_no_private_paths(detail_payloads, label="comparison detail payloads")
    except ValueError as exc:
        raise ComparisonExportError(str(exc)) from exc
    return payload, detail_payloads


def build_comparison_payload(
    suite_dir: Path,
    *,
    variant_name: str | None = None,
    artifact_suffix: str | None = None,
    include_repo_line_counts: bool = False,
) -> dict[str, Any]:
    payload, _ = build_comparison_export(
        suite_dir,
        variant_name=variant_name,
        artifact_suffix=artifact_suffix,
        include_repo_line_counts=include_repo_line_counts,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one evaluated run-suite comparison for the frontend.")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=DEFAULT_SUITE_DIR,
        help=f"Run-suite directory to export. Default: {DEFAULT_SUITE_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=f"Optional single variant to export from the suite. Example default use: {DEFAULT_VARIANT}",
    )
    parser.add_argument(
        "--detail-dir",
        type=Path,
        default=DEFAULT_DETAIL_DIR,
        help=f"Directory for per-instance detail JSON files. Default: {DEFAULT_DETAIL_DIR}",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default=None,
        help="Optional postprocess artifact suffix, e.g. 'aligned' reads pred.aligned.jsonl and eval.aligned.jsonl.",
    )
    parser.add_argument(
        "--include-repo-line-counts",
        action="store_true",
        help=(
            "Include git-tracked text line counts in repository-size metadata. "
            "This fails closed when blob contents are unavailable locally and never lazy-fetches missing blobs."
        ),
    )
    return parser.parse_args()


def _write_instance_detail_files(
    *,
    detail_dir: Path,
    detail_payloads: dict[str, dict[str, Any]],
) -> None:
    for detail_payload in detail_payloads.values():
        comparison_id = str(detail_payload.get("comparisonId") or "").strip()
        instance_id = str(detail_payload.get("instanceId") or "").strip()
        if not comparison_id or not instance_id:
            continue
        target_path = detail_dir / comparison_id / f"{instance_id}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(detail_payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload, detail_payloads = build_comparison_export(
        args.suite_dir.resolve(),
        variant_name=args.variant,
        artifact_suffix=args.artifact_suffix,
        include_repo_line_counts=args.include_repo_line_counts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_instance_detail_files(detail_dir=args.detail_dir.resolve(), detail_payloads=detail_payloads)


if __name__ == "__main__":
    main()
