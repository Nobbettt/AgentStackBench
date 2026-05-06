
from __future__ import annotations

import argparse
import json
import re
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


def _truncate_text(value: str, limit: int = 4_000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


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
    entries: list[dict[str, Any]] = []
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        item = event.get("item") or {}
        if not isinstance(item, dict):
            continue
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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    effective_config_path = _resolve_path(variant_manifest["effective_config_path"], suite_dir)
    pred_path = _resolve_path(Path(variant_manifest["output_dir"]) / "pred.jsonl", suite_dir)
    eval_path = _resolve_path(
        variant_manifest.get("eval_results_path") or Path(variant_manifest["output_dir"]) / "eval.jsonl",
        suite_dir,
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
        cost_usd = _extract_cost_usd(raw_response) if isinstance(raw_response, dict) else None
        has_model_patch = bool(str(record.get("model_patch") or "").strip())
        has_prediction = pred_row is not None

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
                    "toolCalls": len(record.get("tool_calls") or []),
                    "costUsd": cost_usd,
                },
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
                "tokenUsage": record.get("token_usage"),
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
) -> dict[str, str]:
    pred_path = _resolve_path(Path(variant_manifest["output_dir"]) / "pred.jsonl", suite_dir)
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
    context_f1 = (file_f1 + symbol_f1 + span_f1) / 3
    efficiency = (traj_auc_file + traj_auc_symbol + traj_auc_span) / 3
    redundancy = (traj_redundancy_file + traj_redundancy_symbol + traj_redundancy_span) / 3

    return {
        "contextF1": _format_metric(context_f1),
        "fileF1": _format_metric(file_f1),
        "symbolF1": _format_metric(symbol_f1),
        "spanF1": _format_metric(span_f1),
        "avgLineF1": _format_metric(line_f1),
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
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    effective_config_path = _resolve_path(variant_manifest["effective_config_path"], suite_dir)
    task_results_path = _resolve_path(variant_manifest["task_results_path"], suite_dir)
    eval_path = _resolve_path(
        variant_manifest.get("eval_results_path") or Path(variant_manifest["output_dir"]) / "eval.jsonl",
        suite_dir,
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
        raise ComparisonExportError(f"Missing eval.jsonl for variant {variant_manifest['name']}")
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
    )
    patch_producing_runs = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("hasModelPatch"))
    converted_predictions = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("hasPrediction"))
    valid_evaluations = sum(1 for row in instance_rows if (row.get("artifacts") or {}).get("evaluationStatus") == "valid")
    quality = _aggregate_eval_rows(eval_rows)
    skill_usage = _aggregate_skill_usage_from_instances(instance_rows)
    tool_usage = _aggregate_tool_usage_from_instances(instance_rows)
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
                "fileF1": quality["fileF1"],
                "symbolF1": quality["symbolF1"],
                "spanF1": quality["spanF1"],
                "avgLineF1": quality["avgLineF1"],
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
                "cost": cost_metric,
            },
            "skills": skill_usage,
            "tools": tool_usage,
        },
        "instances": instance_rows,
    }, instance_details


def build_comparison_export(
    suite_dir: Path,
    *,
    variant_name: str | None = None,
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


def build_comparison_payload(suite_dir: Path, *, variant_name: str | None = None) -> dict[str, Any]:
    payload, _ = build_comparison_export(suite_dir, variant_name=variant_name)
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
    payload, detail_payloads = build_comparison_export(args.suite_dir.resolve(), variant_name=args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_instance_detail_files(detail_dir=args.detail_dir.resolve(), detail_payloads=detail_payloads)


if __name__ == "__main__":
    main()
