"""Patch-derived edit-location overlap metrics."""

from __future__ import annotations

import re
from typing import Any

LineMap = dict[str, list[tuple[int, int]]]


_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _normalize_diff_path(path: str) -> str:
    value = str(path or "").strip()
    if not value or value == "/dev/null":
        return ""
    if "\t" in value:
        value = value.split("\t", 1)[0]
    if " " in value and not (value.startswith('"') and value.endswith('"')):
        value = value.split(" ", 1)[0]
    value = value.strip().strip('"')
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    while value.startswith("./"):
        value = value[2:]
    return value.lstrip("/")


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _add_line(lines_by_file: LineMap, file_path: str, line_number: int) -> None:
    if not file_path:
        return
    lines_by_file.setdefault(file_path, []).append((max(1, int(line_number)), max(1, int(line_number))))


def parse_patch_edit_locations(diff_text: str) -> LineMap:
    """Extract deterministic old-side edit locations from a unified diff.

    Deleted and replaced source lines are represented by their old-file line
    numbers. Pure insertions have no deleted source line, so each contiguous
    insertion block contributes the old-file anchor line where it was inserted.
    """
    result: LineMap = {}
    old_path = ""
    new_path = ""
    old_line = 0
    group_old_lines: list[int] = []
    group_has_add = False
    group_anchor = 0

    def location_path(*, has_old_lines: bool) -> str:
        if has_old_lines:
            return old_path or new_path
        return new_path or old_path

    def flush_group() -> None:
        nonlocal group_old_lines, group_has_add, group_anchor
        if group_old_lines:
            file_path = location_path(has_old_lines=True)
            for line_number in group_old_lines:
                _add_line(result, file_path, line_number)
        elif group_has_add:
            _add_line(result, location_path(has_old_lines=False), group_anchor)
        group_old_lines = []
        group_has_add = False
        group_anchor = 0

    for raw_line in str(diff_text or "").splitlines():
        if raw_line.startswith("diff --git "):
            flush_group()
            old_path = ""
            new_path = ""
            continue
        if raw_line.startswith("--- "):
            flush_group()
            old_path = _normalize_diff_path(raw_line[4:])
            continue
        if raw_line.startswith("+++ "):
            flush_group()
            new_path = _normalize_diff_path(raw_line[4:])
            continue
        match = _HUNK_RE.match(raw_line)
        if match:
            flush_group()
            old_line = int(match.group(1))
            continue
        if not old_path and not new_path:
            continue
        if raw_line.startswith("\\ No newline"):
            continue
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            if not group_old_lines and not group_has_add:
                group_anchor = old_line
            group_old_lines.append(old_line)
            old_line += 1
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            if not group_old_lines and not group_has_add:
                group_anchor = old_line
            group_has_add = True
            continue
        if raw_line.startswith(" "):
            flush_group()
            old_line += 1
            continue
        flush_group()

    flush_group()
    return {file_path: _merge_intervals(intervals) for file_path, intervals in result.items() if intervals}


def line_total(lines_by_file: LineMap) -> int:
    return sum(end - start + 1 for intervals in lines_by_file.values() for start, end in _merge_intervals(intervals))


def line_intersection_size(left: LineMap, right: LineMap) -> int:
    total = 0
    for file_path in set(left) & set(right):
        left_intervals = _merge_intervals(left.get(file_path, []))
        right_intervals = _merge_intervals(right.get(file_path, []))
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


def _f1(recall: float, precision: float) -> float:
    denominator = recall + precision
    return 0.0 if denominator == 0 else 2 * recall * precision / denominator


def compute_line_overlap(
    gold_locations: LineMap,
    pred_locations: LineMap,
    *,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    gold_size = line_total(gold_locations)
    pred_size = line_total(pred_locations)
    if unavailable_reason:
        return {
            "status": "unavailable",
            "reason": unavailable_reason,
            "intersection": 0,
            "gold_size": gold_size,
            "pred_size": pred_size,
            "recall": None,
            "precision": None,
            "f1": None,
        }
    intersection = line_intersection_size(gold_locations, pred_locations)
    recall = intersection / gold_size if gold_size else 0.0
    precision = intersection / pred_size if pred_size else 0.0
    return {
        "status": "available",
        "intersection": intersection,
        "gold_size": gold_size,
        "pred_size": pred_size,
        "recall": recall,
        "precision": precision,
        "f1": _f1(recall, precision),
    }


def compute_patch_editloc(gold_patch: str, model_patch: str) -> dict[str, Any]:
    """Compute patch-derived FIX-location overlap against the reference patch."""
    gold_text = str(gold_patch or "").strip()
    model_text = str(model_patch or "").strip()
    if not gold_text:
        return compute_line_overlap({}, {}, unavailable_reason="missing_gold_patch")

    gold_locations = parse_patch_edit_locations(gold_text)
    if not gold_locations:
        return compute_line_overlap(gold_locations, {}, unavailable_reason="no_gold_edit_locations")
    if not model_text:
        return compute_line_overlap(gold_locations, {}, unavailable_reason="missing_model_patch")

    model_locations = parse_patch_edit_locations(model_text)
    if not model_locations:
        return compute_line_overlap(gold_locations, model_locations, unavailable_reason="no_model_edit_locations")
    return compute_line_overlap(gold_locations, model_locations)


def compute_patch_to_patch_overlap(left_patch: str, right_patch: str) -> dict[str, Any]:
    """Compute symmetric location overlap between two model patches."""
    left_text = str(left_patch or "").strip()
    right_text = str(right_patch or "").strip()
    if not left_text:
        return compute_line_overlap({}, {}, unavailable_reason="missing_left_patch")
    if not right_text:
        left_locations = parse_patch_edit_locations(left_text)
        return compute_line_overlap(left_locations, {}, unavailable_reason="missing_right_patch")

    left_locations = parse_patch_edit_locations(left_text)
    right_locations = parse_patch_edit_locations(right_text)
    if not left_locations:
        return compute_line_overlap(left_locations, right_locations, unavailable_reason="no_left_edit_locations")
    if not right_locations:
        return compute_line_overlap(left_locations, right_locations, unavailable_reason="no_right_edit_locations")
    return compute_line_overlap(left_locations, right_locations)
