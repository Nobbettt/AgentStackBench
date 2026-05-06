"""Extract trajectory from Prometheus answer_issue_logs `.log` format.

Prometheus logs contain repeated context blocks:
- --- BEGIN CONTEXT --- ... --- END CONTEXT ---
and one or more aggregated blocks:
- --- BEGIN AGGREGATED CONTEXT --- ... --- END AGGREGATED CONTEXT ---

This extractor converts them into ContextBench's unified format:
{
  'pred_steps': [{'files': [...], 'spans': {file: [{'type':'line','start':..,'end':..}]}}],
  'pred_files': [...],
  'pred_spans': {...}
}
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

_BEGIN_CONTEXT = "--- BEGIN CONTEXT ---"
_END_CONTEXT = "--- END CONTEXT ---"
_BEGIN_AGG = "--- BEGIN AGGREGATED CONTEXT ---"
_END_AGG = "--- END AGGREGATED CONTEXT ---"

_FILE_RE = re.compile(r"^\s*File:\s*(.+?)\s*$")
_RANGE_RE = re.compile(r"^\s*Line\s*number\s*range\s*:\s*(\d+)\s*-\s*(\d+)\s*$", re.IGNORECASE)


def _normalize_file_path(file_path: str) -> str:
    p = (file_path or "").strip().replace("\\\\", "/")
    if p.startswith("/testbed/"):
        p = p[len("/testbed/") :]
    elif p.startswith("/workspace/"):
        rest = p[len("/workspace/") :]
        parts = rest.split("/", 1)
        p = parts[1] if len(parts) == 2 else parts[0]
    if p.startswith("/"):
        p = p.lstrip("/")
    return p


def _merge_line_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted((int(a), int(b)) for a, b in intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _extract_blocks(text: str, begin: str, end: str) -> List[str]:
    if not text:
        return []
    blocks: List[str] = []
    start = 0
    while True:
        i = text.find(begin, start)
        if i < 0:
            break
        j = text.find(end, i + len(begin))
        if j < 0:
            break
        payload = text[i + len(begin) : j]
        blocks.append(payload.strip("\n"))
        start = j + len(end)
    return blocks


def _parse_file_ranges(block_text: str) -> Dict[str, List[Tuple[int, int]]]:
    spans_by_file: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None

    for raw in (block_text or "").splitlines():
        line = (raw or "").rstrip("\n")

        m = _FILE_RE.match(line)
        if m:
            current_file = _normalize_file_path(m.group(1))
            if current_file:
                spans_by_file.setdefault(current_file, [])
            else:
                current_file = None
            continue

        m = _RANGE_RE.match(line)
        if m and current_file:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            spans_by_file.setdefault(current_file, []).append((a, b))
            continue

    out: Dict[str, List[Tuple[int, int]]] = {}
    for f, ivs in spans_by_file.items():
        merged = _merge_line_intervals([iv for iv in ivs if iv and len(iv) == 2])
        if merged:
            out[f] = merged
    return out


def _to_unified_spans(ranges_by_file: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[Dict[str, int]]]:
    out: Dict[str, List[Dict[str, int]]] = {}
    for f, ivs in (ranges_by_file or {}).items():
        if not f:
            continue
        for a, b in ivs:
            out.setdefault(f, []).append({"type": "line", "start": int(a), "end": int(b)})
    return out


def extract_trajectory(log_file: str) -> Dict[str, Any]:
    if not log_file or not os.path.isfile(log_file):
        return {"pred_steps": [], "pred_files": [], "pred_spans": {}}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    step_blocks = _extract_blocks(text, _BEGIN_CONTEXT, _END_CONTEXT)
    pred_steps: List[Dict[str, Any]] = []
    for blk in step_blocks:
        ranges = _parse_file_ranges(blk)
        spans = _to_unified_spans(ranges)
        files = sorted(spans.keys())
        if files:
            pred_steps.append({"files": files, "spans": spans})

    agg_blocks = _extract_blocks(text, _BEGIN_AGG, _END_AGG)
    final_ranges = _parse_file_ranges(agg_blocks[-1]) if agg_blocks else {}
    final_spans = _to_unified_spans(final_ranges)
    final_files = sorted(final_spans.keys())

    if not final_files:
        union_spans: Dict[str, List[Tuple[int, int]]] = {}
        for step in pred_steps:
            for fpath, file_spans in (step.get("spans") or {}).items():
                for sp in file_spans:
                    union_spans.setdefault(fpath, []).append((sp.get("start", 1), sp.get("end", 1)))
        merged_union = {f: _merge_line_intervals(ivs) for f, ivs in union_spans.items()}
        final_spans = _to_unified_spans(merged_union)
        final_files = sorted(final_spans.keys())

    if not pred_steps and final_files:
        pred_steps = [{"files": final_files, "spans": final_spans}]

    return {"pred_steps": pred_steps, "pred_files": final_files, "pred_spans": final_spans}


__all__ = ["extract_trajectory"]


