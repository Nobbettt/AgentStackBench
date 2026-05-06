#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract Agentless "retrieved context" into a unified format for ContextBench.

It builds three views per instance:
- file-level: predicted files
- function-level: predicted symbols (functions/classes) with optional line ranges
- span-level: predicted line spans (from edit-location)

Designed to be robust to different Agentless dumps (json/jsonl, per-folder).

Example:
  python contextbench_agentless_extract.py \
    --agentless_root /path/to/agentless_swebench_verified \
    --instance_id django__django-14631 \
    --out pred_django__django-14631.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...parsers.diff import parse_diff_lines


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Some dumps might wrap JSON in a list or have trailing commas; ignore bad lines.
                continue


def _scan_records(paths: List[str]) -> Iterable[Dict[str, Any]]:
    """Yield dict records from a list of .json/.jsonl files."""
    for p in paths:
        if not os.path.isfile(p):
            continue
        if p.endswith(".jsonl"):
            yield from _iter_jsonl(p)
        elif p.endswith(".json"):
            obj = _read_json(p)
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        yield it


def _find_files_in_dir(dir_path: str) -> List[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    pats = ["*.jsonl", "*.json", "**/*.jsonl", "**/*.json"]
    out: List[str] = []
    for pat in pats:
        out.extend(glob.glob(os.path.join(dir_path, pat), recursive=True))
    # de-dup, stable order
    seen = set()
    uniq = []
    for p in sorted(out):
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _index_by_instance(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        inst = r.get("instance_id") or r.get("inst_id") or r.get("original_inst_id")
        if not inst:
            continue
        idx.setdefault(inst, []).append(r)
    return idx


_LINE_RE = re.compile(r"^line\s*:\s*(\d+)\s*$")
_LINES_RE = re.compile(r"^lines\s*:\s*(\d+)\s*-\s*(\d+)\s*$")


def _parse_edit_loc_lines(loc_lines: List[str]) -> List[Tuple[int, int]]:
    """Parse edit-location strings into line spans.

    Agentless merged dumps often store multi-line blocks like:
      "function: Foo\\nline: 12\\nline: 15"
    We therefore extract *all* occurrences of `line:` / `lines:` inside each string.
    """
    spans: List[Tuple[int, int]] = []
    for raw in loc_lines:
        raw = (raw or "").strip()
        if not raw:
            continue
        # First try strict single-line patterns
        m = _LINES_RE.match(raw)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            spans.append((a, b))
            continue
        m = _LINE_RE.match(raw)
        if m:
            a = int(m.group(1))
            spans.append((a, a))
            continue

        # Fallback: scan inside multi-line / free-form blocks
        for line in raw.splitlines():
            line = (line or "").strip()
            if not line:
                continue
            m = _LINES_RE.match(line)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    a, b = b, a
                spans.append((a, b))
                continue
            m = _LINE_RE.match(line)
            if m:
                a = int(m.group(1))
                spans.append((a, a))
                continue
    return spans


def _norm_symbol(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^function\s*:\s*", "", s)
    s = re.sub(r"^class\s*:\s*", "", s)
    s = re.sub(r"^variable\s*:\s*", "", s)
    return s.strip()


def _extract_symbols_from_blocks(items: Any) -> List[str]:
    """Extract function/class symbols from Agentless blocks.

    `items` is usually a list of strings, where each string can be:
    - a single line like "function: foo"
    - a multi-line block with multiple symbols
    """
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not it:
            continue
        if not isinstance(it, str):
            continue
        for line in it.splitlines():
            line = (line or "").strip()
            if not line:
                continue
            if line.startswith("function:") or line.startswith("class:") or line.startswith("variable:"):
                sym = _norm_symbol(line)
                if sym:
                    out.append(sym)
    return out


@dataclass
class PredContext:
    instance_id: str
    pred_files: List[str]
    pred_symbols: Dict[str, List[str]]  # file -> list of symbols (qualnames)
    pred_spans: List[Dict[str, Any]]    # [{file,start_line,end_line,source}]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "pred_files": self.pred_files,
            "pred_symbols": self.pred_symbols,
            "pred_spans": self.pred_spans,
            "provenance": self.provenance,
        }

def _extract_related_elements_symbols(rel: Any) -> Dict[str, List[str]]:
    """Parse agentless `4_related_elements` into {file: [symbolName, ...]}.

    Input values are typically list[str], where each string may contain multiple lines like:
      "function: foo\\nclass: Bar\\nvariable: x"
    """
    out: Dict[str, List[str]] = {}
    if not isinstance(rel, dict):
        return out
    for f, items in rel.items():
        if not f:
            continue
        syms = _extract_symbols_from_blocks(items)
        if syms:
            out[f] = sorted(set(syms))
    return out


def _spans_from_patch(diff_text: str) -> Dict[str, List[Dict[str, int]]]:
    """Convert a unified diff into {file: [{type:'line', start, end}, ...]} (new-file coordinates)."""
    spans: Dict[str, List[Dict[str, int]]] = {}
    if not diff_text:
        return spans
    line_ranges = parse_diff_lines(diff_text, deletions_only=False)
    for f, intervals in (line_ranges or {}).items():
        for a, b in intervals:
            spans.setdefault(f, []).append({"type": "line", "start": int(a), "end": int(b)})
    return spans


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


def _spans_from_edit_locs(data: Dict[str, Any]) -> Dict[str, List[Dict[str, int]]]:
    """Extract span-level predictions from agentless sampled edit locations.

    We choose the first sample whose edit_locs contains any `line:`/`lines:` entries,
    then parse all line mentions for each file and merge adjacent line intervals.
    """
    out: Dict[str, List[Dict[str, int]]] = {}
    samples = data.get("5_sampled_edit_locs_and_patches") or []
    if not isinstance(samples, list):
        return out

    chosen: Optional[Dict[str, Any]] = None
    for s in samples:
        if not isinstance(s, dict):
            continue
        edit_locs = s.get("edit_locs")
        if not isinstance(edit_locs, dict):
            continue
        # Detect presence of any explicit line annotations in this sample.
        has_line = False
        for loc_lines in edit_locs.values():
            if not isinstance(loc_lines, list):
                continue
            for raw in loc_lines:
                if isinstance(raw, str) and ("line:" in raw or "lines:" in raw):
                    has_line = True
                    break
            if has_line:
                break
        if has_line:
            chosen = s
            break

    if chosen is None:
        return out

    edit_locs = chosen.get("edit_locs") or {}
    if not isinstance(edit_locs, dict):
        return out

    for f, loc_lines in edit_locs.items():
        if not f or not isinstance(loc_lines, list):
            continue
        spans = _parse_edit_loc_lines(loc_lines)  # returns list[(start,end)]
        spans = _merge_line_intervals(spans)
        if spans:
            out[f] = [{"type": "line", "start": a, "end": b} for a, b in spans]
    return out


def extract_trajectory(traj_file: str) -> Dict[str, Any]:
    """Extract trajectory from agentless `*_traj.json`.

    Returns the same unified format expected by `contextbench.parsers.trajectory`:
    {
      'pred_steps': [{'files': [...], 'spans': {...}, 'symbols': {...}}],
      'pred_files': [...],
      'pred_spans': {...},
      'pred_symbols': {...}   # optional, used by extended symbol metrics
    }
    """
    with open(traj_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}}

    pred_files = data.get("3_final_combined_files") or data.get("2_embedding_selected_files") or []
    pred_files = sorted(set(f for f in pred_files if isinstance(f, str) and f))

    pred_symbols = _extract_related_elements_symbols(data.get("4_related_elements"))
    # Span-level should come from sampled edit locations (line numbers), not from the patch.
    # The patch is reserved for EditLoc metrics only.
    pred_spans = _spans_from_edit_locs(data)

    step = {"files": pred_files, "spans": pred_spans, "symbols": pred_symbols}
    return {"pred_steps": [step], "pred_files": pred_files, "pred_spans": pred_spans, "pred_symbols": pred_symbols}


def _read_file_span_text(repo_dir: str, rel_file: str, start_line: int, end_line: int) -> str:
    """Read exact text for a line span from a checked-out repo.

    Uses splitlines(keepends=True) to preserve original newlines for char-level matching.
    Returns empty string if file missing/unreadable.
    """
    if not repo_dir or not rel_file:
        return ""
    abs_path = os.path.join(repo_dir, rel_file)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines(True)
    except OSError:
        return ""
    if not lines:
        return ""
    a = max(1, int(start_line))
    b = max(a, int(end_line))
    a0 = a - 1
    b0 = min(b, len(lines))
    if a0 >= len(lines):
        return ""
    return "".join(lines[a0:b0])


def _run_git(args: List[str], cwd: str = "", git_dir: str = "") -> None:
    cmd = ["git"]
    if git_dir:
        cmd.append(f"--git-dir={git_dir}")
    cmd.extend(args)
    subprocess.run(cmd, cwd=cwd or None, check=True)


def _safe_repo_key(repo_url: str) -> str:
    """Turn a git URL into a stable folder name."""
    s = (repo_url or "").strip()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^git@", "", s)
    s = s.replace(":", "/")
    s = s.rstrip("/")
    s = s.replace("/", "__")
    s = s.replace(".git", "")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s or "repo"


def ensure_repo_worktree(repo_url: str, commit: str, checkout_root: str) -> str:
    """Ensure a checked-out worktree at a specific commit exists; return worktree path.

    Uses a bare mirror per repo and a detached worktree per commit:
      <checkout_root>/<repo_key>.git        (bare)
      <checkout_root>/<repo_key>/<commit>/ (worktree)
    """
    if not repo_url or not commit:
        return ""
    if not checkout_root:
        return ""
    os.makedirs(checkout_root, exist_ok=True)

    repo_key = _safe_repo_key(repo_url)
    bare_dir = os.path.join(checkout_root, f"{repo_key}.git")
    wt_dir = os.path.join(checkout_root, repo_key, commit)

    if not os.path.isdir(bare_dir):
        _run_git(["clone", "--mirror", repo_url, bare_dir])
    else:
        # Keep it fresh; if offline, continue with existing objects.
        try:
            _run_git(["fetch", "--all", "--prune"], git_dir=bare_dir)
        except Exception:
            pass

    if not os.path.isdir(wt_dir):
        os.makedirs(os.path.dirname(wt_dir), exist_ok=True)
        _run_git(["worktree", "add", "--detach", wt_dir, commit], git_dir=bare_dir)
    return wt_dir


def load_eval_instance_meta(eval_root: str) -> Dict[str, Dict[str, str]]:
    """Load {instance_id: {repo_url, commit}} from eval/*/annot.json.

    IMPORTANT: annot.json is used only for repo metadata, NOT as source text.
    """
    out: Dict[str, Dict[str, str]] = {}
    if not eval_root or not os.path.isdir(eval_root):
        return out
    for p in glob.glob(os.path.join(eval_root, "**", "annot.json"), recursive=True):
        try:
            obj = _read_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        iid = obj.get("original_inst_id") or obj.get("instance_id") or obj.get("inst_id")
        repo_url = (obj.get("repo_url") or "").strip()
        commit = (obj.get("commit") or "").strip()
        if iid and repo_url and commit:
            out[iid] = {"repo_url": repo_url, "commit": commit}
    return out


def extract_agentless(
    agentless_root: str,
    instance_id: str,
    edit_mode: str = "union",
    repo_dir: str = "",
) -> PredContext:
    """Extract predicted context for one instance."""
    # --- preferred: merged loc file (already contains files, symbols, spans) ---
    merged_candidates = [
        agentless_root if agentless_root.endswith(".jsonl") else "",
        os.path.join(agentless_root, "edit_location_individual", "loc_all_merged_outputs.jsonl"),
        os.path.join(agentless_root, "loc_all_merged_outputs.jsonl"),
    ]
    # Also support loc_merged_*-outputs.jsonl (e.g. loc_merged_0-0_outputs.jsonl)
    edit_dir = os.path.join(agentless_root, "edit_location_individual")
    if os.path.isdir(edit_dir):
        for f in sorted(os.listdir(edit_dir)):
            if f.startswith("loc_merged_") and f.endswith("_outputs.jsonl"):
                merged_candidates.insert(1, os.path.join(edit_dir, f))
                break
    merged_path = next((p for p in merged_candidates if p and os.path.isfile(p)), "")

    if merged_path:
        idx = _index_by_instance(_iter_jsonl(merged_path))
        recs = idx.get(instance_id, [])
        pred_files: List[str] = []
        pred_symbols: Dict[str, List[str]] = {}
        pred_spans: List[Dict[str, Any]] = []

        for r in recs:
            pred_files.extend(r.get("found_files", []) or [])
            # related symbols
            frl = r.get("found_related_locs") or {}
            if isinstance(frl, dict):
                for f, items in frl.items():
                    if not f:
                        continue
                    syms = _extract_symbols_from_blocks(items)
                    if syms:
                        pred_symbols.setdefault(f, []).extend(syms)
            # edit spans (and occasionally contains function/class lines too)
            fel = r.get("found_edit_locs") or {}
            if isinstance(fel, dict):
                for f, loc_lines in fel.items():
                    if not f or not isinstance(loc_lines, list):
                        continue
                    # If function/class lines are embedded here, also keep them as symbols
                    extra_syms = _extract_symbols_from_blocks(loc_lines)
                    if extra_syms:
                        pred_symbols.setdefault(f, []).extend(extra_syms)
                    spans = _parse_edit_loc_lines(loc_lines)
                    for a, b in spans:
                        span_obj: Dict[str, Any] = {
                            "file": f,
                            "start_line": a,
                            "end_line": b,
                            "source": "loc_all_merged_outputs",
                        }
                        if repo_dir:
                            span_obj["text"] = _read_file_span_text(repo_dir, f, a, b)
                        pred_spans.append(span_obj)

        # normalize / de-dup
        pred_files = sorted(set(f for f in pred_files if f))
        pred_symbols = {f: sorted(set(v)) for f, v in pred_symbols.items() if f and v}

        uniq = {}
        for s in pred_spans:
            key = (s["file"], int(s["start_line"]), int(s["end_line"]))
            uniq[key] = s
        pred_spans = list(uniq.values())
        pred_spans.sort(key=lambda x: (x["file"], x["start_line"], x["end_line"]))

        prov = {
            "agentless_root": os.path.abspath(agentless_root),
            "merged_path": os.path.abspath(merged_path),
            "file_source": "loc_all_merged_outputs:found_files",
            "symbol_source": "loc_all_merged_outputs:found_related_locs(+found_edit_locs)",
            "span_source": "loc_all_merged_outputs:found_edit_locs",
            "edit_mode": edit_mode,
            "repo_dir": os.path.abspath(repo_dir) if repo_dir else "",
        }

        return PredContext(
            instance_id=instance_id,
            pred_files=pred_files,
            pred_symbols=pred_symbols,
            pred_spans=pred_spans,
            provenance=prov,
        )

    # --- files ---
    file_sources = [
        ("file_level_combined/model_used_locs.jsonl", os.path.join(agentless_root, "file_level_combined", "model_used_locs.jsonl")),
        ("file_level_combined/combined_locs.jsonl", os.path.join(agentless_root, "file_level_combined", "combined_locs.jsonl")),
    ]

    pred_files: List[str] = []
    file_source_used: Optional[str] = None
    for name, path in file_sources:
        if os.path.isfile(path):
            idx = _index_by_instance(_iter_jsonl(path))
            recs = idx.get(instance_id, [])
            if recs:
                # most dumps store a list in found_files
                files: List[str] = []
                for r in recs:
                    files.extend(r.get("found_files", []) or [])
                    files.extend(r.get("files", []) or [])
                pred_files = sorted(set(f for f in files if f))
                file_source_used = name
                break

    # fallback: scan file_level folder
    if not pred_files:
        dir_path = os.path.join(agentless_root, "file_level")
        paths = _find_files_in_dir(dir_path)
        idx = _index_by_instance(_scan_records(paths))
        recs = idx.get(instance_id, [])
        files: List[str] = []
        for r in recs:
            files.extend(r.get("found_files", []) or [])
        pred_files = sorted(set(f for f in files if f))
        if pred_files:
            file_source_used = "file_level/*"

    # --- related elements (function/class names) ---
    pred_symbols: Dict[str, List[str]] = {}
    sym_source_used: Optional[str] = None
    rel_dir = os.path.join(agentless_root, "related_elements")
    rel_paths = _find_files_in_dir(rel_dir)
    if rel_paths:
        idx = _index_by_instance(_scan_records(rel_paths))
        recs = idx.get(instance_id, [])
        if recs:
            # Agentless often uses found_related_locs: {file: ["function: ...", ...]}
            for r in recs:
                frl = r.get("found_related_locs") or r.get("related_locs") or {}
                if isinstance(frl, dict):
                    for f, locs in frl.items():
                        if not f:
                            continue
                        if not isinstance(locs, list):
                            continue
                        # Some dumps store multi-line blocks; extract per-line.
                        syms = _extract_symbols_from_blocks(locs)
                        if syms:
                            pred_symbols.setdefault(f, []).extend(syms)
            # de-dup
            pred_symbols = {f: sorted(set(v)) for f, v in pred_symbols.items()}
            if pred_symbols:
                sym_source_used = "related_elements/*"

    # --- edit spans ---
    pred_spans: List[Dict[str, Any]] = []
    span_source_used: Optional[str] = None

    # candidates (union them later)
    cand_dirs = [
        ("edit_location_samples", os.path.join(agentless_root, "edit_location_samples")),
        ("edit_location_individual", os.path.join(agentless_root, "edit_location_individual")),
    ]

    def add_from_record(r: Dict[str, Any], source: str) -> None:
        fel = r.get("found_edit_locs")
        if not fel:
            return
        # found_edit_locs can be a dict or a list of dicts (multiple samples)
        dicts: List[Dict[str, List[str]]] = []
        if isinstance(fel, dict):
            dicts = [fel]
        elif isinstance(fel, list):
            dicts = [d for d in fel if isinstance(d, dict)]
        else:
            return

        if edit_mode == "first" and dicts:
            dicts = dicts[:1]

        for d in dicts:
            for f, loc_lines in d.items():
                if not f or not isinstance(loc_lines, list):
                    continue
                spans = _parse_edit_loc_lines(loc_lines)
                for a, b in spans:
                    pred_spans.append({
                        "file": f,
                        "start_line": a,
                        "end_line": b,
                        "source": source,
                    })

    any_span = False
    for name, d in cand_dirs:
        paths = _find_files_in_dir(d)
        if not paths:
            continue
        idx = _index_by_instance(_scan_records(paths))
        recs = idx.get(instance_id, [])
        if not recs:
            continue
        for r in recs:
            add_from_record(r, f"{name}")
        if pred_spans:
            any_span = True
            span_source_used = span_source_used or name

    # de-dup spans
    uniq = {}
    for s in pred_spans:
        key = (s["file"], int(s["start_line"]), int(s["end_line"]))
        uniq[key] = s
    pred_spans = list(uniq.values())
    pred_spans.sort(key=lambda x: (x["file"], x["start_line"], x["end_line"]))

    prov = {
        "agentless_root": os.path.abspath(agentless_root),
        "file_source": file_source_used,
        "symbol_source": sym_source_used,
        "span_source": span_source_used,
        "edit_mode": edit_mode,
    }

    return PredContext(
        instance_id=instance_id,
        pred_files=pred_files,
        pred_symbols=pred_symbols,
        pred_spans=pred_spans,
        provenance=prov,
    )


def extract_all_from_merged(merged_path: str, agentless_root: str = "", edit_mode: str = "union") -> List[PredContext]:
    """Extract predicted context for all instances from a merged Agentless JSONL."""
    idx = _index_by_instance(_iter_jsonl(merged_path))
    out: List[PredContext] = []
    for inst_id in sorted(idx.keys()):
        recs = idx.get(inst_id, [])
        pred_files: List[str] = []
        pred_symbols: Dict[str, List[str]] = {}
        pred_spans: List[Dict[str, Any]] = []

        for r in recs:
            pred_files.extend(r.get("found_files", []) or [])
            frl = r.get("found_related_locs") or {}
            if isinstance(frl, dict):
                for f, items in frl.items():
                    if not f:
                        continue
                    syms = _extract_symbols_from_blocks(items)
                    if syms:
                        pred_symbols.setdefault(f, []).extend(syms)

            fel = r.get("found_edit_locs") or {}
            if isinstance(fel, dict):
                for f, loc_lines in fel.items():
                    if not f or not isinstance(loc_lines, list):
                        continue
                    extra_syms = _extract_symbols_from_blocks(loc_lines)
                    if extra_syms:
                        pred_symbols.setdefault(f, []).extend(extra_syms)
                    spans = _parse_edit_loc_lines(loc_lines)
                    for a, b in spans:
                        pred_spans.append(
                            {"file": f, "start_line": a, "end_line": b, "source": "loc_all_merged_outputs"}
                        )

        pred_files = sorted(set(f for f in pred_files if f))
        pred_symbols = {f: sorted(set(v)) for f, v in pred_symbols.items() if f and v}

        uniq = {}
        for s in pred_spans:
            key = (s["file"], int(s["start_line"]), int(s["end_line"]))
            uniq[key] = s
        pred_spans = list(uniq.values())
        pred_spans.sort(key=lambda x: (x["file"], x["start_line"], x["end_line"]))

        prov = {
            "agentless_root": os.path.abspath(agentless_root) if agentless_root else "",
            "merged_path": os.path.abspath(merged_path),
            "file_source": "loc_all_merged_outputs:found_files",
            "symbol_source": "loc_all_merged_outputs:found_related_locs(+found_edit_locs)",
            "span_source": "loc_all_merged_outputs:found_edit_locs",
            "edit_mode": edit_mode,
        }

        out.append(
            PredContext(
                instance_id=inst_id,
                pred_files=pred_files,
                pred_symbols=pred_symbols,
                pred_spans=pred_spans,
                provenance=prov,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentless_root", required=True, help="Path to agentless_* output root OR a merged .jsonl file")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--instance_id", help="Agentless instance_id (e.g., django__django-14631)")
    mx.add_argument("--all", action="store_true", help="Extract all instances (writes JSONL if --out endswith .jsonl)")
    ap.add_argument("--edit_mode", default="union", choices=["union", "first"], help="How to use multiple edit-location samples")
    ap.add_argument("--repo_dir", default="", help="Optional: checked-out repo dir; if set, attach span 'text' for char-level matching")
    ap.add_argument("--eval_root", default="", help="Optional: eval root containing annot.json; used only to find repo_url+commit per instance")
    ap.add_argument("--checkout_root", default="", help="Optional: where to create per-commit git worktrees (requires --eval_root)")
    ap.add_argument("--out", default="", help="Output JSON path (default: print to stdout)")
    args = ap.parse_args()

    if args.all:
        # Resolve merged path similarly to extract_agentless
        merged_candidates = [
            args.agentless_root if args.agentless_root.endswith(".jsonl") else "",
            os.path.join(args.agentless_root, "edit_location_individual", "loc_all_merged_outputs.jsonl"),
            os.path.join(args.agentless_root, "loc_all_merged_outputs.jsonl"),
        ]
        merged_path = next((p for p in merged_candidates if p and os.path.isfile(p)), "")
        if not merged_path:
            raise SystemExit("Cannot find loc_all_merged_outputs.jsonl under agentless_root (or agentless_root is not a .jsonl).")

        preds = extract_all_from_merged(merged_path, agentless_root=args.agentless_root, edit_mode=args.edit_mode)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            if args.out.endswith(".jsonl"):
                with open(args.out, "w", encoding="utf-8") as f:
                    for p in preds:
                        f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            else:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump([p.to_dict() for p in preds], f, ensure_ascii=False, indent=2)
        else:
            for p in preds:
                print(json.dumps(p.to_dict(), ensure_ascii=False))
        return

    repo_dir = args.repo_dir
    if not repo_dir and args.eval_root and args.checkout_root and args.instance_id:
        meta = load_eval_instance_meta(args.eval_root)
        m = meta.get(args.instance_id, {})
        if m.get("repo_url") and m.get("commit"):
            repo_dir = ensure_repo_worktree(m["repo_url"], m["commit"], args.checkout_root)

    pred = extract_agentless(args.agentless_root, args.instance_id, edit_mode=args.edit_mode, repo_dir=repo_dir)
    obj = pred.to_dict()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
