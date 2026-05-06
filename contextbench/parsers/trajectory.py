# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: extend unified trajectory parsing for Codex and Claude records and normalize effective file context.

"""Unified trajectory parsing interface."""

from collections.abc import Iterable, Mapping

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

class Step:
    """One retrieval step."""
    def __init__(self, files=None, spans=None, symbols=None):
        self.files = files or []
        self.spans = spans or []  # [{file, start_line, end_line}]
        self.symbols = symbols or {}  # {file: [symbolName, ...]}


def _append_file(bucket: list[str], raw_path: object) -> None:
    path = str(raw_path or "").strip()
    if path and path not in bucket:
        bucket.append(path)


def _is_iterable_of_values(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def effective_file_list(*, files=None, spans=None, symbols=None) -> list[str]:
    """Return the effective file set implied by files, spans, and symbols."""
    merged: list[str] = []

    if _is_iterable_of_values(files):
        for file_path in files:
            _append_file(merged, file_path)

    if isinstance(spans, Mapping):
        for file_path in spans:
            _append_file(merged, file_path)
    elif _is_iterable_of_values(spans):
        for span in spans:
            if isinstance(span, Mapping):
                _append_file(merged, span.get("file"))

    if isinstance(symbols, Mapping):
        for file_path in symbols:
            _append_file(merged, file_path)

    return merged


def effective_step_files(step: Step | None) -> list[str]:
    """Return the effective files implied by a parsed trajectory step."""
    if step is None:
        return []
    return effective_file_list(
        files=getattr(step, "files", []),
        spans=getattr(step, "spans", []),
        symbols=getattr(step, "symbols", {}),
    )


def parse_trajectory(data: dict) -> Tuple[List[Step], Optional[Step]]:
    """Parse trajectory from unified agent data format.
    
    Args:
        data: dict with 'traj_data' containing:
            - pred_steps: list of {'files': [...], 'spans': {...}}
            - pred_files: final file list
            - pred_spans: final span dict
    
    Returns:
        (trajectory_steps, final_step)
    """
    traj_data = data.get('traj_data', {})
    
    # Convert pred_steps to Step objects
    traj_steps = []
    for step_data in traj_data.get('pred_steps', []):
        spans_dict = step_data.get('spans', {})
        symbols_dict = step_data.get('symbols', {}) or {}
        files = effective_file_list(
            files=step_data.get('files', []),
            spans=spans_dict,
            symbols=symbols_dict,
        )
        
        # Convert spans dict to list format
        spans = []
        for file_path, file_spans in spans_dict.items():
            for span in file_spans:
                spans.append({
                    'file': file_path,
                    'start_line': span['start'],
                    'end_line': span['end']
                })
        
        traj_steps.append(Step(files, spans, symbols_dict))
    
    # Build final step
    final_spans_dict = traj_data.get('pred_spans', {})
    final_symbols_dict = traj_data.get('pred_symbols', {}) or {}
    final_files = effective_file_list(
        files=traj_data.get('pred_files', []),
        spans=final_spans_dict,
        symbols=final_symbols_dict,
    )
    
    final_spans = []
    for file_path, file_spans in final_spans_dict.items():
        for span in file_spans:
            final_spans.append({
                'file': file_path,
                'start_line': span['start'],
                'end_line': span['end']
            })
    
    final_step = Step(final_files, final_spans, final_symbols_dict)
    
    return traj_steps, final_step


def load_traj_file(traj_file: str) -> dict:
    """Load trajectory file using unified agent interface."""
    # Check if it's a directory (OpenHands llm_completions format)
    if os.path.isdir(traj_file):
        from ..agents.openhands import extract_trajectory_from_llm_completions
        instance_id = os.path.basename(traj_file)
        result = extract_trajectory_from_llm_completions(traj_file)
        return {
            "instance_id": instance_id,
            "traj_data": result,
            "model_patch": ""
        }
    
    from ..agents import extract_trajectory as extract_unified
    result = extract_unified(traj_file)
    
    # Extract instance_id from filename
    basename = os.path.basename(traj_file)
    instance_id = ""
    model_patch = ""
    
    if basename.endswith('.traj.json'):
        instance_id = basename.replace('.traj.json', '')
        with open(traj_file) as f:
            data = json.load(f)
            model_patch = data.get("info", {}).get("submission", "")
    elif basename.endswith('.checkpoints.jsonl'):
        instance_id = basename.replace('.checkpoints.jsonl', '')
    elif basename.endswith('_traj.json'):
        instance_id = basename.replace('_traj.json', '')
        with open(traj_file) as f:
            data = json.load(f)
            # Prefer explicit instance_id when present
            if isinstance(data, dict) and data.get("instance_id"):
                instance_id = data.get("instance_id")
            model_patch = data.get("6_final_selected_patch", "") if isinstance(data, dict) else ""
    elif basename.endswith('.context.json'):
        instance_id = basename.replace('.context.json', '')
        # Extract patch from info.submission if available
        try:
            with open(traj_file) as f:
                data = json.load(f)
                # Extract patch from info.submission if available
                if isinstance(data, dict) and data.get("info", {}).get("submission"):
                    model_patch = data.get("info", {}).get("submission", "")
        except Exception:
            pass
    elif basename.endswith('patch_context.txt'):
        # Extract instance_id from directory structure
        # e.g., traj/sweagent/multi/multi_wjm/owner__repo-1234/owner__repo-1234/owner__repo-1234.patch_context.txt
        dir_path = os.path.dirname(traj_file)
        parent_dir = os.path.basename(dir_path)
        if parent_dir and '__' in parent_dir:
            instance_id = parent_dir
        else:
            instance_id = basename.replace('.patch_context.txt', '')
    elif basename.endswith('.traj'):
        # Extract instance_id from directory structure for extended format
        # e.g., traj/sweagent/pro/pro_extended/instance_owner__repo-hash/instance_owner__repo-hash.traj
        # Keep the instance_ prefix as it may be needed for gold matching
        dir_path = os.path.dirname(traj_file)
        parent_dir = os.path.basename(dir_path)
        if parent_dir.startswith('instance_'):
            # Keep instance_ prefix: instance_owner__repo-hash
            instance_id = parent_dir
        else:
            instance_id = basename.replace('.traj', '')
        # Extract patch from info.submission if available
        try:
            with open(traj_file) as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get("info", {}).get("submission"):
                    model_patch = data.get("info", {}).get("submission", "")
        except Exception:
            pass
    elif basename.endswith('.log'):
        instance_id = basename.replace('.log', '')
    elif basename.endswith('.codex-record.json'):
        instance_id = basename.replace('.codex-record.json', '')
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if data.get("instance_id"):
                        instance_id = data.get("instance_id")
                    model_patch = data.get("model_patch", "") or ""
        except Exception:
            pass
    elif basename.endswith('.claude-record.json'):
        instance_id = basename.replace('.claude-record.json', '')
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if data.get("instance_id"):
                        instance_id = data.get("instance_id")
                    model_patch = data.get("model_patch", "") or ""
        except Exception:
            pass
    else:
        instance_id = basename
    
    return {
        "instance_id": instance_id,
        "traj_data": result,
        "model_patch": model_patch
    }


def _is_git_lfs_pointer(path: str) -> bool:
    """Check if a file is a Git LFS pointer file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def _load_from_llm_completions_dir(llm_completions_dir: str) -> List[dict]:
    """Load trajectory data from llm_completions directory structure.
    
    Args:
        llm_completions_dir: Path to llm_completions directory containing instance subdirectories
        
    Returns:
        List of trajectory dicts, one per instance
    """
    from ..agents.openhands import extract_trajectory_from_llm_completions
    
    results = []
    llm_dir = Path(llm_completions_dir)
    
    if not llm_dir.is_dir():
        return results
    
    # Get all instance directories
    instance_dirs = [d for d in llm_dir.iterdir() if d.is_dir()]
    
    for instance_dir in sorted(instance_dirs):
        instance_id = instance_dir.name
        try:
            traj_data = extract_trajectory_from_llm_completions(str(instance_dir))
            # Try to extract model_patch from the last JSON file if available
            model_patch = ""
            json_files = sorted(instance_dir.glob("*.json"))
            if json_files:
                try:
                    with open(json_files[-1], 'r', encoding='utf-8') as f:
                        last_data = json.load(f)
                        # Look for patch in messages or response
                        messages = last_data.get('messages', [])
                        for msg in reversed(messages):
                            content = msg.get('content', '')
                            if 'git_patch' in content or 'patch' in content.lower():
                                # Try to extract patch from content
                                import re
                                patch_match = re.search(r'```(?:diff|patch)?\n(.*?)\n```', content, re.DOTALL)
                                if patch_match:
                                    model_patch = patch_match.group(1)
                                    break
                except Exception:
                    pass
            
            results.append({
                "instance_id": instance_id,
                "traj_data": traj_data,
                "model_patch": model_patch
            })
        except Exception as e:
            import sys
            print(f"  Warning: Failed to extract trajectory from {instance_id}: {e}", file=sys.stderr)
            continue
    
    return results


def load_pred(path: str) -> List[dict]:
    """Load prediction data from JSON/JSONL or trajectory files."""
    if _is_git_lfs_pointer(path):
        raise RuntimeError(f"Prediction file is a Git LFS pointer; run `git lfs pull --include '{path}'`.")
    
    # Handle trajectory files directly (.traj.json, .checkpoints.jsonl, .context.json, patch_context.txt, .traj)
    if (
        path.endswith('.traj.json')
        or path.endswith('.checkpoints.jsonl')
        or path.endswith('_traj.json')
        or path.endswith('.log')
        or path.endswith('.codex-record.json')
        or path.endswith('.claude-record.json')
        or path.endswith('.context.json')
        or path.endswith('patch_context.txt')
        or path.endswith('.traj')
    ):
        loaded = load_traj_file(path)
        return [loaded]
    
    def _load_openhands_jsonl(openhands_path: str) -> List[dict]:
        """Load OpenHands-style JSONL where each line is a trajectory dict."""
        from ..agents import extract_trajectory as extract_unified

        results = []
        with open(openhands_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance_id = data.get("instance_id") or data.get("original_inst_id") or ""
                traj_data = extract_unified(data)
                model_patch = data.get("test_result", {}).get("git_patch", "") if isinstance(data, dict) else ""
                results.append(
                    {
                        "instance_id": instance_id,
                        "traj_data": traj_data,
                        "model_patch": model_patch,
                    }
                )
        return results

    # Handle OpenHands output.jsonl (multi-instance trajectory file)
    if path.endswith("output.jsonl"):
        return _load_openhands_jsonl(path)

    # Handle OpenHands Multi benchmark JSONL (per-language files like c.jsonl).
    # These are also OpenHands-style: one instance dict per line with a `history` field.
    if path.endswith(".jsonl") and not path.endswith(".checkpoints.jsonl"):
        try:
            first_obj = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    first_obj = json.loads(line)
                    break
            if (
                isinstance(first_obj, dict)
                and "history" in first_obj
                and ("instance_id" in first_obj or "original_inst_id" in first_obj)
            ):
                return _load_openhands_jsonl(path)
        except Exception:
            # Fall back to generic JSONL loading.
            pass
    
    # Handle regular JSON/JSONL prediction files
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        obj = json.load(f)
        if isinstance(obj, list):
            return obj
        return [obj]
