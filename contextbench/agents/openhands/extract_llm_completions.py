"""Extract trajectory from OpenHands llm_completions format.

OpenHands Verified benchmark uses llm_completions directory structure:
- Each instance has its own directory
- Multiple JSON files per instance (one per LLM completion)
- Each JSON contains full conversation history up to that point
- Tool calls include str_replace_editor (view) and execute_bash

This extractor converts them into ContextBench's unified format.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_trajectory_from_llm_completions(instance_dir_path: str) -> Dict[str, Any]:
    """Extract trajectory from OpenHands llm_completions directory.
    
    Args:
        instance_dir_path: Path to instance directory containing JSON files
        
    Returns:
        Dictionary with:
        - pred_steps: List of trajectory steps with files and spans
        - pred_files: Final context files (aggregated from all steps)
        - pred_spans: Final context spans (aggregated from all steps)
    """
    instance_dir = Path(instance_dir_path)
    if not instance_dir.is_dir():
        return {"pred_steps": [], "pred_files": [], "pred_spans": {}}
    
    # Get all JSON completion files, sorted by timestamp
    json_files = sorted(instance_dir.glob("*.json"))
    if not json_files:
        return {"pred_steps": [], "pred_files": [], "pred_spans": {}}
    
    # Read the last (most complete) file
    last_file = json_files[-1]
    try:
        with open(last_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {"pred_steps": [], "pred_files": [], "pred_spans": {}}
    
    messages = data.get('messages', [])
    steps: List[Dict[str, Any]] = []
    
    for msg in messages:
        if msg.get('role') != 'assistant' or 'tool_calls' not in msg:
            continue
        
        for tool_call in msg.get('tool_calls', []):
            func_name = tool_call.get('function', {}).get('name')
            try:
                args_str = tool_call.get('function', {}).get('arguments', '{}')
                args = json.loads(args_str)
            except Exception:
                continue
            
            step_data = None
            
            # Handle str_replace_editor view command
            if func_name == 'str_replace_editor' and args.get('command') == 'view':
                step_data = _extract_from_editor_view(args)
            
            # Handle execute_bash commands that view files
            elif func_name == 'execute_bash':
                step_data = _extract_from_bash_command(args)
            
            if step_data and step_data.get('files'):
                steps.append(step_data)
    
    # Aggregate all steps into final context
    final_files, final_spans = _aggregate_steps(steps)
    
    return {
        "pred_steps": steps,
        "pred_files": final_files,
        "pred_spans": final_spans
    }


def _extract_from_editor_view(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract file and span info from str_replace_editor view command."""
    path = args.get('path', '')
    if not path:
        return None
    
    # Normalize path (remove /workspace prefix)
    norm_path = _normalize_path(path)
    if not norm_path:
        return None
    
    result = {
        'files': [norm_path],
        'spans': {}
    }
    
    # Extract view_range if present
    view_range = args.get('view_range')
    if view_range and isinstance(view_range, list) and len(view_range) >= 2:
        start_line = int(view_range[0])
        end_line = int(view_range[1])
        if start_line > 0 and end_line > 0:
            result['spans'][norm_path] = [
                {'type': 'line', 'start': start_line, 'end': end_line}
            ]
    
    return result


def _extract_from_bash_command(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract file and span info from execute_bash command."""
    command = args.get('command', '')
    if not command:
        return None
    
    # Skip non-file-viewing commands
    if any(kw in command for kw in ['git add', 'git commit', 'rm ', 'mkdir', 'echo ', 'sed -i']):
        return None
    
    views: List[Dict[str, Any]] = []
    
    # Try various command patterns
    # 1. sed -n 'start,endp' file
    for match in re.finditer(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s&|>;<]+)", command):
        start, end, file_path = int(match.group(1)), int(match.group(2)), match.group(3)
        norm_path = _normalize_path(file_path)
        if norm_path:
            views.append({'file': norm_path, 'start': start, 'end': end})
    
    # 2. head -n N file
    for match in re.finditer(r"\bhead\s+-n\s+(\d+)\s+([^\s&|>]+)", command):
        n, file_path = int(match.group(1)), match.group(2).strip("'\"")
        norm_path = _normalize_path(file_path)
        if norm_path:
            views.append({'file': norm_path, 'start': 1, 'end': n})
    
    # 3. cat file
    for match in re.finditer(r"\bcat\s+(?:-n\s+)?([^\s&|>]+)", command):
        file_path = match.group(1).strip("'\"")
        norm_path = _normalize_path(file_path)
        if norm_path and '/' in file_path:  # Only if looks like a file path
            views.append({'file': norm_path})
    
    if not views:
        return None
    
    # Build result
    files = []
    spans_by_file: Dict[str, List[Dict[str, int]]] = {}
    
    for v in views:
        file_path = v['file']
        files.append(file_path)
        
        if 'start' in v and 'end' in v:
            spans_by_file.setdefault(file_path, []).append({
                'type': 'line',
                'start': v['start'],
                'end': v['end']
            })
    
    return {
        'files': sorted(set(files)),
        'spans': spans_by_file
    }


def _normalize_path(path: str) -> str:
    """Normalize file path by removing container prefixes."""
    p = (path or "").strip().strip("'\"")
    if not p:
        return ""
    
    # Remove /workspace/ prefix
    if p.startswith("/workspace/"):
        p = p[len("/workspace/"):]
    
    # Remove leading /
    if p.startswith("/"):
        p = p.lstrip("/")
    
    # Normalize leading "./"
    if p.startswith("./"):
        p = p[2:]
    
    return p


def _aggregate_steps(steps: List[Dict[str, Any]]) -> tuple:
    """Aggregate all steps into final context (files and spans)."""
    if not steps:
        return [], {}
    
    # Collect all unique files
    all_files = set()
    for step in steps:
        all_files.update(step.get('files', []))
    
    # Collect and merge all spans per file
    spans_by_file: Dict[str, List[tuple]] = {}
    for step in steps:
        for file_path, spans in step.get('spans', {}).items():
            if file_path not in spans_by_file:
                spans_by_file[file_path] = []
            for span in spans:
                spans_by_file[file_path].append((span['start'], span['end']))
    
    # Merge overlapping spans for each file
    final_spans: Dict[str, List[Dict[str, int]]] = {}
    for file_path, intervals in spans_by_file.items():
        merged = _merge_intervals(intervals)
        final_spans[file_path] = [
            {'type': 'line', 'start': a, 'end': b}
            for a, b in merged
        ]
    
    return sorted(all_files), final_spans


def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
    """Merge overlapping or adjacent line intervals."""
    if not intervals:
        return []
    
    sorted_intervals = sorted((int(a), int(b)) for a, b in intervals)
    merged = [sorted_intervals[0]]
    
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged
