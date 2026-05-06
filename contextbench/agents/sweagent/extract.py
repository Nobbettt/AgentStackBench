import json
import os
import re
from typing import Dict, Any, Optional, Tuple, List


def extract_view_command(action: str) -> Optional[Tuple[str, int, int]]:
    """Extract file path and line range from str_replace_editor view command.
    
    Returns (file_path, start_line, end_line) or None if not a view command.
    """
    if not action or 'str_replace_editor view' not in action:
        return None
    
    match = re.search(r'str_replace_editor view\s+(\S+)(?:\s+--view_range\s+(\d+)\s+(\d+))?', action)
    if not match:
        return None
    
    file_path = match.group(1)
    if match.group(2) and match.group(3):
        start_line = int(match.group(2))
        end_line = int(match.group(3))
        return (file_path, start_line, end_line)
    else:
        return (file_path, 1, -1)


def extract_content_from_observation(observation: str) -> str:
    """Extract actual content from observation, removing headers like 'Here's'."""
    if not observation:
        return ""
    
    lines = observation.split('\n')
    result_lines = []
    skip_header = False
    
    for i, line in enumerate(lines):
        lower_line = line.lower().strip()
        if i == 0 and (lower_line.startswith("here's") or lower_line.startswith("here is")):
            skip_header = True
            continue
        if skip_header and i == 1:
            skip_header = False
            continue
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()


def _normalize_file_path(file_path: str) -> str:
    """Normalize file path by removing common prefixes."""
    # Remove /testbed/ prefix
    if file_path.startswith('/testbed/'):
        file_path = file_path[9:]
    # Remove leading /
    elif file_path.startswith('/'):
        file_path = file_path[1:]
    return file_path


def parse_patch_context(patch_context_str: str) -> Dict[str, Any]:
    """Parse patch_context string format.
    
    Format:
    File: /path/to/file
    Lines: start-end
    File: /another/file
    Lines: start-end
    """
    files = {}
    lines = patch_context_str.strip().split('\n')
    current_file = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('File:'):
            file_path = line.replace('File:', '').strip()
            current_file = _normalize_file_path(file_path)
            if current_file not in files:
                files[current_file] = []
        elif line.startswith('Lines:') and current_file:
            range_str = line.replace('Lines:', '').strip()
            if '-' in range_str:
                start, end = range_str.split('-')
                files[current_file].append({
                    'type': 'line',
                    'start': int(start),
                    'end': int(end)
                })
    
    return files


def parse_context_json(context_file: str) -> Dict[str, Any]:
    """Parse context.json file format.
    
    Format:
    {
        "viewed_files": [...],
        "edited_files": [...],
        "file_line_ranges": {
            "file_path": [{"start": int, "end": int, "type": str}, ...]
        },
        "instance_id": str
    }
    """
    with open(context_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract files and spans from file_line_ranges
    files = set()
    spans = {}
    
    file_line_ranges = data.get('file_line_ranges', {})
    viewed_files = data.get('viewed_files', [])
    
    # Process file_line_ranges
    for file_path, ranges in file_line_ranges.items():
        normalized_path = _normalize_file_path(file_path)
        files.add(normalized_path)
        
        if normalized_path not in spans:
            spans[normalized_path] = []
        
        for range_info in ranges:
            if isinstance(range_info, dict):
                start = range_info.get('start', 1)
                end = range_info.get('end', 1)
                spans[normalized_path].append({
                    'type': 'line',
                    'start': start,
                    'end': end
                })
    
    # Add viewed files that don't have line ranges
    for file_path in viewed_files:
        normalized_path = _normalize_file_path(file_path)
        files.add(normalized_path)
        if normalized_path not in spans:
            spans[normalized_path] = []
    
    return {
        'files': sorted(files),
        'spans': spans
    }


def parse_patch_context_file(patch_context_file: str) -> Dict[str, Any]:
    """Parse patch_context.txt file format."""
    with open(patch_context_file, 'r', encoding='utf-8') as f:
        patch_context_str = f.read()
    
    parsed = parse_patch_context(patch_context_str)
    
    return {
        'files': sorted(parsed.keys()),
        'spans': parsed
    }


def parse_traj_file(traj_file: str) -> Dict[str, Any]:
    """Parse .traj file format (extended format with patch_context).
    
    Format:
    {
        "info": {
            "patch_context": str  # patch_context string
        },
        ...
    }
    """
    with open(traj_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract patch_context from info.patch_context
    patch_context_str = ""
    if isinstance(data, dict) and 'info' in data:
        info = data.get('info', {})
        patch_context_str = info.get('patch_context', '')
    
    if not patch_context_str:
        return {
            'files': [],
            'spans': {}
        }
    
    parsed = parse_patch_context(patch_context_str)
    
    return {
        'files': sorted(parsed.keys()),
        'spans': parsed
    }


def extract_trajectory(checkpoint_file: str) -> Dict[str, Any]:
    """Extract trajectory steps and final context from SWE-agent checkpoint file.
    
    Supports multiple formats:
    - .checkpoints.jsonl: checkpoint file format
    - .context.json: context.json file format
    - patch_context.txt: patch context text file format
    - .traj: traj file format with info.patch_context (extended format)
    
    Args:
        checkpoint_file: Path to checkpoint file (.checkpoints.jsonl, .context.json, patch_context.txt, or .traj)
        
    Returns:
        Dictionary with:
        - pred_steps: List of trajectory steps with files and spans
        - pred_files: Final context files
        - pred_spans: Final context spans
    """
    # Handle context.json files
    if checkpoint_file.endswith('.context.json'):
        context_data = parse_context_json(checkpoint_file)
        return {
            'pred_steps': [],  # No step-by-step trajectory for context.json
            'pred_files': context_data['files'],
            'pred_spans': context_data['spans']
        }
    
    # Handle patch_context.txt files
    if checkpoint_file.endswith('patch_context.txt'):
        context_data = parse_patch_context_file(checkpoint_file)
        return {
            'pred_steps': [],  # No step-by-step trajectory for patch_context.txt
            'pred_files': context_data['files'],
            'pred_spans': context_data['spans']
        }
    
    # Handle .traj files (extended format with patch_context)
    if checkpoint_file.endswith('.traj'):
        context_data = parse_traj_file(checkpoint_file)
        return {
            'pred_steps': [],  # No step-by-step trajectory for .traj files
            'pred_files': context_data['files'],
            'pred_spans': context_data['spans']
        }
    
    # Handle .checkpoints.jsonl files (original format)
    steps = []
    final_context = None
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            checkpoint = json.loads(line)
            
            if checkpoint.get('type') == 'patch_context':
                patch_context_str = checkpoint.get('patch_context', '')
                if patch_context_str:
                    parsed = parse_patch_context(patch_context_str)
                    final_context = {
                        'files': list(parsed.keys()),
                        'spans': parsed
                    }
                continue
            
            action = checkpoint.get('action', '')
            observation = checkpoint.get('observation', '')
            
            view_result = extract_view_command(action)
            if view_result:
                file_path, start_line, end_line = view_result
                
                if start_line > 0 and end_line > 0:
                    content = extract_content_from_observation(observation)
                    
                    # Normalize file path
                    normalized_path = _normalize_file_path(file_path)
                    
                    step_data = {
                        'files': [normalized_path],
                        'spans': {
                            normalized_path: [
                                {'type': 'line', 'start': start_line, 'end': end_line}
                            ]
                        }
                    }
                    
                    steps.append(step_data)
    
    result = {'pred_steps': steps}
    
    if final_context:
        result['pred_files'] = final_context['files']
        result['pred_spans'] = final_context['spans']
    elif steps:
        last_step = steps[-1]
        result['pred_files'] = last_step.get('files', [])
        result['pred_spans'] = last_step.get('spans', {})
    else:
        result['pred_files'] = []
        result['pred_spans'] = {}
    
    return result

