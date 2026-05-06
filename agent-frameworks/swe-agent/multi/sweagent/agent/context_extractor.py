"""
Context Extractor for MSWE-agent
Extracts file context information from agent trajectories
"""

import re
import json
from pathlib import Path
from typing import Any


class ContextExtractor:
    """Extract and save context information from agent trajectories"""
    
    def __init__(self):
        # Patterns to match file viewing commands
        self.open_pattern = re.compile(r'open\s+["\']?([^"\'>\s]+)["\']?')
        self.cat_pattern = re.compile(r'cat\s+["\']?([^"\'>\s|]+)["\']?')
        self.head_pattern = re.compile(r'head\s+(?:-n\s+\d+\s+)?["\']?([^"\'>\s|]+)["\']?')
        self.tail_pattern = re.compile(r'tail\s+(?:-n\s+\d+\s+)?["\']?([^"\'>\s|]+)["\']?')
        self.sed_pattern = re.compile(r'sed\s+.*["\']?([^"\'>\s]+)["\']?')
        self.grep_pattern = re.compile(r'grep\s+.*["\']?([^"\'>\s]+)["\']?')
        self.edit_pattern = re.compile(r'edit\s+(\d+):(\d+)')
        self.goto_pattern = re.compile(r'goto\s+(\d+)')
        self.scroll_pattern = re.compile(r'scroll_(up|down)')
        
        # Pattern to extract PATCH_CONTEXT
        self.patch_context_pattern = re.compile(r'<PATCH_CONTEXT>(.*?)</PATCH_CONTEXT>', re.DOTALL)
        
    def extract_from_trajectory(self, trajectory: list[dict[str, Any]]) -> dict:
        """Extract all context information from a trajectory"""
        
        # Files that were viewed/opened
        viewed_files = set()
        # Files that were edited
        edited_files = set()
        # Line ranges viewed per file
        file_line_ranges = {}
        # PATCH_CONTEXT if provided
        patch_context = None
        # Current open file tracking
        current_file = None
        current_line = 0
        
        for step in trajectory:
            action = step.get('action', '')
            observation = step.get('observation', '')
            response = step.get('response', '')
            
            # Extract PATCH_CONTEXT from response
            match = self.patch_context_pattern.search(response)
            if match:
                patch_context = match.group(1).strip()
            
            # Track file operations
            # Open command
            open_match = self.open_pattern.search(action)
            if open_match:
                current_file = open_match.group(1)
                viewed_files.add(current_file)
                if current_file not in file_line_ranges:
                    file_line_ranges[current_file] = []
            
            # Cat command
            cat_match = self.cat_pattern.search(action)
            if cat_match:
                viewed_files.add(cat_match.group(1))
            
            # Edit command
            edit_match = self.edit_pattern.search(action)
            if edit_match and current_file:
                start_line = int(edit_match.group(1))
                end_line = int(edit_match.group(2))
                edited_files.add(current_file)
                file_line_ranges.setdefault(current_file, []).append({
                    'start': start_line,
                    'end': end_line,
                    'type': 'edit'
                })
            
            # Goto command
            goto_match = self.goto_pattern.search(action)
            if goto_match and current_file:
                line_num = int(goto_match.group(1))
                file_line_ranges.setdefault(current_file, []).append({
                    'start': max(1, line_num - 50),
                    'end': line_num + 50,
                    'type': 'view'
                })
            
            # Extract current file from observation
            file_match = re.search(r'\[File:\s*([^\]]+)\s*\(', observation)
            if file_match:
                current_file = file_match.group(1).strip()
                viewed_files.add(current_file)
        
        # Merge overlapping line ranges
        for file_path in file_line_ranges:
            file_line_ranges[file_path] = self._merge_ranges(file_line_ranges[file_path])
        
        return {
            'viewed_files': list(viewed_files),
            'edited_files': list(edited_files),
            'file_line_ranges': file_line_ranges,
            'patch_context': patch_context,
            'patch_context_parsed': self._parse_patch_context(patch_context) if patch_context else None,
        }
    
    def _merge_ranges(self, ranges: list[dict]) -> list[dict]:
        """Merge overlapping line ranges"""
        if not ranges:
            return []
        
        # Sort by start line
        sorted_ranges = sorted(ranges, key=lambda x: x['start'])
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current['start'] <= last['end'] + 1:
                # Merge overlapping ranges
                last['end'] = max(last['end'], current['end'])
                if current.get('type') == 'edit':
                    last['type'] = 'edit'
            else:
                merged.append(current)
        
        return merged
    
    def _parse_patch_context(self, patch_context: str) -> list[dict]:
        """Parse PATCH_CONTEXT block into structured data"""
        files = []
        file_pattern = re.compile(r'File:\s*([^\n]+)')
        lines_pattern = re.compile(r'Lines:\s*(\d+)-(\d+)')
        
        current_file = None
        for line in patch_context.split('\n'):
            file_match = file_pattern.search(line)
            if file_match:
                current_file = file_match.group(1).strip()
            
            lines_match = lines_pattern.search(line)
            if lines_match and current_file:
                files.append({
                    'file': current_file,
                    'start_line': int(lines_match.group(1)),
                    'end_line': int(lines_match.group(2)),
                })
                current_file = None
        
        return files
    
    def save_context(self, context: dict, output_path: Path) -> None:
        """Save extracted context to a JSON file"""
        output_path.write_text(json.dumps(context, indent=2))


def extract_context_from_traj_file(traj_path: Path) -> dict:
    """Extract context from a trajectory file"""
    extractor = ContextExtractor()
    
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    trajectory = traj_data.get('trajectory', [])
    context = extractor.extract_from_trajectory(trajectory)
    
    # Add metadata
    context['instance_id'] = traj_data.get('environment', '')
    context['info'] = traj_data.get('info', {})
    
    return context


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        traj_path = Path(sys.argv[1])
        context = extract_context_from_traj_file(traj_path)
        print(json.dumps(context, indent=2))
