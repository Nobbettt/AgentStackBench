# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-03-16 from upstream ContextBench.
# Summary of changes: add Codex and Claude extractors to the unified agent interface.

"""Unified agent trajectory extraction interface."""

from .minisweagent import extract_trajectory as extract_miniswe
from .sweagent import extract_trajectory as extract_swe
from .agentless import extract_trajectory as extract_agentless
from .prometheus import extract_trajectory as extract_prometheus
from .openhands import extract_trajectory as extract_openhands
from .registry import get_coding_agent_adapter, has_coding_agent_adapter, iter_coding_agent_adapters, normalize_coding_agent_name


def extract_trajectory(traj_file_or_data) -> dict:
    """Auto-detect a supported trajectory format and extract unified data.

    Supported inputs include:

    - MiniSWE-agent ``.traj.json`` files
    - SWE-agent ``.checkpoints.jsonl`` files
    - Agentless ``*_traj.json`` files
    - Prometheus ``.log`` files
    - OpenHands ``output.jsonl`` files or pre-parsed dict payloads with
      ``history``
    - Coding-agent run records such as ``*.codex-record.json`` and
      ``*.claude-record.json``

    Args:
        traj_file_or_data: File path string or pre-parsed OpenHands data.

    Returns:
        Dict with ``pred_steps``, ``pred_files``, ``pred_spans``, and
        optional ``pred_symbols``.
    """
    # Handle dict input (OpenHands pre-parsed data)
    if isinstance(traj_file_or_data, dict):
        if 'history' in traj_file_or_data:
            return extract_openhands(traj_file_or_data)
        else:
            raise ValueError(f"Unsupported dict format (no 'history' field)")
    
    # Handle file path input
    traj_file = traj_file_or_data
    if (traj_file.endswith('.checkpoints.jsonl') 
        or traj_file.endswith('.context.json') 
        or traj_file.endswith('patch_context.txt')
        or traj_file.endswith('.traj')):
        return extract_swe(traj_file)
    elif traj_file.endswith('.traj.json'):
        return extract_miniswe(traj_file)
    elif traj_file.endswith('_traj.json'):
        return extract_agentless(traj_file)
    elif traj_file.endswith('output.jsonl'):
        return extract_openhands(traj_file)
    else:
        for adapter in iter_coding_agent_adapters():
            if traj_file.endswith(f".{adapter.record_suffix}-record.json"):
                return adapter.create_parser().extract_trajectory(traj_file)
        if traj_file.endswith('.log'):
            # Prometheus .log files can be very large and the context markers may not
            # appear in the first few KB. Let the Prometheus extractor decide.
            data = extract_prometheus(traj_file)
            if data.get("pred_steps") or data.get("pred_files") or data.get("pred_spans"):
                return data
            raise ValueError(f"Unsupported .log trajectory format: {traj_file}")
        raise ValueError(f"Unsupported trajectory format: {traj_file}")


__all__ = [
    'extract_trajectory',
    'get_coding_agent_adapter',
    'has_coding_agent_adapter',
    'iter_coding_agent_adapters',
    'normalize_coding_agent_name',
]
