
"""Prompt construction for coding-agent integrations."""

from __future__ import annotations

from ..agents.registry import get_coding_agent_adapter

def build_prompt(task: dict[str, object], agent_name: str) -> str:
    """Dispatch to the agent-specific prompt builder."""
    return get_coding_agent_adapter(agent_name).build_prompt(task)
