# SPDX-License-Identifier: Apache-2.0

"""Claude v2 OpenTelemetry-backed coding-agent adapter."""

__all__ = ["ClaudeOtelAgentParser", "ClaudeOtelAdapter", "CODING_AGENT_ADAPTER", "build_prompt"]


def __getattr__(name: str):
    if name == "ClaudeOtelAgentParser":
        from .parser import ClaudeOtelAgentParser

        return ClaudeOtelAgentParser
    if name == "ClaudeOtelAdapter":
        from .adapter import ClaudeOtelAdapter

        return ClaudeOtelAdapter
    if name == "CODING_AGENT_ADAPTER":
        from .adapter import CODING_AGENT_ADAPTER

        return CODING_AGENT_ADAPTER
    if name == "build_prompt":
        from .prompting import build_prompt

        return build_prompt
    raise AttributeError(name)
