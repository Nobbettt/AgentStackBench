
__all__ = ["extract_trajectory", "CodexAgentParser", "CodexAdapter", "CODING_AGENT_ADAPTER", "build_prompt"]


def __getattr__(name: str):
    if name == "extract_trajectory":
        from .extract import extract_trajectory

        return extract_trajectory
    if name == "CodexAgentParser":
        from .parser import CodexAgentParser

        return CodexAgentParser
    if name == "CodexAdapter":
        from .adapter import CodexAdapter

        return CodexAdapter
    if name == "CODING_AGENT_ADAPTER":
        from .adapter import CODING_AGENT_ADAPTER

        return CODING_AGENT_ADAPTER
    if name == "build_prompt":
        from .prompting import build_prompt

        return build_prompt
    raise AttributeError(name)
