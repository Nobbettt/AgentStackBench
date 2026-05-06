
"""Registry helpers for coding-agent adapters."""

from __future__ import annotations

from .adapter_base import BaseCodingAgentAdapter
from .claude.adapter import CODING_AGENT_ADAPTER as CLAUDE_CODING_AGENT_ADAPTER
from .codex.adapter import CODING_AGENT_ADAPTER as CODEX_CODING_AGENT_ADAPTER

_REGISTERED_CODING_AGENT_ADAPTERS: tuple[BaseCodingAgentAdapter, ...] = (
    CODEX_CODING_AGENT_ADAPTER,
    CLAUDE_CODING_AGENT_ADAPTER,
)

_CODING_AGENT_ADAPTERS_BY_NAME: dict[str, BaseCodingAgentAdapter] = {}
for _adapter in _REGISTERED_CODING_AGENT_ADAPTERS:
    for _name in _adapter.all_names:
        _normalized = _name.strip().lower()
        if not _normalized:
            continue
        if _normalized in _CODING_AGENT_ADAPTERS_BY_NAME:
            raise RuntimeError(f"Duplicate coding-agent adapter registration for '{_normalized}'")
        _CODING_AGENT_ADAPTERS_BY_NAME[_normalized] = _adapter


def normalize_coding_agent_name(name: object) -> str | None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return None
    adapter = _CODING_AGENT_ADAPTERS_BY_NAME.get(normalized)
    if adapter is None:
        return None
    return adapter.name


def has_coding_agent_adapter(name: object) -> bool:
    return normalize_coding_agent_name(name) is not None


def get_coding_agent_adapter(name: object) -> BaseCodingAgentAdapter:
    normalized = normalize_coding_agent_name(name)
    if normalized is None:
        available = ", ".join(sorted(adapter.name for adapter in _REGISTERED_CODING_AGENT_ADAPTERS))
        raise ValueError(f"Unsupported coding agent adapter: {name!r}. Available: {available}")
    return _CODING_AGENT_ADAPTERS_BY_NAME[normalized]


def iter_coding_agent_adapters() -> tuple[BaseCodingAgentAdapter, ...]:
    return _REGISTERED_CODING_AGENT_ADAPTERS
