# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: add setup-run metadata for unscored coding-agent bootstrap prompts.

"""Typed structures for coding-agent integration records."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ServerToolUseCounts(TypedDict, total=False):
    web_search_requests: int
    web_fetch_requests: int


class TokenUsage(TypedDict):
    source: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_input_tokens: NotRequired[int]
    cache_creation_input_tokens: NotRequired[int]
    cached_input_tokens: NotRequired[int]
    reasoning_tokens: NotRequired[int]
    server_tool_use: NotRequired[ServerToolUseCounts]


class CommandResult(TypedDict):
    ok: bool
    exit_code: int | None
    signal: str | None
    timeout: bool


class ToolCall(TypedDict):
    source: str
    tool_name: str
    payload: dict[str, object]


class LineSpan(TypedDict):
    start: int
    end: int


SpanMap = dict[str, list[LineSpan]]
SymbolMap = dict[str, list[str]]


class RetrievalStep(TypedDict):
    files: list[str]
    spans: SpanMap
    symbols: SymbolMap


class FileProvenanceMap(TypedDict, total=False):
    pass


class SpanProvenanceEntry(TypedDict):
    start: int
    end: int
    source: str


SpanProvenanceMap = dict[str, list[SpanProvenanceEntry]]


class SymbolProvenanceEntry(TypedDict):
    name: str
    source: str


SymbolProvenanceMap = dict[str, list[SymbolProvenanceEntry]]


class TraceInferenceMeta(TypedDict, total=False):
    dropped_large_command_outputs: int
    dropped_large_grep_lines: int
    dropped_env_var_lines: int
    grep_match_cap_hits: int
    file_list_cap_hits: int


class StructuredOutput(TypedDict):
    task_id: str
    status: str
    final_answer: str
    touched_files: list[str]
    retrieval_steps: list[RetrievalStep]
    retrieved_context_files: list[str]
    retrieved_context_spans: SpanMap
    retrieved_context_symbols: SymbolMap
    notes: str


class TrajectoryData(TypedDict):
    pred_steps: list[RetrievalStep]
    pred_files: list[str]
    pred_spans: SpanMap
    pred_symbols: SymbolMap
    pred_files_provenance: NotRequired[dict[str, str]]
    pred_spans_provenance: NotRequired[SpanProvenanceMap]
    pred_symbols_provenance: NotRequired[SymbolProvenanceMap]
    pred_files_source: NotRequired[list[str]]
    pred_spans_source: NotRequired[list[str]]
    pred_symbols_source: NotRequired[list[str]]
    trace_inference_meta: NotRequired[TraceInferenceMeta]


class CodexRawResponse(TypedDict):
    agent: str
    response_format: str
    events: list[object]
    final_message: NotRequired[object]


class ClaudeRawResponse(TypedDict):
    agent: str
    response_format: str
    response: object | None


CodingAgentRawResponse = CodexRawResponse | ClaudeRawResponse


class SetupRunRecord(TypedDict):
    prompt_path: str
    stderr_path: str
    started_at: str
    completed_at: str
    duration_ms: int
    timeout: bool
    exit_code: int | None
    signal: str | None
    ok: bool
    status: str
    raw_response_path: str | None
    token_usage: TokenUsage | None
    tool_calls: list[ToolCall]


class TaskRecord(TypedDict):
    agent: str
    bench: object
    instance_id: object
    original_inst_id: object | None
    repo: object | None
    repo_url: object | None
    commit: object | None
    language: object | None
    workspace_path: str
    task_dir: str
    prompt_path: str
    started_at: str
    completed_at: str
    duration_ms: int
    timeout: bool
    exit_code: object
    signal: object
    ok: bool
    status: object
    final_output: StructuredOutput | None
    token_usage: TokenUsage | None
    tool_calls: list[ToolCall]
    raw_response_path: str | None
    diff_path: str | None
    model_patch: str
    setup_run: NotRequired[SetupRunRecord | None]
    runtime: NotRequired[dict[str, object]]


class LoadedTask(TypedDict):
    bench: str
    instance_id: str
    original_inst_id: str
    repo: str
    repo_url: str
    commit: str
    prompt: str
    language: str
    raw: dict[str, object]
