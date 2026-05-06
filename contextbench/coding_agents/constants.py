
"""Constants for coding-agent integrations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Default prompt-capable gold/task source for Codex and Claude runs.
DEFAULT_GOLD_PATH = REPO_ROOT / "data" / "full.parquet"
# Default task-ordering/filtering CSV used to mirror the selected upstream slice.
DEFAULT_SUBSET_CSV = REPO_ROOT / "data" / "selected_500_instances.csv"
# Default output root for coding-agent run artifacts.
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "coding_agents"
# Default local repository checkout/worktree cache.
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "repos"
# Built-in schemas used to constrain benchmark-facing structured agent answers.
CODEX_OUTPUT_SCHEMA_PATH = REPO_ROOT / "contextbench" / "agents" / "codex" / "output.schema.json"
CLAUDE_OUTPUT_SCHEMA_PATH = REPO_ROOT / "contextbench" / "agents" / "claude" / "output.schema.json"
# Pinned local Docker images for deterministic benchmark infrastructure.
DEFAULT_CODEX_RUNTIME_IMAGE = "contextbench-codex-runtime:codex-cli-0.122.0"
DEFAULT_AGENT_RUNTIME_IMAGES = {
    "codex": DEFAULT_CODEX_RUNTIME_IMAGE,
}
DEFAULT_POSTPROCESS_RUNTIME_IMAGE = "contextbench-postprocess:2026-04-23"

# Candidate task-data field names, in priority order, for the natural-language issue prompt.
DEFAULT_PROMPT_FIELDS = ["prompt", "question", "task", "problem_statement", "instruction"]
# Required keys used to recognize the benchmark summary object inside raw CLI responses.
RICH_OUTPUT_KEYS = [
    "status",
    "final_answer",
    "touched_files",
    "retrieval_steps",
    "retrieved_context_files",
    "retrieved_context_spans",
    "retrieved_context_symbols",
    "notes",
]
SEMANTIC_OUTPUT_KEYS = [
    "status",
    "final_answer",
    "retrieved_context_files",
    "retrieved_context_spans",
    "retrieved_context_symbols",
    "notes",
]
BENCH_LABEL_PREFIXES = [
    ("SWE-Bench-Verified__", "Verified"),
    ("SWE-Bench-Pro__", "Pro"),
    ("SWE-PolyBench__", "Poly"),
    ("Multi-SWE-Bench__", "Multi"),
]
