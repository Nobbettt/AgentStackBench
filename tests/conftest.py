
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from contextbench.coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "coding_agents"
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fixtures_root() -> Path:
    return FIXTURES_ROOT


@pytest.fixture
def make_final_output():
    def _make_final_output(
        *,
        task_id: str = "task-1",
        touched_files: list[str] | None = None,
        retrieval_steps: list[dict[str, object]] | None = None,
        retrieved_context_files: list[str] | None = None,
        retrieved_context_spans: list[dict[str, object]] | dict[str, list[dict[str, int]]] | None = None,
        retrieved_context_symbols: list[dict[str, object]] | dict[str, list[str]] | None = None,
        final_answer: str = "done",
        status: str = "completed",
        notes: str = "",
    ) -> dict[str, object]:
        return {
            "task_id": task_id,
            "status": status,
            "final_answer": final_answer,
            "touched_files": touched_files or [],
            "retrieval_steps": retrieval_steps or [],
            "retrieved_context_files": retrieved_context_files or [],
            "retrieved_context_spans": retrieved_context_spans or [],
            "retrieved_context_symbols": retrieved_context_symbols or [],
            "notes": notes,
        }

    return _make_final_output


@pytest.fixture
def make_record(make_final_output):
    def _make_record(
        *,
        agent: str = "codex",
        instance_id: str = "task-1",
        final_output: dict[str, object] | None = None,
        model_patch: str = "",
        repo_url: str = "https://github.com/example/repo.git",
        commit: str = "abc123",
    ) -> dict[str, object]:
        return {
            "agent": agent,
            "instance_id": instance_id,
            "original_inst_id": instance_id,
            "repo_url": repo_url,
            "commit": commit,
            "model_patch": model_patch,
            "final_output": final_output if final_output is not None else make_final_output(task_id=instance_id),
        }

    return _make_record


@pytest.fixture
def schema_path() -> Path:
    return CODEX_OUTPUT_SCHEMA_PATH.resolve()


@pytest.fixture
def output_schema(schema_path) -> dict[str, object]:
    return json.loads(schema_path.read_text(encoding="utf-8"))
