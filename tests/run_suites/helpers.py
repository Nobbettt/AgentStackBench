
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
import contextbench.run_suites_setup as run_suites_setup
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CODEX_RUNTIME_IMAGE,
)
from contextbench.run_suites_core.postprocess import (
    ResolutionCommandError,
    describe_resolution_backend_support,
    evaluate_resolution_for_suite,
    export_resolution_predictions,
    run_resolution_evaluation,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_resolution_wrapper_module(relative_path: str, module_name: str):
    module_path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task_inputs(tmp_path: Path, *, count: int = 2) -> tuple[Path, Path]:
    task_rows = []
    csv_rows = [
        "bench,instance_id,original_inst_id,language,status,patch_files,patch_blocks,patch_span,gold_context_length,num_agents,repo,commit"
    ]
    for index in range(count):
        instance_id = f"psf__requests-{1000 + index}"
        task_rows.append(
            {
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "repo_url": "https://github.com/psf/requests.git",
                "base_commit": f"abc12{index}",
                "commit": f"abc12{index}",
                "gold_ctx": [{"file": "requests/api.py", "start_line": 1, "end_line": 4}],
                "problem_statement": f"Fix bug {index}.",
                "language": "python",
            }
        )
        csv_rows.append(
            f"Verified,{instance_id},{instance_id},python,pass,1,1,1,10,1,,abc12{index}"
        )

    task_data = tmp_path / "tasks.json"
    task_csv = tmp_path / "tasks.csv"
    task_data.write_text(json.dumps(task_rows), encoding="utf-8")
    task_csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    return task_data, task_csv


def _make_fake_agent_record(
    *,
    task: dict[str, object],
    agent: str,
    task_dir: Path,
    workspace_path: Path,
    status: str = "completed",
    timeout: bool = False,
) -> dict[str, object]:
    completed = status == "completed"
    return {
        "agent": agent,
        "bench": task.get("bench"),
        "instance_id": task.get("instance_id"),
        "original_inst_id": task.get("original_inst_id"),
        "repo_url": task.get("repo_url"),
        "commit": task.get("commit") or task.get("base_commit"),
        "task_dir": str(task_dir),
        "workspace_path": str(workspace_path),
        "prompt_path": str(task_dir / "prompt.txt"),
        "started_at": "2026-03-22T00:00:00Z",
        "completed_at": "2026-03-22T00:00:01Z",
        "duration_ms": 1000,
        "timeout": timeout,
        "exit_code": 0 if completed else None,
        "signal": None,
        "ok": completed,
        "status": status,
        "final_output": {
            "task_id": task.get("instance_id"),
            "status": status,
            "final_answer": "done",
            "touched_files": ["requests/api.py"],
            "retrieval_steps": [
                {
                    "files": ["requests/api.py"],
                    "spans": {"requests/api.py": [{"start": 1, "end": 4}]},
                    "symbols": {"requests/api.py": ["request"]},
                }
            ],
            "retrieved_context_files": ["requests/api.py"],
            "retrieved_context_spans": {"requests/api.py": [{"start": 1, "end": 4}]},
            "retrieved_context_symbols": {"requests/api.py": ["request"]},
            "notes": "",
        }
        if completed
        else None,
        "token_usage": None,
        "tool_calls": [],
        "raw_response_path": None,
        "diff_path": None,
        "model_patch": "",
    }


def _fake_run_coding_agent_task(call_log: list[dict[str, object]]):
    def _run(
        *,
        task,
        agent,
        output_dir,
        cache_dir,
        schema_path,
        timeout,
        model=None,
        reasoning_effort=None,
        agent_args=(),
        env_overrides=None,
        prompt_preamble=None,
        setup=None,
        workspace_key=None,
        runtime_backend="host",
        runtime_image=None,
        runtime_env=None,
        runtime_setup_commands=(),
        runtime_keep_failed=False,
    ):
        del cache_dir, schema_path, timeout, model
        task_id = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        workspace_path = task_dir / "workspaces" / safe_path_component(workspace_key or f"{agent}-{task_id}")
        workspace_path.mkdir(parents=True, exist_ok=True)
        prompt_text = (prompt_preamble or "") + "\nFix prompt"
        (task_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        record = _make_fake_agent_record(task=task, agent=agent, task_dir=task_dir, workspace_path=workspace_path)
        suffix = "codex" if agent == "codex" else "claude"
        record_path = task_dir / f"{task_id}.{suffix}-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        call_log.append(
            {
                "task_id": task.get("instance_id"),
                "agent": agent,
                "agent_args": list(agent_args),
                "reasoning_effort": reasoning_effort,
                "env": dict(env_overrides or {}),
                "prompt_preamble": prompt_preamble,
                "setup": dict(setup or {}),
                "workspace_key": workspace_key,
                "runtime_backend": runtime_backend,
                "runtime_image": runtime_image,
                "runtime_env": dict(runtime_env or {}),
                "runtime_setup_commands": list(runtime_setup_commands or []),
                "runtime_keep_failed": runtime_keep_failed,
            }
        )
        return record

    return _run
