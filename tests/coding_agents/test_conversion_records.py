# Fork note: Modified by Norbert Laszlo on 2026-04-16 from upstream ContextBench.
# Summary of changes: add regression coverage for record-path resolution and provenance-source attribution.

from __future__ import annotations

import json

import pytest

from contextbench.agents import extract_trajectory
from contextbench.coding_agents import (
    convert_records_with_summary,
    convert_run_record,
    load_predictions_from_path,
    load_predictions_with_summary_from_path,
    parse_unified_diff,
    record_is_convertible,
)
from contextbench.run_suites_core.postprocess import convert_records_to_jsonl

def test_parse_unified_diff_extracts_new_file_spans() -> None:
    diff = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -10,2 +10,4 @@
 old
+new
"""

    assert parse_unified_diff(diff) == {"foo.py": [{"start": 10, "end": 13}]}

def test_convert_run_record_uses_reported_retrieval(make_final_output, make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="psf__requests-1142",
        repo_url="https://github.com/psf/requests.git",
        final_output=make_final_output(
            task_id="psf__requests-1142",
            touched_files=["requests/models.py"],
            retrieval_steps=[
                {
                    "files": ["requests/models.py"],
                    "spans": [{"file": "requests/models.py", "start": 1, "end": 20}],
                    "symbols": [{"file": "requests/models.py", "name": "Response"}],
                }
            ],
            retrieved_context_files=["requests/models.py"],
            retrieved_context_spans=[{"file": "requests/models.py", "start": 1, "end": 20}],
            retrieved_context_symbols=[{"file": "requests/models.py", "name": "Response"}],
        ),
    )

    converted = convert_run_record(record)

    assert converted["instance_id"] == "psf__requests-1142"
    assert converted["traj_data"]["pred_files"] == ["requests/models.py"]
    assert converted["traj_data"]["pred_spans"] == {"requests/models.py": [{"start": 1, "end": 20}]}
    assert converted["traj_data"]["pred_symbols"] == {"requests/models.py": ["Response"]}
    assert converted["traj_data"]["pred_files_source"] == ["agent_report"]
    assert converted["traj_data"]["pred_spans_source"] == ["agent_report"]
    assert converted["traj_data"]["pred_symbols_source"] == ["agent_report"]


def test_convert_run_record_rejects_conflicting_agent_task_id(make_final_output, make_record) -> None:
    record = make_record(
        agent="claude",
        instance_id="task-a",
        final_output=make_final_output(
            task_id="task-b",
            retrieved_context_files=["a.py"],
        ),
    )
    record["original_inst_id"] = "task-a-original"

    converted, summary = convert_records_with_summary([record], expected_agent="claude", selected_task_count=1)

    assert converted == []
    assert summary["input_error_count"] == 1
    assert summary["conversion_error_count"] == 1
    assert summary["conversion_errors"][0]["error"] == "task_id_identity_mismatch"
    assert summary["conversion_errors"][0]["reported_task_id"] == "task-b"


def test_convert_run_record_uses_runner_instance_id_when_agent_task_id_is_absent(make_final_output, make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="task-a",
        final_output=make_final_output(task_id=None, retrieved_context_files=["a.py"]),
    )
    record["original_inst_id"] = "task-a-original"

    converted = convert_run_record(record)

    assert converted["instance_id"] == "task-a"


def test_convert_run_record_ignores_patch_and_touched_files_as_retrieval(make_final_output, make_record) -> None:
    record = make_record(
        agent="claude",
        instance_id="psf__requests-1142",
        repo_url="https://github.com/psf/requests.git",
        model_patch="""diff --git a/requests/api.py b/requests/api.py
--- a/requests/api.py
+++ b/requests/api.py
@@ -5,0 +5,3 @@
+x
""",
        final_output=make_final_output(
            task_id="psf__requests-1142",
            touched_files=["requests/api.py"],
        ),
    )

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == []
    assert converted["traj_data"]["pred_spans"] == {}
    assert converted["traj_data"]["pred_files_source"] == []
    assert converted["traj_data"]["pred_spans_source"] == []


def test_convert_run_record_does_not_fallback_to_diff_path_for_model_patch(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    diff_path = tmp_path / "workspace.diff"
    diff_path.write_text(
        """diff --git a/requests/api.py b/requests/api.py
--- a/requests/api.py
+++ b/requests/api.py
@@ -1 +1 @@
-x
+y
""",
        encoding="utf-8",
    )
    record = make_record(
        agent="codex",
        instance_id="psf__requests-1142",
        repo_url="https://github.com/psf/requests.git",
        model_patch="",
        final_output=make_final_output(
            task_id="psf__requests-1142",
            retrieved_context_files=["requests/models.py"],
        ),
    )
    record["diff_path"] = str(diff_path)

    converted = convert_run_record(record)

    assert converted["model_patch"] == ""


def test_convert_run_record_accepts_minimal_final_output() -> None:
    record = {
        "agent": "codex",
        "instance_id": "psf__requests-1142",
        "repo_url": "https://github.com/psf/requests.git",
        "workspace_path": "/tmp/workspace",
        "model_patch": """diff --git a/requests/api.py b/requests/api.py
--- a/requests/api.py
+++ b/requests/api.py
@@ -5,0 +5,3 @@
+x
""",
        "final_output": {
            "task_id": "psf__requests-1142",
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": ["requests/models.py"],
            "retrieved_context_spans": [{"file": "requests/models.py", "start": 1, "end": 20}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["requests/models.py"]
    assert converted["traj_data"]["pred_spans"]["requests/models.py"][0]["start"] == 1
    assert converted["traj_data"]["pred_steps"] == []


def test_convert_run_record_normalizes_absolute_paths_under_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": str(workspace),
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "model_patch": "",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "retrieved_context_files": [str(workspace / "pkg" / "mod.py")],
            "retrieved_context_spans": [{"file": str(workspace / "pkg" / "mod.py"), "start": 1, "end": 3}],
            "retrieved_context_symbols": [{"file": str(workspace / "pkg" / "mod.py"), "name": "Thing"}],
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["pkg/mod.py"]
    assert converted["traj_data"]["pred_spans"] == {"pkg/mod.py": [{"start": 1, "end": 3}]}
    assert converted["traj_data"]["pred_symbols"] == {"pkg/mod.py": ["Thing"]}


def test_convert_records_with_summary_fails_outside_workspace_context(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": str(workspace),
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "model_patch": "",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "retrieved_context_files": [str(tmp_path / "home" / ".agents" / "skills" / "superpowers" / "SKILL.md")],
        },
    }

    predictions, summary = convert_records_with_summary([record], expected_agent="codex")

    assert predictions == []
    assert summary["input_error_count"] == 1
    assert summary["conversion_error_count"] == 1
    assert summary["conversion_errors"][0]["error"] == "invalid_predicted_context_path"
    assert "SKILL.md" in summary["conversion_errors"][0]["invalid_paths"][0]


def test_convert_run_record_drops_trace_inferred_paths_outside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    skill_path = tmp_path / "runtime-home" / ".agents" / "skills" / "superpowers" / "SKILL.md"

    class Parser:
        def load_raw_response(self, record):
            return {"events": []}

        def infer_trajectory_data(self, raw_response, *, record):
            return {
                "pred_steps": [{"files": [str(skill_path)], "spans": {}, "symbols": {}}],
                "pred_files": [str(skill_path), str(workspace / "pkg" / "mod.py")],
                "pred_spans": {},
                "pred_symbols": {},
            }

    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": str(workspace),
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "model_patch": "",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
        },
    }

    converted = convert_run_record(record, parser=Parser())

    assert converted["traj_data"]["pred_files"] == ["pkg/mod.py"]

def test_convert_run_record_merges_inferred_and_reported_retrieval() -> None:
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "raw_response": {
            "agent": "codex",
            "response_format": "jsonl-events",
            "events": [
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/zsh -lc 'rg -n \"fill_value\" sklearn/impute/_iterative.py'",
                        "aggregated_output": "sklearn/impute/_iterative.py:120:    fill_value : str or numerical value, default=None\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ],
        },
        "model_patch": "",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "touched_files": [],
            "retrieval_steps": [
                {
                    "files": ["sklearn/impute/tests/test_impute.py"],
                    "spans": [{"file": "sklearn/impute/tests/test_impute.py", "start": 10, "end": 20}],
                    "symbols": [],
                }
            ],
            "retrieved_context_files": ["sklearn/impute/tests/test_impute.py"],
            "retrieved_context_spans": [{"file": "sklearn/impute/tests/test_impute.py", "start": 10, "end": 20}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == [
        "sklearn/impute/_iterative.py",
        "sklearn/impute/tests/test_impute.py",
    ]
    assert converted["traj_data"]["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 120
    assert converted["traj_data"]["pred_spans"]["sklearn/impute/tests/test_impute.py"][0]["start"] == 10
    assert len(converted["traj_data"]["pred_steps"]) == 2
    assert converted["traj_data"]["pred_files_provenance"]["sklearn/impute/_iterative.py"] == "trace_inference"
    assert converted["traj_data"]["pred_files_provenance"]["sklearn/impute/tests/test_impute.py"] == "agent_report"
    assert "trace_inference" in converted["traj_data"]["pred_files_source"]
    assert "agent_report" in converted["traj_data"]["pred_files_source"]


def test_convert_records_to_jsonl_resolves_host_absolute_raw_response_sidecar(tmp_path) -> None:
    source_dir = tmp_path / "variants" / "baseline" / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-1"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task.codex-record.json"
    raw_response_path = task_dir / "raw-response.json"
    raw_response_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "response_format": "jsonl-events",
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "/bin/zsh -lc 'sed -n 1,5p pkg/mod.py'",
                            "aggregated_output": "line 1\nline 2\n",
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-1",
                "workspace_path": "/workspaces/task-1",
                "repo_url": "https://github.com/example/repo.git",
                "commit": "abc123",
                "raw_response_path": "/Users/alice/private/run/raw-response.json",
                "model_patch": "",
                "final_output": {
                    "task_id": "task-1",
                    "status": "completed",
                    "retrieved_context_files": [],
                    "retrieved_context_spans": [],
                    "retrieved_context_symbols": [],
                },
            }
        ),
        encoding="utf-8",
    )
    task_results = source_dir.parent.parent / "task-results.jsonl"
    task_results.write_text(
        json.dumps({"instance_id": "task-1", "bench": "Verified", "record_path": str(record_path)}) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pred.jsonl"

    summary = convert_records_to_jsonl(source_dir=source_dir, expected_agent="codex", out_path=out_path)
    predictions = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert summary["prediction_count"] == 1
    assert summary["conversion_error_count"] == 0
    assert predictions[0]["traj_data"]["pred_files"] == ["pkg/mod.py"]
    assert predictions[0]["traj_data"]["pred_files_source"] == ["trace_inference"]


def test_convert_run_record_preserves_symbols_when_merging_duplicate_steps() -> None:
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "raw_response": {
            "agent": "codex",
            "response_format": "jsonl-events",
            "events": [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "/bin/zsh -lc 'rg -n \"class Foo\" a.py'",
                        "aggregated_output": "a.py:1:class Foo:\n",
                    },
                }
            ],
        },
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "retrieval_steps": [
                {
                    "files": ["a.py"],
                    "spans": [{"file": "a.py", "start": 1, "end": 1}],
                    "symbols": [{"file": "a.py", "name": "Foo"}],
                }
            ],
            "retrieved_context_files": [],
            "retrieved_context_spans": [],
            "retrieved_context_symbols": [],
            "notes": "",
        },
        "model_patch": "",
    }

    converted = convert_run_record(record)

    assert len(converted["traj_data"]["pred_steps"]) == 1
    assert converted["traj_data"]["pred_steps"][0]["symbols"] == {"a.py": ["Foo"]}
