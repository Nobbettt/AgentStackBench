
from __future__ import annotations

import csv
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


from .helpers import _fake_run_coding_agent_task, _make_fake_agent_record, _write_task_inputs

def test_export_resolution_predictions_uses_original_instance_id_and_patch(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-a"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-a.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "SWE-Bench-Verified__python__maintenance__bugfix__task-a",
                "original_inst_id": "psf__requests-1000",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/requests/api.py b/requests/api.py\n--- a/requests/api.py\n+++ b/requests/api.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "SWE-Bench-Verified__python__maintenance__bugfix__task-a",
                "original_inst_id": "psf__requests-1000",
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export_resolution_predictions(
        source_dir=source_dir,
        expected_agent="codex",
        bench="Verified",
        out_path=variant_dir / "resolution-preds" / "verified.jsonl",
    )

    rows = [
        json.loads(line)
        for line in (variant_dir / "resolution-preds" / "verified.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["task_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["coverage_of_attempted_tasks"] == 1.0
    assert summary["is_partial"] is False
    assert rows == [
        {
            "instance_id": "psf__requests-1000",
            "model_patch": "diff --git a/requests/api.py b/requests/api.py\n--- a/requests/api.py\n+++ b/requests/api.py\n@@ -1 +1 @@\n-x\n+y\n",
            "model_name_or_path": "codex",
        }
    ]


def test_export_resolution_predictions_restores_final_patch_newline_and_preserves_whitespace(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-a"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-a.codex-record.json"
    patch = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new  "
    )
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-a",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": patch,
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_resolution_predictions(
        source_dir=source_dir,
        expected_agent="codex",
        bench="Verified",
        out_path=variant_dir / "resolution-preds" / "verified.jsonl",
    )

    [row] = [
        json.loads(line)
        for line in (variant_dir / "resolution-preds" / "verified.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert row["model_patch"] == f"{patch}\n"


def test_export_resolution_predictions_skips_failed_records_even_with_patch(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-a"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-a.codex-record.json"
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-a",
                "status": "failed",
                "ok": False,
                "timeout": False,
                "model_patch": patch,
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "bench": "Verified",
                "status": "failed",
                "ok": False,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export_resolution_predictions(
        source_dir=source_dir,
        expected_agent="codex",
        bench="Verified",
        out_path=variant_dir / "resolution-preds" / "verified.jsonl",
    )

    assert summary["prediction_count"] == 0
    assert summary["missing_patch_count"] == 0
    assert summary["skipped_ineligible_count"] == 1
    assert summary["skipped_ineligible_reasons"] == {"task_result_status_failed": 1}
    assert summary["is_partial"] is True
    assert (variant_dir / "resolution-preds" / "verified.jsonl").read_text(encoding="utf-8") == ""


def test_export_resolution_predictions_keeps_empty_patch_missing(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-a"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-a.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-a",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "   \n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export_resolution_predictions(
        source_dir=source_dir,
        expected_agent="codex",
        bench="Verified",
        out_path=variant_dir / "resolution-preds" / "verified.jsonl",
    )

    assert summary["prediction_count"] == 0
    assert summary["missing_patch_count"] == 1
    assert (variant_dir / "resolution-preds" / "verified.jsonl").read_text(encoding="utf-8") == ""


def test_run_resolution_evaluation_preserves_backend_error_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": "psf__requests-1000", "model_patch": "diff --git a/a.py b/a.py\n"})
        + "\n",
        encoding="utf-8",
    )

    def fake_run_resolution_command(*, cwd: Path, **_: object) -> tuple[int, str]:
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": [], "unresolved_ids": [], "error_ids": ["psf__requests-1000"]}),
            encoding="utf-8",
        )
        return 0, ""

    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=tmp_path / "resolution",
        max_workers=1,
    )

    assert summary["resolved_ids"] == []
    assert summary["unresolved_ids"] == []
    assert summary["error_ids"] == ["psf__requests-1000"]
    assert summary["total_instances"] == 1
    instance_summary = json.loads(
        (tmp_path / "resolution" / "instances" / "psf__requests-1000" / "resolution-result.json").read_text(encoding="utf-8")
    )
    assert instance_summary["status"] == "error"


def test_export_resolution_predictions_does_not_fallback_to_diff_path(tmp_path: Path) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-a"
    task_dir.mkdir(parents=True)
    diff_path = task_dir / "workspace.diff"
    diff_path.write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
        encoding="utf-8",
    )
    record_path = task_dir / "task-a.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-a",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "",
                "diff_path": str(diff_path),
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export_resolution_predictions(
        source_dir=source_dir,
        expected_agent="codex",
        bench="Verified",
        out_path=variant_dir / "resolution-preds" / "verified.jsonl",
    )

    assert summary["prediction_count"] == 0
    assert summary["missing_patch_count"] == 1
    assert (variant_dir / "resolution-preds" / "verified.jsonl").read_text(encoding="utf-8") == ""
