
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

def test_evaluate_resolution_for_suite_writes_error_summary_for_failed_backend(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_id = "psf__requests-1000"
    task_dir = source_dir / "Verified" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{instance_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "original_inst_id": instance_id,
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

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    def fake_swebench_run(**kwargs):
        raise ResolutionCommandError(
            message="boom",
            exit_code=1,
            log_path=str((variant_dir / "resolution-eval" / "verified" / "resolution-command.log").resolve()),
            tail="tail output",
        )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_resolution_evaluation", fake_swebench_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    bench_summary = summary["per_bench"]["Verified"]
    error_summary_path = Path(str(bench_summary["error_summary_path"]))
    payload = json.loads(error_summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert bench_summary["status"] == "failed"
    assert bench_summary["prediction_ids"] == [instance_id]
    assert bench_summary["log_path"].endswith("resolution-command.log")
    assert payload["tail"] == "tail output"
    assert payload["exit_code"] == 1
def test_evaluate_resolution_for_suite_marks_backend_partial_when_error_ids_exist(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_id = "SWE-PolyBench__python__task-a"
    task_dir = source_dir / "Poly" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{instance_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "bench": "Poly",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    def fake_poly_run(**kwargs):
        del kwargs
        return {
            "resolved_ids": [],
            "unresolved_ids": [],
            "error_ids": [instance_id],
            "resolved_count": 0,
        }

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_poly_resolution_evaluation", fake_poly_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    bench_summary = summary["per_bench"]["Poly"]
    assert summary["status"] == "failed"
    assert summary["failed_benches"] == ["Poly"]
    assert bench_summary["status"] == "failed"
    assert bench_summary["error_ids"] == [instance_id]
    assert bench_summary["unknown_ids"] == []
def test_evaluate_resolution_for_suite_fails_when_backend_omits_prediction_id(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_id = "psf__requests-1000"
    task_dir = source_dir / "Verified" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{instance_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps({"instance_id": instance_id, "original_inst_id": instance_id, "bench": "Verified", "status": "completed", "ok": True, "timeout": False, "record_path": str(record_path)}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess.run_resolution_evaluation",
        lambda **kwargs: {"resolved_ids": [], "unresolved_ids": [], "error_ids": [], "total_instances": 0},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    bench_summary = summary["per_bench"]["Verified"]
    assert summary["status"] == "failed"
    assert bench_summary["status"] == "failed"
    assert bench_summary["unknown_ids"] == [instance_id]
    assert "missing evaluator result ids" in bench_summary["error_detail"]
    assert Path(bench_summary["error_summary_path"]).exists()
def test_evaluate_resolution_for_suite_preserves_partial_results_on_resolution_command_error(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_a = "psf__requests-1000"
    instance_b = "psf__requests-1001"
    task_rows = []
    for instance_id in [instance_a, instance_b]:
        task_dir = source_dir / "Verified" / instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        record_path = task_dir / f"{instance_id}.codex-record.json"
        record_path.write_text(
            json.dumps(
                {
                    "agent": "codex",
                    "instance_id": instance_id,
                    "original_inst_id": instance_id,
                    "status": "completed",
                    "ok": True,
                    "timeout": False,
                    "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
                }
            ),
            encoding="utf-8",
        )
        task_rows.append(
            {
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
    (variant_dir / "task-results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in task_rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    partial_report_path = variant_dir / "resolution-eval" / "verified" / "demo-suite-baseline-Verified-resolution-attempt" / "report.json"
    partial_report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report_path.write_text(
        json.dumps({"resolved_ids": [instance_a], "unresolved_ids": [], "error_ids": []}),
        encoding="utf-8",
    )

    def fake_swebench_run(**kwargs):
        raise ResolutionCommandError(
            message="boom",
            exit_code=137,
            log_path=str((partial_report_path.parent / "resolution-command.log").resolve()),
            tail="tail output",
        )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_resolution_evaluation", fake_swebench_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
        run_suffix="attempt",
    )

    bench_summary = summary["per_bench"]["Verified"]
    assert summary["status"] == "failed"
    assert summary["successful_benches"] == []
    assert summary["failed_benches"] == ["Verified"]
    assert summary["evaluated_task_count"] == 0
    assert summary["resolved_count"] == 0
    assert bench_summary["status"] == "failed"
    assert bench_summary["resolved_ids"] == [instance_a]
    assert bench_summary["is_partial"] is True
    assert bench_summary["pass_at_1"] is None
    assert Path(bench_summary["error_summary_path"]).exists()
def test_evaluate_resolution_for_suite_fails_partial_report_from_evaluator_error(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_id = "psf__requests-1000"
    task_dir = source_dir / "Verified" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{instance_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps({"instance_id": instance_id, "original_inst_id": instance_id, "bench": "Verified", "status": "completed", "ok": True, "timeout": False, "record_path": str(record_path)}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess.run_resolution_evaluation",
        lambda **kwargs: {"resolved_ids": [], "unresolved_ids": [], "_partial_from_error": True},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert summary["evaluated_task_count"] == 0
    assert summary["evaluated_prediction_count"] == 0
    assert summary["evaluated_coverage_of_attempted_tasks"] == 0.0
    assert summary["status"] == "failed"
    assert summary["failed_benches"] == ["Verified"]
    assert summary["per_bench"]["Verified"]["status"] == "failed"
    assert summary["per_bench"]["Verified"]["pass_at_1"] is None
    assert Path(summary["per_bench"]["Verified"]["error_summary_path"]).exists()
