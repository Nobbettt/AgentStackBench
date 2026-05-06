
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

def test_evaluate_resolution_for_suite_clears_stale_resolution_artifacts_by_default(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    stale_dir = variant_dir / "resolution-eval" / "verified" / "old-run"
    stale_dir.mkdir(parents=True)
    (stale_dir / "report.json").write_text("{}", encoding="utf-8")
    stale_export = variant_dir / "resolution-exports" / "old.jsonl"
    stale_export.parent.mkdir(parents=True)
    stale_export.write_text("{}\n", encoding="utf-8")

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
        lambda **kwargs: {"resolved_ids": [instance_id], "unresolved_ids": []},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
        run_suffix="fresh",
        clean_resolution_artifacts=True,
    )

    assert summary["status"] == "completed"
    assert not stale_dir.exists()
    assert not stale_export.exists()
def test_evaluate_resolution_for_suite_removes_stale_error_marker_before_success(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    run_id = "demo-suite-baseline-Verified-resolution-fresh"
    stale_error = variant_dir / "resolution-eval" / "verified" / run_id / "resolution-error.json"
    stale_error.parent.mkdir(parents=True)
    stale_error.write_text(json.dumps({"status": "backend_unavailable"}), encoding="utf-8")

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
        lambda **kwargs: {"resolved_ids": [instance_id], "unresolved_ids": []},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
        run_suffix="fresh",
    )

    assert summary["status"] == "completed"
    assert not stale_error.exists()
