
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

def test_evaluate_resolution_for_suite_routes_verified_to_swebench_and_not_poly(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    for bench, instance_id in (("Verified", "psf__requests-1000"), ("Poly", "SWE-PolyBench__python__task-a")):
        task_dir = source_dir / bench / instance_id
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
        "\n".join(
            [
                json.dumps({"instance_id": "psf__requests-1000", "original_inst_id": "psf__requests-1000", "bench": "Verified", "status": "completed", "ok": True, "timeout": False, "record_path": str(source_dir / "Verified" / "psf__requests-1000" / "psf__requests-1000.codex-record.json")}),
                json.dumps({"instance_id": "SWE-PolyBench__python__task-a", "original_inst_id": "SWE-PolyBench__python__task-a", "bench": "Poly", "status": "completed", "ok": True, "timeout": False, "record_path": str(source_dir / "Poly" / "SWE-PolyBench__python__task-a" / "SWE-PolyBench__python__task-a.codex-record.json")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, str, Path]] = []

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess._module_available_with_python",
        lambda module_name, python_executable: module_name == "swebench.harness.run_evaluation",
    )

    def fake_swebench_run(**kwargs):
        calls.append(("swebench", kwargs["dataset_name"], kwargs["work_dir"]))
        return {"resolved_ids": ["psf__requests-1000"], "unresolved_ids": []}

    def fake_poly_run(**kwargs):
        calls.append(("poly", kwargs["dataset_name"], kwargs["work_dir"]))
        return {"resolved_ids": ["SWE-PolyBench__python__task-a"], "unresolved_ids": []}

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_resolution_evaluation", fake_swebench_run)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_poly_resolution_evaluation", fake_poly_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert [(backend, dataset) for backend, dataset, _ in calls] == [("swebench", "princeton-nlp/SWE-bench_Verified")]
    assert calls[0][2].parent == variant_dir / "resolution-eval" / "verified"
    assert calls[0][2].name.startswith("demo-suite-baseline-Verified-resolution-")
    assert summary["status"] == "partial"
    assert summary["successful_benches"] == ["Verified"]
    assert summary["failed_benches"] == ["Poly"]
    assert summary["per_bench"]["Verified"]["backend"] == "swebench"
    assert summary["per_bench"]["Verified"]["status"] == "completed"
    assert summary["per_bench"]["Poly"]["backend"] == "swe-polybench"
    assert summary["per_bench"]["Poly"]["status"] == "backend_unavailable"

def test_evaluate_resolution_for_suite_preserves_successful_benches_when_one_backend_fails(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    for bench, instance_id in (("Verified", "psf__requests-1000"), ("Verified", "psf__requests-1001"), ("Poly", "SWE-PolyBench__python__task-a")):
        task_dir = source_dir / bench / instance_id
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
        "\n".join(
            [
                json.dumps({"instance_id": "psf__requests-1000", "original_inst_id": "psf__requests-1000", "bench": "Verified", "status": "completed", "ok": True, "timeout": False, "record_path": str(source_dir / "Verified" / "psf__requests-1000" / "psf__requests-1000.codex-record.json")}),
                json.dumps({"instance_id": "psf__requests-1001", "original_inst_id": "psf__requests-1001", "bench": "Verified", "status": "completed", "ok": True, "timeout": False, "record_path": str(source_dir / "Verified" / "psf__requests-1001" / "psf__requests-1001.codex-record.json")}),
                json.dumps({"instance_id": "SWE-PolyBench__python__task-a", "original_inst_id": "SWE-PolyBench__python__task-a", "bench": "Poly", "status": "completed", "ok": True, "timeout": False, "record_path": str(source_dir / "Poly" / "SWE-PolyBench__python__task-a" / "SWE-PolyBench__python__task-a.codex-record.json")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess._module_available_with_python",
        lambda module_name, python_executable: module_name == "swebench.harness.run_evaluation",
    )
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess.run_resolution_evaluation",
        lambda **kwargs: {"resolved_ids": ["psf__requests-1000"], "unresolved_ids": ["psf__requests-1001"]},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert summary["status"] == "partial"
    assert summary["task_count"] == 3
    assert summary["evaluated_task_count"] == 2
    assert summary["resolved_count"] == 1
    assert summary["pass_at_1"] is None
    assert summary["pass_at_1_on_evaluated"] == 0.5
    assert summary["successful_benches"] == ["Verified"]
    assert summary["failed_benches"] == ["Poly"]
    assert summary["per_bench"]["Verified"]["prediction_ids"] == ["psf__requests-1000", "psf__requests-1001"]
    assert summary["per_bench"]["Verified"]["unknown_ids"] == []

def test_evaluate_resolution_for_suite_marks_partial_when_backend_succeeds_with_missing_patches(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    rows: list[str] = []
    for instance_id, patch in (
        ("psf__requests-1000", "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"),
        ("psf__requests-1001", ""),
    ):
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
                    "model_patch": patch,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
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
        )
    (variant_dir / "task-results.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess.run_resolution_evaluation",
        lambda **kwargs: {"resolved_ids": ["psf__requests-1000"], "unresolved_ids": []},
    )

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert summary["status"] == "partial"
    assert summary["is_partial"] is True
    assert summary["partial_benches"] == ["Verified"]
    assert summary["coverage_of_attempted_tasks"] == 0.5
    assert summary["evaluated_coverage_of_attempted_tasks"] == 0.5
    assert summary["per_bench"]["Verified"]["is_partial"] is True
