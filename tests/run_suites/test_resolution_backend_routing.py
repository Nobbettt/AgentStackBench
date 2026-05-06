
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

def test_evaluate_resolution_for_suite_uses_poly_backend_when_available(tmp_path: Path, monkeypatch) -> None:
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

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    def fake_poly_run(**kwargs):
        calls.append(("poly", kwargs["dataset_name"]))
        return {"resolved_ids": [instance_id], "unresolved_ids": []}

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_poly_resolution_evaluation", fake_poly_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert calls == [("poly", "AmazonScience/SWE-PolyBench")]
    assert summary["status"] == "completed"
    assert summary["successful_benches"] == ["Poly"]
    assert summary["failed_benches"] == []
    assert summary["unsupported_benches"] == []
    assert summary["per_bench"]["Poly"]["backend"] == "swe-polybench"
    assert summary["per_bench"]["Poly"]["status"] == "completed"
    assert summary["per_bench"]["Poly"]["prediction_ids"] == [instance_id]
def test_evaluate_resolution_for_suite_uses_pro_backend_when_available(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    contextbench_id = "SWE-Bench-Pro__python__maintenance__bugfix__task-a"
    original_id = "instance_repo__repo-1"
    task_dir = source_dir / "Pro" / contextbench_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{contextbench_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": contextbench_id,
                "original_inst_id": original_id,
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
                "instance_id": contextbench_id,
                "original_inst_id": original_id,
                "bench": "Pro",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pro_root = tmp_path / ".cache" / "probench-eval"
    pro_python = tmp_path / ".cache" / "probench-eval-venv" / "bin" / "python"
    evaluator = pro_root / "swe_bench_pro_eval.py"
    run_scripts = pro_root / "run_scripts"
    dockerfiles = pro_root / "dockerfiles"
    raw_sample_jsonl = pro_root / "helper_code" / "sweap_eval_full_v2.jsonl"
    pro_python.parent.mkdir(parents=True)
    pro_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    run_scripts.mkdir(parents=True)
    dockerfiles.mkdir(parents=True)
    raw_sample_jsonl.parent.mkdir(parents=True)
    raw_sample_jsonl.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    calls: list[tuple[str, str, Path]] = []

    def fake_pro_run(**kwargs):
        calls.append(("pro", kwargs["dataset_name"], kwargs["predictions_path"]))
        return {"resolved_ids": [original_id], "unresolved_ids": []}

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_pro_resolution_evaluation", fake_pro_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert calls == [("pro", "ScaleAI/SWE-bench_Pro", variant_dir / "resolution-exports" / "pro-swebench-pro.json")]
    predictions = json.loads((variant_dir / "resolution-exports" / "pro-swebench-pro.json").read_text(encoding="utf-8"))
    assert predictions == [
        {
            "instance_id": original_id,
            "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
            "prefix": "codex",
        }
    ]
    assert summary["status"] == "completed"
    assert summary["successful_benches"] == ["Pro"]
    assert summary["failed_benches"] == []
    assert summary["unsupported_benches"] == []
    assert summary["per_bench"]["Pro"]["backend"] == "swebench-pro"
    assert summary["per_bench"]["Pro"]["status"] == "completed"
    assert summary["per_bench"]["Pro"]["prediction_ids"] == [original_id]
def test_evaluate_resolution_for_suite_uses_multi_backend_when_available(tmp_path: Path, monkeypatch) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    contextbench_id = "Multi-SWE-Bench__javascript__maintenance__bugfix__task-a"
    original_id = "iamkun__dayjs-734"
    task_dir = source_dir / "Multi" / contextbench_id
    task_dir.mkdir(parents=True, exist_ok=True)
    record_path = task_dir / f"{contextbench_id}.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": contextbench_id,
                "original_inst_id": original_id,
                "status": "completed",
                "ok": True,
                "timeout": False,
                "model_patch": "diff --git a/a.js b/a.js\n--- a/a.js\n+++ b/a.js\n@@ -1 +1 @@\n-x\n+y\n",
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": contextbench_id,
                "original_inst_id": original_id,
                "bench": "Multi",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, Path]] = []

    def fake_multi_run(**kwargs):
        calls.append((kwargs["dataset_name"], kwargs["predictions_path"]))
        return {"resolved_ids": [original_id], "unresolved_ids": []}

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_multi_resolution_evaluation", fake_multi_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    assert calls == [("bytedance-research/Multi-SWE-Bench", variant_dir / "resolution-exports" / "multi-multi-swebench.jsonl")]
    predictions = [
        json.loads(line)
        for line in (variant_dir / "resolution-exports" / "multi-multi-swebench.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert predictions == [
        {
            "org": "iamkun",
            "repo": "dayjs",
            "number": 734,
            "fix_patch": "diff --git a/a.js b/a.js\n--- a/a.js\n+++ b/a.js\n@@ -1 +1 @@\n-x\n+y\n",
        }
    ]
    assert summary["status"] == "completed"
    assert summary["successful_benches"] == ["Multi"]
    assert summary["failed_benches"] == []
    assert summary["unsupported_benches"] == []
    assert summary["per_bench"]["Multi"]["backend"] == "multi-swebench"
    assert summary["per_bench"]["Multi"]["status"] == "completed"
    assert summary["per_bench"]["Multi"]["prediction_ids"] == [original_id]
