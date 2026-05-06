
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

def test_resolution_run_id_is_fresh_by_default_and_reuses_only_when_requested(tmp_path: Path) -> None:
    eval_root = tmp_path / "resolution-eval"
    verified_root = eval_root / "verified"
    verified_root.mkdir(parents=True, exist_ok=True)
    old = verified_root / "demo-suite-baseline-Verified-resolution-1"
    new = verified_root / "demo-suite-baseline-Verified-resolution-2"
    old.mkdir()
    new.mkdir()
    old.touch()
    new.touch()

    fresh_run_id = postprocess._resolution_run_id(
        eval_root=eval_root,
        suite_name="demo-suite",
        variant_name="baseline",
        bench="Verified",
        run_suffix="fresh",
    )

    assert fresh_run_id == "demo-suite-baseline-Verified-resolution-fresh"

    resumed_run_id = postprocess._resolution_run_id(
        eval_root=eval_root,
        suite_name="demo-suite",
        variant_name="baseline",
        bench="Verified",
        run_suffix="fresh",
        resume_existing=True,
    )

    assert resumed_run_id == "demo-suite-baseline-Verified-resolution-2"
def test_resolution_instance_resume_does_not_reuse_error_summaries(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "resolution-result.json").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": ["task-a"],
                "status": "error",
            }
        ),
        encoding="utf-8",
    )

    assert postprocess._read_resolution_instance_summary(instance_dir) is None

    (instance_dir / "resolution-result.json").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "resolved_ids": [],
                "unresolved_ids": ["task-a"],
                "error_ids": [],
                "status": "unresolved",
            }
        ),
        encoding="utf-8",
    )

    assert postprocess._read_resolution_instance_summary(instance_dir)["status"] == "unresolved"
def test_resolution_instance_resume_does_not_reuse_empty_unresolved_summary(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "resolution-result.json").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": [],
                "status": "unresolved",
            }
        ),
        encoding="utf-8",
    )

    assert postprocess._read_resolution_instance_summary(instance_dir) is None
def test_resolution_instance_resume_requires_matching_input_metadata(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "resolution-result.json").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "resolved_ids": ["task-a"],
                "unresolved_ids": [],
                "error_ids": [],
                "status": "resolved",
                "input_metadata": {"prediction_sha256": "old"},
            }
        ),
        encoding="utf-8",
    )

    assert (
        postprocess._read_resolution_instance_summary(
            instance_dir,
            expected_input_metadata={"prediction_sha256": "new"},
        )
        is None
    )
    assert postprocess._read_resolution_instance_summary(
        instance_dir,
        expected_input_metadata={"prediction_sha256": "old"},
    )["status"] == "resolved"
def test_run_resolution_evaluation_rewrites_reused_instance_log(tmp_path: Path, monkeypatch) -> None:
    instance_id = "psf__requests-1000"
    work_dir = tmp_path / "work"
    instance_dir = work_dir / "instances" / instance_id
    instance_dir.mkdir(parents=True)
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    input_metadata = postprocess._resolution_instance_input_metadata(
        prediction,
        backend=postprocess._resolution_backend_for_bench("Verified"),
        dataset_name="princeton-nlp/SWE-bench_Verified",
        harness_args=None,
    )
    summary_path = instance_dir / "resolution-result.json"
    summary_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "resolved_ids": [instance_id],
                "unresolved_ids": [],
                "error_ids": [],
                "status": "resolved",
                "input_metadata": input_metadata,
            }
        ),
        encoding="utf-8",
    )
    instance_log = instance_dir / "resolution-command.log"
    instance_log.write_text("old traceback\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path("/python"))

    summary = postprocess.run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
    )

    assert summary["resolved_ids"] == [instance_id]
    assert instance_log.read_text(encoding="utf-8").startswith(f"[reuse] {instance_id} -> {summary_path}")
    assert "old traceback" not in instance_log.read_text(encoding="utf-8")
