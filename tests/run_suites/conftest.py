
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


@pytest.fixture(autouse=True)
def _mock_swebench_resolution(monkeypatch, tmp_path):
    pro_root = tmp_path / "probench-eval"
    pro_python = tmp_path / "probench-eval-venv" / "bin" / "python"
    pro_python.parent.mkdir(parents=True, exist_ok=True)
    pro_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (pro_root / "helper_code").mkdir(parents=True, exist_ok=True)
    (pro_root / "run_scripts").mkdir(parents=True, exist_ok=True)
    (pro_root / "dockerfiles").mkdir(parents=True, exist_ok=True)
    (pro_root / "swe_bench_pro_eval.py").write_text("# evaluator\n", encoding="utf-8")
    (pro_root / "helper_code" / "sweap_eval_full_v2.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._DEFAULT_SWE_BENCH_PYTHON", Path(sys.executable))
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._DEFAULT_POLY_BENCH_PYTHON", Path(sys.executable))
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._DEFAULT_MULTI_BENCH_PYTHON", Path(sys.executable))
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess._module_available_with_python",
        lambda module_name, python_executable: module_name in {
            "swebench.harness.run_evaluation",
            "poly_bench_evaluation.run_evaluation",
            "multi_swe_bench",
        },
    )
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_ROOT", pro_root)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_PYTHON", pro_python)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_EVALUATOR", pro_root / "swe_bench_pro_eval.py")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_RUN_SCRIPTS", pro_root / "run_scripts")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_DOCKERFILES", pro_root / "dockerfiles")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_RAW_SAMPLE_JSONL", pro_root / "helper_code" / "sweap_eval_full_v2.jsonl")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_id", lambda image: "sha256:test-postprocess")
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_image_id", lambda image: "sha256:test-postprocess")
    monkeypatch.setattr("contextbench.run_suites_core.runner._postprocess_image_supports_evaluation", lambda image: (True, "ok"))
    monkeypatch.setattr("contextbench.run_suites_core.runner._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.evaluate_resolution_for_suite",
        lambda **kwargs: {
            "status": "completed",
            "backend": "mixed",
            "task_count": 0,
            "prediction_count": 0,
            "evaluated_task_count": 0,
            "evaluated_prediction_count": 0,
            "resolved_count": 0,
            "pass_at_1": None,
            "supported_benches": [],
            "successful_benches": [],
            "failed_benches": [],
            "unsupported_benches": [],
            "coverage_of_attempted_tasks": 0.0,
            "is_partial": False,
            "per_bench": {},
        },
    )
