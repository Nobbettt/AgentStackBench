
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

def test_probench_setup_installs_pinned_evaluator_into_expected_venv(monkeypatch, tmp_path: Path) -> None:
    pro_root = tmp_path / ".cache" / "probench-eval"
    expected_python = tmp_path / ".cache" / "probench-eval-venv" / "bin" / "python"
    commands: list[list[str]] = []

    monkeypatch.setattr(run_suites_setup, "PRO_BENCH_ROOT", pro_root)
    monkeypatch.setattr(run_suites_setup, "PRO_BENCH_PYTHON", expected_python)
    monkeypatch.setattr(run_suites_setup.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def fake_run(command, check):
        del check
        commands.append(list(command))
        if command[:3] == ["git", "clone", "--no-checkout"]:
            pro_root.mkdir(parents=True)
            (pro_root / "requirements.txt").write_text("docker\n", encoding="utf-8")
        if command[:2] == ["/usr/bin/uv", "venv"]:
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(run_suites_setup.subprocess, "run", fake_run)

    assert run_suites_setup.setup_probench() == 0
    assert commands[0] == ["git", "clone", "--no-checkout", run_suites_setup.PRO_BENCH_REPO, str(pro_root)]
    assert commands[1] == ["git", "-C", str(pro_root), "fetch", "--depth", "1", "origin", run_suites_setup.PRO_BENCH_COMMIT]
    assert commands[2] == ["git", "-C", str(pro_root), "checkout", "--detach", run_suites_setup.PRO_BENCH_COMMIT]
    assert commands[3] == ["/usr/bin/uv", "venv", "--python", "3.11", str(expected_python.parents[1])]
    assert commands[4] == ["/usr/bin/uv", "pip", "install", "--python", str(expected_python), f"pip=={run_suites_setup.PRO_BENCH_PIP_VERSION}"]
    assert commands[5] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        str(expected_python),
        "-r",
        str(pro_root / "requirements.txt"),
        "-c",
        str(run_suites_setup.PRO_BENCH_CONSTRAINTS),
    ]
    assert commands[6] == [str(expected_python), str(pro_root / "swe_bench_pro_eval.py"), "--help"]
