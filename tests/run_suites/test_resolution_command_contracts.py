
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

def test_resolution_command_log_redacts_secret_env_values(tmp_path: Path) -> None:
    log_path = tmp_path / "resolution-command.log"

    returncode, _tail = postprocess._run_resolution_command(
        command=[
            sys.executable,
            "-c",
            "print('ok')",
            "-e",
            "HF_TOKEN=secret-token",
            "--env",
            "OPENAI_API_KEY=secret-key",
            "PLAIN=value",
        ],
        cwd=tmp_path,
        log_path=log_path,
        log_prefix="[test]",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert returncode == 0
    assert "HF_TOKEN=<redacted>" in log_text
    assert "OPENAI_API_KEY=<redacted>" in log_text
    assert "secret-token" not in log_text
    assert "secret-key" not in log_text
    assert "PLAIN=value" in log_text
def test_resolution_command_log_emits_heartbeat_while_subprocess_is_quiet(tmp_path: Path) -> None:
    log_path = tmp_path / "resolution-command.log"

    returncode, tail = postprocess._run_resolution_command(
        command=[sys.executable, "-c", "import time; time.sleep(0.25); print('done')"],
        cwd=tmp_path,
        log_path=log_path,
        log_prefix="[test]",
        heartbeat_interval_seconds=0.05,
        heartbeat_label="quiet-test",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert returncode == 0
    assert "[heartbeat] command=quiet-test" in log_text
    assert "subprocess still running" in log_text
    assert "done" in tail
def test_resolution_heartbeat_label_recognizes_all_resolution_wrappers() -> None:
    assert postprocess._heartbeat_label_for_command(["python", "swebench_wrapper.py"]) == "swebench_wrapper.py"
    assert postprocess._heartbeat_label_for_command(["python", "polybench.py"]) == "polybench.py"
    assert postprocess._heartbeat_label_for_command(["python", "probench.py"]) == "probench.py"
    assert postprocess._heartbeat_label_for_command(["python", "multibench.py"]) == "multibench.py"
