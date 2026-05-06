
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .helpers import load_resolution_wrapper_module as _load_module

def test_multibench_wrapper_rejects_stale_final_report(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/multibench.py", "multibench_resolution_wrapper_stale_report")

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"org":"demo","repo":"project","number":123,"model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"org":"demo","repo":"project","number":123}\n', encoding="utf-8")
    output_dir = tmp_path / "evaluation_results"
    output_dir.mkdir()
    stale_report = output_dir / "final_report.json"
    stale_report.write_text('{"resolved_ids":["demo__project-123"],"unresolved_ids":[],"error_ids":[]}\n', encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            predictions_path=pred,
            dataset_path=dataset,
            output_dir=output_dir,
            repo_dir=tmp_path / "repos",
            log_dir=tmp_path / "logs",
            max_workers=1,
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check, cwd: subprocess.CompletedProcess(command, 0),
    )

    assert module.main() == 1
    assert not stale_report.exists()
    error = json.loads((output_dir / "evaluation-error.json").read_text(encoding="utf-8"))
    assert "produced no final_report.json" in error["detail"]
def test_multibench_wrapper_invokes_official_module_and_requires_final_report(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/multibench.py", "multi_resolution_wrapper")
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path.write_text('{"org":"iamkun","repo":"dayjs","number":734,"fix_patch":"diff"}\n', encoding="utf-8")
    dataset_path.write_text('{"org":"iamkun","repo":"dayjs","number":734,"instance_id":"iamkun__dayjs-734"}\n', encoding="utf-8")
    output_dir = tmp_path / "evaluation_results"
    commands: list[list[str]] = []

    def fake_run(command, check, cwd):
        del check
        commands.append(list(command))
        assert cwd == str(output_dir.parent.resolve())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            predictions_path=predictions_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            repo_dir=tmp_path / "repos",
            log_dir=tmp_path / "logs",
            max_workers=1,
        ),
    )

    assert module.main() == 1
    assert commands[0][:3] == [sys.executable, "-m", "multi_swe_bench.harness.run_evaluation"]
    assert commands[0][3:] == ["--config", str(output_dir.parent / "multibench-config.json")]
    config = json.loads((output_dir.parent / "multibench-config.json").read_text(encoding="utf-8"))
    assert config["specifics"] == ["iamkun/dayjs:pr-734"]
    assert config["workdir"] == str(output_dir.parent / "work")
    assert (output_dir.parent / "work").is_dir()
    error = json.loads((output_dir / "evaluation-error.json").read_text(encoding="utf-8"))
    assert error["instance_id"] == "iamkun__dayjs-734"
    assert error["exit_code"] is None
def test_multibench_wrapper_success_when_final_report_exists(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/multibench.py", "multi_resolution_wrapper_success")
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path.write_text('{"org":"iamkun","repo":"dayjs","number":734,"fix_patch":"diff"}\n', encoding="utf-8")
    dataset_path.write_text('{"org":"iamkun","repo":"dayjs","number":734,"instance_id":"iamkun__dayjs-734"}\n', encoding="utf-8")
    output_dir = tmp_path / "evaluation_results"

    def fake_run(command, check, cwd):
        del command, check, cwd
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_report.json").write_text('{"resolved_ids":["iamkun__dayjs-734"],"unresolved_ids":[],"error_ids":[]}\n', encoding="utf-8")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            predictions_path=predictions_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            repo_dir=tmp_path / "repos",
            log_dir=tmp_path / "logs",
            max_workers=1,
        ),
    )

    assert module.main() == 0
    assert not (output_dir / "evaluation-error.json").exists()
