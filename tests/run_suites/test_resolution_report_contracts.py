
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

def test_load_resolution_report_ignores_stale_error_summary(tmp_path: Path) -> None:
    (tmp_path / "resolution-error.json").write_text(
        json.dumps({"resolved_ids": [], "unresolved_ids": ["stale"], "error_detail": "stale"}),
        encoding="utf-8",
    )
    (tmp_path / "resolution-result.json").write_text(
        json.dumps({"resolved_ids": [], "unresolved_ids": [], "status": "unresolved"}),
        encoding="utf-8",
    )
    real_report = tmp_path / "nested" / "report.json"
    real_report.parent.mkdir()
    real_report.write_text(
        json.dumps({"resolved_ids": ["fresh"], "unresolved_ids": []}),
        encoding="utf-8",
    )

    summary = postprocess._load_resolution_report(tmp_path)

    assert summary["resolved_ids"] == ["fresh"]
    assert summary["report_path"] == str(real_report)
def test_run_pro_resolution_evaluation_uses_official_local_docker_contract(tmp_path: Path, monkeypatch) -> None:
    pro_root = tmp_path / ".cache" / "probench-eval"
    pro_python = tmp_path / ".cache" / "probench-eval-venv" / "bin" / "python"
    evaluator = pro_root / "swe_bench_pro_eval.py"
    run_scripts = pro_root / "run_scripts"
    dockerfiles = pro_root / "dockerfiles"
    raw_sample_jsonl = pro_root / "helper_code" / "sweap_eval_full_v2.jsonl"
    real_python = tmp_path / "homebrew" / "python3.11"
    real_python.parent.mkdir(parents=True)
    real_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    pro_python.parent.mkdir(parents=True)
    pro_python.symlink_to(real_python)
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    run_scripts.mkdir(parents=True)
    dockerfiles.mkdir(parents=True)
    raw_sample_jsonl.parent.mkdir(parents=True)
    raw_sample_jsonl.write_text(
        json.dumps(
            {
                "instance_id": "instance_repo__repo-1",
                "before_repo_set_cmd": "git checkout abc",
                "selected_test_files_to_run": ["tests/test_a.py"],
                "base_commit": "abc",
                "repo": "owner/repo",
                "FAIL_TO_PASS": ["tests/test_a.py::test_bug"],
                "PASS_TO_PASS": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postprocess, "_PRO_BENCH_ROOT", pro_root)
    monkeypatch.setattr(postprocess, "_PRO_BENCH_PYTHON", pro_python)
    monkeypatch.setattr(postprocess, "_PRO_BENCH_EVALUATOR", evaluator)
    monkeypatch.setattr(postprocess, "_PRO_BENCH_RUN_SCRIPTS", run_scripts)
    monkeypatch.setattr(postprocess, "_PRO_BENCH_DOCKERFILES", dockerfiles)
    monkeypatch.setattr(postprocess, "_PRO_BENCH_RAW_SAMPLE_JSONL", raw_sample_jsonl)
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "instance_repo__repo-1",
                    "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
                    "prefix": "codex",
                }
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["log_path"] = log_path
        captured["log_prefix"] = log_prefix
        output_dir = tmp_path / "work" / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_repo__repo-1": True}),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_pro_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        run_id="demo-pro",
        work_dir=tmp_path / "work",
        max_workers=3,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == [str(pro_python), str(postprocess._PROBENCH_RESOLUTION_WRAPPER)]
    assert "--use_local_docker" in command
    assert command[command.index("--dockerhub_username") + 1] == "jefzda"
    assert command[command.index("--num_workers") + 1] == "3"
    assert captured["cwd"] == (tmp_path / "work").resolve()
    assert captured["log_path"] == (tmp_path / "work" / "resolution-command.log").resolve()
    assert summary["resolved_ids"] == ["instance_repo__repo-1"]
    assert summary["unresolved_ids"] == []
    assert summary["raw_sample_path"].endswith("raw-sample.csv")
def test_run_poly_resolution_evaluation_preserves_venv_python_symlink(tmp_path: Path, monkeypatch) -> None:
    real_python = tmp_path / "homebrew" / "python3.11"
    poly_python = tmp_path / ".cache" / "polybench-eval-venv" / "bin" / "python"
    real_python.parent.mkdir(parents=True)
    real_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    poly_python.parent.mkdir(parents=True)
    poly_python.symlink_to(real_python)
    monkeypatch.setattr(postprocess, "_DEFAULT_POLY_BENCH_PYTHON", poly_python)

    predictions_path = tmp_path / "poly.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "SWE-PolyBench__python__task-a",
                "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
                "model_name_or_path": "codex",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "poly-work"

    captured: dict[str, object] = {}
    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        captured["command"] = command
        instance_dir = work_dir / "instances" / "SWE-PolyBench__python__task-a"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "result.json").write_text(
            json.dumps(
                {
                    "resolved": ["SWE-PolyBench__python__task-a"],
                    "not_resolved": [],
                    "total_instances": 1,
                    "total_resolved": 1,
                    "total_unresolved": 0,
                    "total_empty_patch_instances": 0,
                    "generation": ["SWE-PolyBench__python__task-a"],
                    "no_generation": [],
                    "patch_applied": ["SWE-PolyBench__python__task-a"],
                    "with_logs": ["SWE-PolyBench__python__task-a"],
                }
            ),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_poly_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="AmazonScience/SWE-PolyBench",
        run_id="demo-poly",
        work_dir=work_dir,
        max_workers=2,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == [str(poly_python), str(postprocess._POLYBENCH_RESOLUTION_WRAPPER)]
    assert summary["resolved_ids"] == ["SWE-PolyBench__python__task-a"]
    assert summary["unresolved_ids"] == []
    assert summary["resolved_count"] == 1
    assert summary["report_path"] == str(work_dir.resolve() / "result.json")
    assert summary["dataset_subset_path"] == str(work_dir.resolve() / "dataset-subset.csv")
def test_run_multi_resolution_evaluation_uses_official_wrapper_contract(tmp_path: Path, monkeypatch) -> None:
    multi_python = tmp_path / ".cache" / "multibench-eval-venv" / "bin" / "python"
    multi_python.parent.mkdir(parents=True)
    multi_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_DEFAULT_MULTI_BENCH_PYTHON", multi_python)

    predictions_path = tmp_path / "multi.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "org": "iamkun",
                "repo": "dayjs",
                "number": 734,
                "fix_patch": "diff --git a/a.js b/a.js\n--- a/a.js\n+++ b/a.js\n@@ -1 +1 @@\n-x\n+y\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "multi-work"

    monkeypatch.setattr(
        postprocess,
        "_write_multi_dataset_jsonl",
        lambda *, dataset_name, instance_ids, out_path: out_path.write_text(
            '{"org":"iamkun","repo":"dayjs","number":734,"instance_id":"iamkun__dayjs-734"}\n',
            encoding="utf-8",
        ),
    )

    captured: dict[str, object] = {}

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        del log_prefix, env
        captured["command"] = command
        captured["cwd"] = cwd
        captured["log_path"] = log_path
        result_root = work_dir.resolve() / "instances" / "iamkun__dayjs-734" / "evaluation_results"
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / "final_report.json").write_text(
            json.dumps(
                {
                    "resolved_ids": ["iamkun/dayjs:pr-734"],
                    "unresolved_ids": [],
                    "error_ids": [],
                    "total_instances": 1,
                    "resolved_instances": 1,
                    "unresolved_instances": 0,
                    "error_instances": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_multi_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="bytedance-research/Multi-SWE-Bench",
        run_id="demo-multi",
        work_dir=work_dir,
        max_workers=1,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == [str(multi_python), str(postprocess._MULTIBENCH_RESOLUTION_WRAPPER)]
    assert command[command.index("--predictions-path") + 1].endswith("predictions.jsonl")
    assert command[command.index("--dataset-path") + 1].endswith("dataset.jsonl")
    assert captured["cwd"] == work_dir.resolve() / "instances" / "iamkun__dayjs-734"
    assert summary["resolved_ids"] == ["iamkun__dayjs-734"]
    assert summary["unresolved_ids"] == []
    assert summary["resolved_count"] == 1
def test_multi_report_ids_must_be_official_or_contextbench_format(tmp_path: Path) -> None:
    result_root = tmp_path / "evaluation_results"
    result_root.mkdir()
    (result_root / "final_report.json").write_text(
        json.dumps(
            {
                "resolved_ids": ["iamkun/dayjs:pr-734"],
                "unresolved_ids": ["iamkun__dayjs-735"],
                "error_ids": ["not-a-valid-multi-id"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invalid Multi-SWE-Bench report id"):
        postprocess._load_multi_resolution_report(result_root)
