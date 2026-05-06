
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .helpers import load_resolution_wrapper_module as _load_module

def test_swebench_wrapper_does_not_patch_official_repo_setup(monkeypatch) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_test")

    del monkeypatch
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_patch_swebench_repo_setup" not in source
    assert "--no-use-pep517" not in source
    assert "git gc --prune=now --aggressive" not in source
def test_swebench_wrapper_loads_codex_report_when_no_report_json_exists(tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_codex_report")
    report_root = tmp_path / "instance"
    report_root.mkdir()
    (report_root / "codex.demo-run.json").write_text(
        '{"resolved_ids":["task-a"],"unresolved_ids":[],"error_ids":["task-b"],"completed_ids":["task-a"],"submitted_ids":["task-a","task-b"],"total_instances":2,"completed_instances":1,"error_instances":1}\n',
        encoding="utf-8",
    )

    report = module._load_instance_report(report_root)

    assert report["resolved_ids"] == ["task-a"]
    assert report["error_ids"] == ["task-b"]
    assert report["submitted_ids"] == ["task-a", "task-b"]
def test_swebench_wrapper_loads_codex_report_from_parent_directory(tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_parent_report")
    report_root = tmp_path / "run-root" / "instances" / "task-a"
    report_root.mkdir(parents=True)
    parent_report = report_root.parent.parent / "codex.demo-run--task-a.json"
    parent_report.write_text(
        '{"resolved_ids":["task-a"],"unresolved_ids":[],"error_ids":[],"completed_ids":["task-a"],"submitted_ids":["task-a"],"total_instances":1,"completed_instances":1,"error_instances":0}\n',
        encoding="utf-8",
    )

    report = module._load_instance_report(report_root)

    assert report["resolved_ids"] == ["task-a"]
    assert report["report_path"] == str(parent_report)
def test_swebench_wrapper_reuses_matching_existing_instance_report(tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_reuse")
    report_root = tmp_path / "run-root" / "instances" / "task-a"
    report_root.mkdir(parents=True)
    parent_report = report_root.parent.parent / "codex.demo-run--task-a.json"
    parent_report.write_text(
        '{"resolved_ids":["task-a"],"unresolved_ids":[],"error_ids":[],"completed_ids":["task-a"],"submitted_ids":["task-a"],"total_instances":1,"completed_instances":1,"error_instances":0}\n',
        encoding="utf-8",
    )

    report = module._load_instance_report_for_id(report_root, "task-a")

    assert report["resolved_ids"] == ["task-a"]
    assert report["report_path"] == str(parent_report)
def test_swebench_wrapper_reruns_when_invoked_with_stale_existing_report(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_rerun_stale")

    run_eval_mod = types.ModuleType("swebench.harness.run_evaluation")
    calls = {"count": 0}

    def fake_run_evaluation_main(**kwargs):
        calls["count"] += 1
        report_dir = Path(str(kwargs["report_dir"]))
        (report_dir / "codex.new-run.json").write_text(
            '{"resolved_ids":[],"unresolved_ids":["task-a"],"error_ids":[],"completed_ids":["task-a"],"submitted_ids":["task-a"],"total_instances":1,"completed_instances":1,"error_instances":0}\n',
            encoding="utf-8",
        )

    run_eval_mod.main = fake_run_evaluation_main
    monkeypatch.setitem(sys.modules, "swebench", types.ModuleType("swebench"))
    monkeypatch.setitem(sys.modules, "swebench.harness", types.ModuleType("swebench.harness"))
    monkeypatch.setitem(sys.modules, "swebench.harness.run_evaluation", run_eval_mod)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    stale_report = tmp_path / "instances" / "task-a" / "codex.old-run.json"
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text(
        '{"resolved_ids":["task-a"],"unresolved_ids":[],"error_ids":[],"completed_ids":["task-a"],"submitted_ids":["task-a"],"total_instances":1,"completed_instances":1,"error_instances":0}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            predictions_path=str(pred),
            max_workers=1,
            run_id="run",
            timeout=60,
            report_dir=str(tmp_path),
        ),
    )

    assert module.main() == 0
    assert calls["count"] == 1
    aggregate = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert aggregate["resolved_ids"] == []
    assert aggregate["unresolved_ids"] == ["task-a"]
def test_swebench_wrapper_propagates_upstream_evaluator_errors(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/swebench_wrapper.py", "swebench_resolution_wrapper_strict_errors")

    run_eval_mod = types.ModuleType("swebench.harness.run_evaluation")

    def fake_run_evaluation_main(**kwargs):
        del kwargs
        raise RuntimeError("docker failed")

    run_eval_mod.main = fake_run_evaluation_main
    monkeypatch.setitem(sys.modules, "swebench", types.ModuleType("swebench"))
    monkeypatch.setitem(sys.modules, "swebench.harness", types.ModuleType("swebench.harness"))
    monkeypatch.setitem(sys.modules, "swebench.harness.run_evaluation", run_eval_mod)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            predictions_path=str(pred),
            max_workers=1,
            run_id="run",
            timeout=60,
            report_dir=str(tmp_path),
        ),
    )

    with pytest.raises(RuntimeError, match="docker failed"):
        module.main()
