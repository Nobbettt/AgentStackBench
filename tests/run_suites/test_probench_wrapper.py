
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .helpers import load_resolution_wrapper_module as _load_module

def test_pro_wrapper_reuses_existing_instance_results_and_merges(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/probench.py", "pro_resolution_wrapper_resume")
    pro_root = tmp_path / "probench"
    raw_sample = pro_root / "helper_code" / "sweap_eval_full_v2.jsonl"
    raw_sample.parent.mkdir(parents=True)
    raw_sample.write_text(
        '{"instance_id":"task-a","before_repo_set_cmd":"","selected_test_files_to_run":[],"base_commit":"abc","repo":"owner/repo","FAIL_TO_PASS":[],"PASS_TO_PASS":[]}\n'
        '{"instance_id":"task-b","before_repo_set_cmd":"","selected_test_files_to_run":[],"base_commit":"abc","repo":"owner/repo","FAIL_TO_PASS":[],"PASS_TO_PASS":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROBENCH_ROOT", pro_root)
    monkeypatch.setattr(module, "PROBENCH_SCRIPT", pro_root / "swe_bench_pro_eval.py")
    monkeypatch.setattr(module, "PROBENCH_RAW_SAMPLE_JSONL", raw_sample)
    monkeypatch.setattr(module, "PROBENCH_RUN_SCRIPTS", pro_root / "run_scripts")

    patch_path = tmp_path / "predictions.json"
    patch_path.write_text(
        '[{"instance_id":"task-a","patch":"diff --git a/a.py b/a.py\\n","prefix":"codex"},'
        '{"instance_id":"task-b","patch":"diff --git a/b.py b/b.py\\n","prefix":"codex"}]\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "work" / "evaluation_results"
    existing_instance_dir = tmp_path / "work" / "instances" / "task-a"
    existing_dir = existing_instance_dir / "evaluation_results"
    existing_dir.mkdir(parents=True)
    (existing_dir / "eval_results.json").write_text('{"task-a": true}\n', encoding="utf-8")
    module._write_metadata(
        existing_instance_dir,
        module._prediction_metadata(
            {
                "instance_id": "task-a",
                "patch": "diff --git a/a.py b/a.py\n",
                "prefix": "codex",
            }
        ),
    )

    commands: list[list[str]] = []

    def fake_run(command, check, cwd):
        del check, cwd
        commands.append(list(command))
        out_dir = Path(command[command.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_results.json").write_text('{"task-b": false}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            patch_path=patch_path,
            output_dir=output_dir,
            num_workers=2,
            dockerhub_username="jefzda",
            use_local_docker=True,
        ),
    )

    assert module.main() == 0
    assert len(commands) == 1
    assert json.loads((output_dir / "eval_results.json").read_text(encoding="utf-8")) == {
        "task-a": True,
        "task-b": False,
    }
    assert (tmp_path / "work" / "instances" / "task-b" / "resolution-input.json").exists()
def test_pro_wrapper_reruns_existing_result_when_prediction_metadata_changes(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/probench.py", "pro_resolution_wrapper_stale")
    pro_root = tmp_path / "probench"
    raw_sample = pro_root / "helper_code" / "sweap_eval_full_v2.jsonl"
    raw_sample.parent.mkdir(parents=True)
    raw_sample.write_text(
        '{"instance_id":"task-a","before_repo_set_cmd":"","selected_test_files_to_run":[],"base_commit":"abc","repo":"owner/repo","FAIL_TO_PASS":[],"PASS_TO_PASS":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROBENCH_ROOT", pro_root)
    monkeypatch.setattr(module, "PROBENCH_SCRIPT", pro_root / "swe_bench_pro_eval.py")
    monkeypatch.setattr(module, "PROBENCH_RAW_SAMPLE_JSONL", raw_sample)
    monkeypatch.setattr(module, "PROBENCH_RUN_SCRIPTS", pro_root / "run_scripts")

    patch_path = tmp_path / "predictions.json"
    patch_path.write_text(
        '[{"instance_id":"task-a","patch":"diff --git a/new.py b/new.py\\n","prefix":"codex"}]\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "work" / "evaluation_results"
    output_dir.mkdir(parents=True)
    (output_dir / "eval_results.json").write_text('{"task-a": true}\n', encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(command, check, cwd):
        del check, cwd
        commands.append(list(command))
        out_dir = Path(command[command.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_results.json").write_text('{"task-a": false}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            patch_path=patch_path,
            output_dir=output_dir,
            num_workers=1,
            dockerhub_username="jefzda",
            use_local_docker=True,
        ),
    )

    assert module.main() == 0
    assert len(commands) == 1
    assert json.loads((output_dir / "eval_results.json").read_text(encoding="utf-8")) == {"task-a": False}
def _prepare_pro_wrapper(tmp_path: Path, monkeypatch):
    module = _load_module("contextbench/run_suites_resolution_wrappers/probench.py", "pro_resolution_wrapper_failure")
    pro_root = tmp_path / "probench"
    raw_sample = pro_root / "helper_code" / "sweap_eval_full_v2.jsonl"
    raw_sample.parent.mkdir(parents=True)
    raw_sample.write_text(
        '{"instance_id":"instance_repo__repo-1","before_repo_set_cmd":"git checkout abc",'
        '"selected_test_files_to_run":["tests/test_a.py"],"base_commit":"abc","repo":"owner/repo",'
        '"FAIL_TO_PASS":["tests/test_a.py::test_bug"],"PASS_TO_PASS":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROBENCH_ROOT", pro_root)
    monkeypatch.setattr(module, "PROBENCH_SCRIPT", pro_root / "swe_bench_pro_eval.py")
    monkeypatch.setattr(module, "PROBENCH_RAW_SAMPLE_JSONL", raw_sample)
    monkeypatch.setattr(module, "PROBENCH_RUN_SCRIPTS", pro_root / "run_scripts")
    return module
def test_probench_wrapper_nonzero_evaluator_writes_error_not_unresolved(tmp_path: Path, monkeypatch) -> None:
    module = _prepare_pro_wrapper(tmp_path, monkeypatch)
    patch_path = tmp_path / "predictions.json"
    output_dir = tmp_path / "evaluation_results"
    patch_path.write_text(
        '[{"instance_id":"instance_repo__repo-1","patch":"diff","prefix":"codex"}]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            patch_path=patch_path,
            output_dir=output_dir,
            num_workers=1,
            dockerhub_username="jefzda",
            use_local_docker=True,
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check, cwd: subprocess.CompletedProcess(command, 2),
    )

    assert module.main() == 1
    assert json.loads((output_dir / "eval_results.json").read_text(encoding="utf-8")) == {}
    errors = json.loads((output_dir / "eval_errors.json").read_text(encoding="utf-8"))
    assert errors[0]["instance_id"] == "instance_repo__repo-1"
    assert errors[0]["exit_code"] == 2
    assert Path(errors[0]["error_path"]).exists()
def test_probench_wrapper_preserves_successful_false_result(tmp_path: Path, monkeypatch) -> None:
    module = _prepare_pro_wrapper(tmp_path, monkeypatch)
    patch_path = tmp_path / "predictions.json"
    output_dir = tmp_path / "evaluation_results"
    patch_path.write_text(
        '[{"instance_id":"instance_repo__repo-1","patch":"diff","prefix":"codex"}]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            patch_path=patch_path,
            output_dir=output_dir,
            num_workers=1,
            dockerhub_username="jefzda",
            use_local_docker=True,
        ),
    )

    def fake_run(command, check, cwd):
        del check, cwd
        instance_output = Path(command[command.index("--output_dir") + 1])
        instance_output.mkdir(parents=True, exist_ok=True)
        (instance_output / "eval_results.json").write_text('{"instance_repo__repo-1": false}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 0
    assert json.loads((output_dir / "eval_results.json").read_text(encoding="utf-8")) == {
        "instance_repo__repo-1": False
    }
    assert not (output_dir / "eval_errors.json").exists()
def test_probench_wrapper_zero_exit_without_result_writes_error_not_unresolved(tmp_path: Path, monkeypatch) -> None:
    module = _prepare_pro_wrapper(tmp_path, monkeypatch)
    patch_path = tmp_path / "predictions.json"
    output_dir = tmp_path / "evaluation_results"
    patch_path.write_text(
        '[{"instance_id":"instance_repo__repo-1","patch":"diff","prefix":"codex"}]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            patch_path=patch_path,
            output_dir=output_dir,
            num_workers=1,
            dockerhub_username="jefzda",
            use_local_docker=True,
        ),
    )

    def fake_run(command, check, cwd):
        del check, cwd
        instance_output = Path(command[command.index("--output_dir") + 1])
        instance_output.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 1
    assert json.loads((output_dir / "eval_results.json").read_text(encoding="utf-8")) == {}
    errors = json.loads((output_dir / "eval_errors.json").read_text(encoding="utf-8"))
    assert errors[0]["instance_id"] == "instance_repo__repo-1"
    assert errors[0]["exit_code"] is None
    assert Path(errors[0]["error_path"]).exists()
