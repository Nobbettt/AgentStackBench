
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .helpers import load_resolution_wrapper_module as _load_module

def test_poly_wrapper_removes_only_stale_nonrunning_containers() -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_test")

    removed: list[tuple[str, bool]] = []

    class NotFound(Exception):
        pass

    class FakeContainer:
        def __init__(self, name: str, status: str) -> None:
            self.name = name
            self.status = status

        def reload(self) -> None:
            return None

        def remove(self, force: bool) -> None:
            removed.append((self.name, force))

    class FakeContainers:
        def __init__(self) -> None:
            self._by_name = {
                "container_polybench_python_keras-team__keras-18553": FakeContainer(
                    "container_polybench_python_keras-team__keras-18553",
                    "exited",
                )
            }

        def get(self, name: str):
            if name not in self._by_name:
                raise NotFound(name)
            return self._by_name[name]

    class FakeClient:
        def __init__(self) -> None:
            self.containers = FakeContainers()

    module._cleanup_stale_poly_containers(
        [
            {"instance_id": "keras-team__keras-18553", "language": "Python"},
            {"instance_id": "huggingface__transformers-28517", "language": "Python"},
        ],
        client=FakeClient(),
    )

    assert removed == [("container_polybench_python_keras-team__keras-18553", True)]
def test_poly_wrapper_fails_for_active_conflicting_container() -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_running_test")

    removed: list[bool] = []

    class FakeContainer:
        def __init__(self, status: str) -> None:
            self.status = status

        def reload(self) -> None:
            return None

        def remove(self, force: bool) -> None:
            removed.append(force)

    class FakeContainers:
        def get(self, name: str):
            assert name == "container_polybench_python_keras-team__keras-18553"
            return FakeContainer("running")

    class FakeClient:
        def __init__(self) -> None:
            self.containers = FakeContainers()

    module._cleanup_stale_poly_containers(
        [{"instance_id": "keras-team__keras-18553", "language": "Python"}],
        client=FakeClient(),
    )

    assert removed == [True]
def test_poly_wrapper_aggregates_error_ids() -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_aggregate")

    payload = module._aggregate_poly_results(
        [
            {"resolved": ["task-a"], "not_resolved": [], "generation": ["task-a"], "no_generation": [], "patch_applied": ["task-a"], "with_logs": ["task-a"], "error_ids": []},
            {"resolved": [], "not_resolved": ["task-b"], "generation": ["task-b"], "no_generation": [], "patch_applied": [], "with_logs": [], "error_ids": []},
            {"resolved": [], "not_resolved": [], "generation": [], "no_generation": [], "patch_applied": [], "with_logs": [], "error_ids": ["task-c"]},
        ]
    )

    assert payload["resolved"] == ["task-a"]
    assert payload["not_resolved"] == ["task-b"]
    assert payload["error_ids"] == ["task-c"]
    assert payload["total_instances"] == 3
def test_poly_wrapper_preserves_official_prebuilt_image_pull(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_local_only")

    docker_utils_mod = types.ModuleType("poly_bench_evaluation.docker_utils")
    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    class DummyManager:
        def try_pull_prebuilt_image(self, instance_id):
            return True

    captured: dict[str, object] = {}

    def fake_evaluate_predictions(**kwargs):
        captured["kwargs"] = kwargs
        assert docker_utils_mod.DockerManager.try_pull_prebuilt_image(None, "task-a") is True
        assert run_eval_mod.DockerManager.try_pull_prebuilt_image(None, "task-a") is True
        result_dir = Path(kwargs["result_path"])
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir.parent / "result.json").write_text(
            '{"resolved":["task-a"],"not_resolved":[],"generation":["task-a"],"no_generation":[],"patch_applied":["task-a"],"with_logs":["task-a"],"total_empty_patch_instances":0,"total_instances":1,"total_resolved":1,"total_unresolved":0}\n',
            encoding="utf-8",
        )

    docker_utils_mod.DockerManager = DummyManager
    run_eval_mod.DockerManager = DummyManager
    run_eval_mod.evaluate_predictions = fake_evaluate_predictions

    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.docker_utils", docker_utils_mod)
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python", "patch": "", "test_patch": "", "repo": "owner/repo", "base_commit": "abc", "Dockerfile": "FROM python:3.11", "F2P": [], "P2P": [], "test_command": "pytest", "modified_nodes": []}], ["instance_id", "language", "patch", "test_patch", "repo", "base_commit", "Dockerfile", "F2P", "P2P", "test_command", "modified_nodes"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    assert module.main() == 0
    assert captured["kwargs"]["delete_image"] is False
    assert captured["kwargs"]["skip_existing"] is True
    assert captured["kwargs"]["repo_path"].endswith("repos")
def test_poly_wrapper_fails_clearly_when_dataset_is_missing_prediction_instance(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_missing_instance")

    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    def fail_if_called(**kwargs):
        del kwargs
        raise AssertionError("evaluator should not run when the selected dataset row is missing")

    run_eval_mod.evaluate_predictions = fail_if_called
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python"}], ["instance_id", "language"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"missing-task","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    with pytest.raises(RuntimeError, match="SWE-PolyBench dataset is missing selected instances: missing-task"):
        module.main()
def test_poly_wrapper_captures_current_cwd_result_json(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_cwd_result")

    docker_utils_mod = types.ModuleType("poly_bench_evaluation.docker_utils")
    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    class DummyManager:
        pass

    def fake_evaluate_predictions(**kwargs):
        del kwargs
        Path("result.json").write_text(
            '{"resolved":["task-a"],"not_resolved":[],"generation":["task-a"],"no_generation":[],"patch_applied":["task-a"],"with_logs":["task-a"],"total_empty_patch_instances":0,"total_instances":1,"total_resolved":1,"total_unresolved":0}\n',
            encoding="utf-8",
        )

    docker_utils_mod.DockerManager = DummyManager
    docker_utils_mod.docker = types.SimpleNamespace(errors=types.SimpleNamespace(BuildError=RuntimeError))
    run_eval_mod.DockerManager = DummyManager
    run_eval_mod.evaluate_predictions = fake_evaluate_predictions

    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.docker_utils", docker_utils_mod)
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python", "patch": "", "test_patch": "", "repo": "owner/repo", "base_commit": "abc", "Dockerfile": "FROM python:3.11", "F2P": [], "P2P": [], "test_command": "pytest", "modified_nodes": []}], ["instance_id", "language", "patch", "test_patch", "repo", "base_commit", "Dockerfile", "F2P", "P2P", "test_command", "modified_nodes"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    assert module.main() == 0
    assert json.loads((tmp_path / "instances" / "task-a" / "result.json").read_text(encoding="utf-8"))["resolved"] == ["task-a"]
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["resolved"] == ["task-a"]
def test_poly_wrapper_records_error_result_for_resume(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_error_result")

    docker_utils_mod = types.ModuleType("poly_bench_evaluation.docker_utils")
    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    class DummyManager:
        pass

    def fake_evaluate_predictions(**kwargs):
        del kwargs
        raise RuntimeError("docker build failed")

    docker_utils_mod.DockerManager = DummyManager
    docker_utils_mod.docker = types.SimpleNamespace(errors=types.SimpleNamespace(BuildError=RuntimeError))
    run_eval_mod.DockerManager = DummyManager
    run_eval_mod.evaluate_predictions = fake_evaluate_predictions

    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.docker_utils", docker_utils_mod)
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python", "patch": "", "test_patch": "", "repo": "owner/repo", "base_commit": "abc", "Dockerfile": "FROM python:3.11", "F2P": [], "P2P": [], "test_command": "pytest", "modified_nodes": []}], ["instance_id", "language", "patch", "test_patch", "repo", "base_commit", "Dockerfile", "F2P", "P2P", "test_command", "modified_nodes"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    assert module.main() == 0
    instance_result = json.loads((tmp_path / "instances" / "task-a" / "result.json").read_text(encoding="utf-8"))
    aggregate_result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert instance_result["error_ids"] == ["task-a"]
    assert aggregate_result["error_ids"] == ["task-a"]
def test_poly_wrapper_retries_existing_error_result(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_retry_error")

    docker_utils_mod = types.ModuleType("poly_bench_evaluation.docker_utils")
    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    class DummyManager:
        pass

    calls = {"count": 0}

    def fake_evaluate_predictions(**kwargs):
        calls["count"] += 1
        result_dir = Path(str(kwargs["result_path"])).parent
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "resolved": ["task-a"],
                    "not_resolved": [],
                    "total_instances": 1,
                    "total_resolved": 1,
                    "total_unresolved": 0,
                    "total_empty_patch_instances": 0,
                    "generation": ["task-a"],
                    "no_generation": [],
                    "patch_applied": ["task-a"],
                    "with_logs": ["task-a"],
                    "error_ids": [],
                }
            ),
            encoding="utf-8",
        )

    docker_utils_mod.DockerManager = DummyManager
    docker_utils_mod.docker = types.SimpleNamespace(errors=types.SimpleNamespace(BuildError=RuntimeError))
    run_eval_mod.DockerManager = DummyManager
    run_eval_mod.evaluate_predictions = fake_evaluate_predictions

    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.docker_utils", docker_utils_mod)
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python", "patch": "", "test_patch": "", "repo": "owner/repo", "base_commit": "abc", "Dockerfile": "FROM python:3.11", "F2P": [], "P2P": [], "test_command": "pytest", "modified_nodes": []}], ["instance_id", "language", "patch", "test_patch", "repo", "base_commit", "Dockerfile", "F2P", "P2P", "test_command", "modified_nodes"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"
    instance_dir = tmp_path / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "result.json").write_text(
        json.dumps(module._error_result_payload("task-a")),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    assert module.main() == 0
    assert calls["count"] == 1
    assert json.loads((instance_dir / "result.json").read_text(encoding="utf-8"))["resolved"] == ["task-a"]
def test_poly_wrapper_reruns_existing_success_result(monkeypatch, tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_rerun_success")

    docker_utils_mod = types.ModuleType("poly_bench_evaluation.docker_utils")
    run_eval_mod = types.ModuleType("poly_bench_evaluation.run_evaluation")

    class DummyManager:
        pass

    calls = {"count": 0}

    def fake_evaluate_predictions(**kwargs):
        calls["count"] += 1
        result_dir = Path(str(kwargs["result_path"])).parent
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "resolved": [],
                    "not_resolved": ["task-a"],
                    "total_instances": 1,
                    "total_resolved": 0,
                    "total_unresolved": 1,
                    "total_empty_patch_instances": 0,
                    "generation": ["task-a"],
                    "no_generation": [],
                    "patch_applied": [],
                    "with_logs": ["task-a"],
                    "error_ids": [],
                }
            ),
            encoding="utf-8",
        )

    docker_utils_mod.DockerManager = DummyManager
    docker_utils_mod.docker = types.SimpleNamespace(errors=types.SimpleNamespace(BuildError=RuntimeError))
    run_eval_mod.DockerManager = DummyManager
    run_eval_mod.evaluate_predictions = fake_evaluate_predictions

    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.docker_utils", docker_utils_mod)
    monkeypatch.setitem(sys.modules, "poly_bench_evaluation.run_evaluation", run_eval_mod)
    monkeypatch.setattr(module, "_load_dataset_rows", lambda dataset_name: ([{"instance_id": "task-a", "language": "Python", "patch": "", "test_patch": "", "repo": "owner/repo", "base_commit": "abc", "Dockerfile": "FROM python:3.11", "F2P": [], "P2P": [], "test_command": "pytest", "modified_nodes": []}], ["instance_id", "language", "patch", "test_patch", "repo", "base_commit", "Dockerfile", "F2P", "P2P", "test_command", "modified_nodes"]))
    monkeypatch.setattr(module, "_cleanup_stale_poly_containers", lambda selected_rows, client=None: None)

    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    result_path = tmp_path / "evaluation_results"
    instance_dir = tmp_path / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "result.json").write_text(
        json.dumps(
            {
                "resolved": ["task-a"],
                "not_resolved": [],
                "total_instances": 1,
                "total_resolved": 1,
                "total_unresolved": 0,
                "total_empty_patch_instances": 0,
                "generation": ["task-a"],
                "no_generation": [],
                "patch_applied": ["task-a"],
                "with_logs": ["task-a"],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "parse_args", lambda: types.SimpleNamespace(dataset_name="AmazonScience/SWE-PolyBench", predictions_path=pred, result_path=result_path, num_threads=1))

    assert module.main() == 0
    assert calls["count"] == 1
    assert json.loads((instance_dir / "result.json").read_text(encoding="utf-8"))["not_resolved"] == ["task-a"]
def test_poly_wrapper_does_not_patch_docker_build_or_dockerfile_content(tmp_path: Path) -> None:
    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_verbose_build")

    del tmp_path
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_install_verbose_docker_build" not in source
    assert "_patch_python_dockerfile" not in source
    assert "try_pull_prebuilt_image" not in source
def test_poly_wrapper_copies_nested_diagnostic_logs(tmp_path: Path) -> None:
    source_dir = tmp_path / "instances" / "task-a"
    build_log = source_dir / "build_logs" / "task-a_build.log"
    build_log.parent.mkdir(parents=True)
    build_log.write_text("build details\n", encoding="utf-8")

    module = _load_module("contextbench/run_suites_resolution_wrappers/polybench.py", "poly_resolution_wrapper_diagnostics")

    module._copy_diagnostic_files(source_dir=source_dir, work_dir=tmp_path)

    assert (tmp_path / "build_logs" / "task-a_build.log").read_text(encoding="utf-8") == "build details\n"
