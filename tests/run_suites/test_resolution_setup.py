
from __future__ import annotations

import csv
import importlib.util
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

def test_run_resolution_evaluation_passes_absolute_predictions_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    predictions_path = Path("variant") / "resolution-preds" / "poly.jsonl"
    predictions_path.parent.mkdir(parents=True)
    predictions_path.write_text(
        json.dumps({"instance_id": "task-a", "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"})
        + "\n",
        encoding="utf-8",
    )
    work_dir = Path("variant") / "resolution-eval" / "poly"
    work_dir.mkdir(parents=True)
    instance_dir = work_dir / "instances" / "task-a"
    instance_dir.mkdir(parents=True)
    (instance_dir / "report.json").write_text(json.dumps({"resolved_ids": ["task-a"], "unresolved_ids": []}), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["log_path"] = log_path
        captured["log_prefix"] = log_prefix
        return 0, "ok"

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._run_resolution_command", fake_run_resolution_command)

    summary = run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="AmazonScience/SWE-PolyBench",
        run_id="demo-run",
        work_dir=work_dir,
        max_workers=1,
    )

    command = captured["command"]
    assert isinstance(command, list)
    pred_index = command.index("--predictions_path")
    assert Path(command[pred_index + 1]).is_absolute()
    assert command[:2] == [str(postprocess._swe_bench_python_executable()), str(postprocess._SWEBENCH_RESOLUTION_WRAPPER)]
    assert captured["cwd"] == instance_dir.resolve()
    assert captured["log_path"] == (instance_dir.resolve() / "resolution-command.log")
    assert str(captured["log_prefix"]).startswith("[resolution:")
    assert summary["resolved_ids"] == ["task-a"]


def test_describe_resolution_backend_support_marks_poly_pro_and_multi_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr(
        "contextbench.run_suites_core.postprocess._module_available_with_python",
        lambda module_name, python_executable: module_name == "swebench.harness.run_evaluation",
    )
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._PRO_BENCH_EVALUATOR", Path("/missing/probench/evaluator.py"))

    support = describe_resolution_backend_support(["Verified", "Poly", "Pro", "Multi"])
    by_bench = {row["bench"]: row for row in support}

    assert by_bench["Verified"]["backend"] == "swebench"
    assert by_bench["Verified"]["status"] == "available"
    assert by_bench["Poly"]["backend"] == "swe-polybench"
    assert by_bench["Poly"]["status"] == "backend_unavailable"
    assert by_bench["Pro"]["backend"] == "swebench-pro"
    assert by_bench["Pro"]["status"] == "backend_unavailable"
    assert by_bench["Multi"]["backend"] == "multi-swebench"
    assert by_bench["Multi"]["status"] == "backend_unavailable"


@pytest.mark.parametrize(
    ("bench", "python_constant", "run_function", "dataset_name"),
    [
        ("Verified", "_DEFAULT_SWE_BENCH_PYTHON", "run_resolution_evaluation", "princeton-nlp/SWE-bench_Verified"),
        ("Poly", "_DEFAULT_POLY_BENCH_PYTHON", "run_poly_resolution_evaluation", "AmazonScience/SWE-PolyBench"),
        ("Pro", "_PRO_BENCH_PYTHON", "run_pro_resolution_evaluation", "ScaleAI/SWE-bench_Pro"),
        ("Multi", "_DEFAULT_MULTI_BENCH_PYTHON", "run_multi_resolution_evaluation", "bytedance-research/Multi-SWE-Bench"),
    ],
)
def test_resolution_backend_commands_use_repo_local_python(
    monkeypatch,
    tmp_path: Path,
    bench: str,
    python_constant: str,
    run_function: str,
    dataset_name: str,
) -> None:
    expected_python = tmp_path / ".cache" / f"{bench.lower()}-eval-venv" / "bin" / "python"
    monkeypatch.setattr(postprocess, python_constant, expected_python)
    monkeypatch.setenv("CONTEXTBENCH_EVALUATOR_PYTHON", str(tmp_path / "host-python"))
    captured: dict[str, object] = {}

    predictions_path = tmp_path / f"{bench.lower()}-predictions.jsonl"
    if bench == "Pro":
        predictions_path = predictions_path.with_suffix(".json")
        predictions_path.write_text(
            json.dumps([{"instance_id": "instance_repo__repo-1", "patch": "diff --git a/a.py b/a.py\n"}]),
            encoding="utf-8",
        )
    elif bench == "Multi":
        predictions_path.write_text(
            json.dumps({"org": "owner", "repo": "repo", "number": 1, "fix_patch": "diff --git a/a.py b/a.py\n"})
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            postprocess,
            "_write_multi_dataset_jsonl",
            lambda *, dataset_name, instance_ids, out_path: out_path.write_text(
                '{"org":"owner","repo":"repo","number":1,"instance_id":"owner__repo-1"}\n',
                encoding="utf-8",
            ),
        )
    else:
        predictions_path.write_text(
            json.dumps({"instance_id": "task-a", "model_patch": "diff --git a/a.py b/a.py\n"}) + "\n",
            encoding="utf-8",
        )

    def fake_run_resolution_command(*, command, cwd, log_path, log_prefix, env=None):
        del log_path, log_prefix, env
        captured["command"] = list(command)
        if bench == "Verified":
            (Path(cwd) / "report.json").write_text(
                json.dumps({"resolved_ids": ["task-a"], "unresolved_ids": []}),
                encoding="utf-8",
            )
        elif bench == "Poly":
            (Path(cwd) / "result.json").write_text(
                json.dumps({"resolved": ["task-a"], "not_resolved": [], "total_resolved": 1}),
                encoding="utf-8",
            )
        elif bench == "Pro":
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "eval_results.json").write_text(
                json.dumps({"instance_repo__repo-1": True}),
                encoding="utf-8",
            )
        elif bench == "Multi":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "final_report.json").write_text(
                json.dumps({"resolved_ids": ["owner__repo-1"], "unresolved_ids": [], "error_ids": []}),
                encoding="utf-8",
            )
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = getattr(postprocess, run_function)(
        predictions_path=predictions_path,
        dataset_name=dataset_name,
        run_id=f"{bench.lower()}-run",
        work_dir=tmp_path / f"{bench.lower()}-work",
        max_workers=1,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(expected_python)
    assert summary["python_executable"] == str(expected_python)
    assert command[0] != str(tmp_path / "host-python")


def test_backend_python_path_keeps_venv_symlink(tmp_path: Path) -> None:
    real_python = tmp_path / "homebrew" / "python3.11"
    venv_python = tmp_path / ".cache" / "polybench-eval-venv" / "bin" / "python"
    real_python.parent.mkdir(parents=True)
    real_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)

    assert postprocess._absolute_without_resolving_symlinks(venv_python) == venv_python
    assert postprocess._absolute_without_resolving_symlinks(venv_python) != venv_python.resolve()


def test_swebench_wrapper_imports_installed_swebench_package(tmp_path: Path, monkeypatch) -> None:
    fake_package = tmp_path / "fake-package"
    run_evaluation = fake_package / "swebench" / "harness" / "run_evaluation.py"
    run_evaluation.parent.mkdir(parents=True)
    (fake_package / "swebench" / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "swebench" / "harness" / "__init__.py").write_text("", encoding="utf-8")
    run_evaluation.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def main(**kwargs):\n"
        "    report_dir = Path(kwargs['report_dir'])\n"
        "    instance_ids = kwargs['instance_ids']\n"
        "    report_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (report_dir / 'codex.fake.json').write_text(json.dumps({\n"
        "        'resolved_ids': instance_ids,\n"
        "        'unresolved_ids': [],\n"
        "        'error_ids': [],\n"
        "        'completed_ids': instance_ids,\n"
        "        'submitted_ids': instance_ids,\n"
        "        'total_instances': len(instance_ids),\n"
        "        'completed_instances': len(instance_ids),\n"
        "        'error_instances': 0,\n"
        "    }))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(fake_package))
    monkeypatch.syspath_prepend(str(postprocess._SWEBENCH_RESOLUTION_WRAPPER.parent))

    spec = importlib.util.spec_from_file_location("contextbench_swebench_wrapper_import_test", postprocess._SWEBENCH_RESOLUTION_WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text('{"instance_id":"task-a","model_patch":"diff --git a/a.py b/a.py\\n"}\n', encoding="utf-8")
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "dataset_name": "princeton-nlp/SWE-bench_Verified",
                "predictions_path": str(predictions_path),
                "max_workers": 1,
                "run_id": "demo-run",
                "timeout": 60,
                "report_dir": str(report_dir),
            },
        )(),
    )

    assert module.main() == 0
    assert json.loads((report_dir / "report.json").read_text(encoding="utf-8"))["resolved_ids"] == ["task-a"]
