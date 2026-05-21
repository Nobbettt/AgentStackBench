# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
from contextbench.run_suites_core.postprocess import evaluate_resolution_for_suite, run_resolution_evaluation

from .helpers import write_agent_resolution_record


def test_evaluate_resolution_for_suite_counts_evaluated_swebench_timeout_as_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    variant_dir = tmp_path / "variant"
    source_dir = variant_dir / "agent_runs" / "codex"
    instance_id = "django__django-15022"
    write_agent_resolution_record(variant_dir=variant_dir, bench="Verified", instance_id=instance_id)

    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_available", lambda: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_image_available", lambda image: True)
    monkeypatch.setattr("contextbench.run_suites_core.postprocess._docker_host_socket_path", lambda: Path("/var/run/docker.sock"))

    def fake_swebench_run(**kwargs):
        del kwargs
        return {
            "resolved_ids": [],
            "unresolved_ids": [instance_id],
            "error_ids": [],
            "evaluation_error_ids": [instance_id],
            "test_timeout_ids": [instance_id],
            "resolved_count": 0,
            "total_instances": 1,
        }

    monkeypatch.setattr("contextbench.run_suites_core.postprocess.run_resolution_evaluation", fake_swebench_run)

    summary = evaluate_resolution_for_suite(
        source_dir=source_dir,
        expected_agent="codex",
        suite_name="demo-suite",
        variant_name="baseline",
        work_dir=variant_dir,
        max_workers=1,
    )

    bench_summary = summary["per_bench"]["Verified"]
    assert summary["status"] == "completed"
    assert summary["failed_benches"] == []
    assert summary["is_partial"] is False
    assert summary["evaluated_task_count"] == 1
    assert summary["pass_at_1"] == 0.0
    assert bench_summary["status"] == "completed"
    assert bench_summary["is_partial"] is False
    assert bench_summary["evaluation_error_ids"] == [instance_id]
    assert bench_summary["test_timeout_ids"] == [instance_id]


def test_run_resolution_evaluation_counts_swebench_test_timeout_as_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_id = "django__django-15022"
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}) + "\n",
        encoding="utf-8",
    )

    def fake_run_resolution_command(*, cwd: Path, **_: object) -> tuple[int, str]:
        run_log = cwd / "logs" / "run_evaluation" / "run" / "codex" / instance_id / "run_instance.log"
        run_log.parent.mkdir(parents=True)
        run_log.write_text(
            "Applied Patch:\n"
            "Test runtime: 7_200.79 seconds\n"
            f"{instance_id}: Test timed out after 7200 seconds.\n",
            encoding="utf-8",
        )
        (run_log.parent / "test_output.txt").write_text(
            ">>>>> Start Test Output\n"
            "test_many_search_terms (admin_changelist.tests.ChangeListTests) ... \n"
            "Timeout error: 7200 seconds exceeded.\n",
            encoding="utf-8",
        )
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": [], "unresolved_ids": [], "error_ids": [instance_id]}),
            encoding="utf-8",
        )
        return 0, ""

    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=tmp_path / "resolution",
        max_workers=1,
        swebench_timeout=7200,
    )

    assert summary["resolved_ids"] == []
    assert summary["unresolved_ids"] == [instance_id]
    assert summary["error_ids"] == []
    assert summary["evaluation_error_ids"] == [instance_id]
    assert summary["test_timeout_ids"] == [instance_id]
    assert summary["total_instances"] == 1
    instance_summary = json.loads(
        (tmp_path / "resolution" / "instances" / instance_id / "resolution-result.json").read_text(encoding="utf-8")
    )
    assert instance_summary["status"] == "unresolved"
    assert instance_summary["resolution_failure_kind"] == "test_timeout"


def test_swebench_timeout_classifier_handles_docker_stopped_after_timeout(tmp_path: Path) -> None:
    instance_id = "django__django-15022"
    instance_dir = tmp_path / "instances" / instance_id
    log_dir = instance_dir / "logs" / "run_evaluation" / "run" / "codex" / instance_id
    log_dir.mkdir(parents=True)
    (log_dir / "run_instance.log").write_text(
        "2026-05-08 15:40:21,759 - INFO - >>>>> Applied Patch:\n"
        "Applied patch django/contrib/admin/options.py cleanly.\n"
        "2026-05-08 23:48:12,718 - INFO - Test runtime: 29_264.39 seconds\n"
        "2026-05-08 23:48:12,950 - ERROR - Error in evaluating model for django__django-15022: "
        '409 Client Error: Conflict ("container abc is not running")\n',
        encoding="utf-8",
    )
    (log_dir / "test_output.txt").write_text(
        "Applied patch tests/admin_changelist/tests.py cleanly.\n"
        "+ : '>>>>> Start Test Output'\n"
        "+ ./tests/runtests.py --verbosity 2 --settings=test_sqlite admin_changelist.tests\n"
        "test_many_search_terms (admin_changelist.tests.ChangeListTests) ... ",
        encoding="utf-8",
    )
    summary = {
        "instance_id": instance_id,
        "resolved_ids": [],
        "unresolved_ids": [],
        "error_ids": [instance_id],
        "status": "error",
        "input_metadata": {"backend": "swebench", "harness_args": ["--timeout", "7200"]},
    }

    normalized = postprocess._normalize_swebench_instance_summary(summary, instance_dir=instance_dir)

    assert normalized["status"] == "unresolved"
    assert normalized["unresolved_ids"] == [instance_id]
    assert normalized["error_ids"] == []
    assert normalized["test_timeout_ids"] == [instance_id]


def test_swebench_timeout_classifier_keeps_short_docker_stop_as_infra_error(tmp_path: Path) -> None:
    instance_id = "django__django-15022"
    instance_dir = tmp_path / "instances" / instance_id
    log_dir = instance_dir / "logs" / "run_evaluation" / "run" / "codex" / instance_id
    log_dir.mkdir(parents=True)
    (log_dir / "run_instance.log").write_text(
        "2026-05-08 15:40:21,759 - INFO - >>>>> Applied Patch:\n"
        "Applied patch django/contrib/admin/options.py cleanly.\n"
        "2026-05-08 15:50:12,718 - INFO - Test runtime: 60.00 seconds\n"
        "2026-05-08 15:50:12,950 - ERROR - Error in evaluating model for django__django-15022: "
        '409 Client Error: Conflict ("container abc is not running")\n',
        encoding="utf-8",
    )
    (log_dir / "test_output.txt").write_text(
        "+ : '>>>>> Start Test Output'\n"
        "+ ./tests/runtests.py --verbosity 2 --settings=test_sqlite admin_changelist.tests\n",
        encoding="utf-8",
    )
    summary = {
        "instance_id": instance_id,
        "resolved_ids": [],
        "unresolved_ids": [],
        "error_ids": [instance_id],
        "status": "error",
        "input_metadata": {"backend": "swebench", "harness_args": ["--timeout", "7200"]},
    }

    normalized = postprocess._normalize_swebench_instance_summary(summary, instance_dir=instance_dir)

    assert normalized == summary


def test_run_resolution_evaluation_passes_swebench_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": "psf__requests-1000", "model_patch": "diff --git a/a.py b/a.py\n"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_resolution_command(*, command: list[str], cwd: Path, **_: object) -> tuple[int, str]:
        captured["command"] = command
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": ["psf__requests-1000"], "unresolved_ids": [], "error_ids": []}),
            encoding="utf-8",
        )
        return 0, ""

    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=tmp_path / "resolution",
        max_workers=1,
        swebench_timeout=7200,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command.count("--timeout") == 1
    assert command[command.index("--timeout") + 1] == "7200"
    assert summary["swebench_timeout"] == 7200
    instance_summary = json.loads(
        (tmp_path / "resolution" / "instances" / "psf__requests-1000" / "resolution-result.json").read_text(encoding="utf-8")
    )
    assert instance_summary["input_metadata"]["harness_args"] == ["--timeout", "7200"]


@pytest.mark.parametrize("harness_args", [["--timeout", "60"], ["--timeout=60"]])
def test_run_resolution_evaluation_rejects_timeout_in_harness_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harness_args: list[str],
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": "psf__requests-1000", "model_patch": "diff --git a/a.py b/a.py\n"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(
        postprocess,
        "_run_resolution_command",
        lambda **_kwargs: pytest.fail("invalid timeout configuration must fail before evaluator execution"),
    )

    with pytest.raises(ValueError, match="postprocess.swebench_timeout"):
        run_resolution_evaluation(
            predictions_path=predictions_path,
            dataset_name="princeton-nlp/SWE-bench_Verified",
            run_id="demo",
            work_dir=tmp_path / "resolution",
            max_workers=1,
            harness_args=harness_args,
            swebench_timeout=7200,
        )
