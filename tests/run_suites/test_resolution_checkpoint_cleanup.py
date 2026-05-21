# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
from contextbench.coding_agents.files import safe_path_component


def _write_checkpoint(
    *,
    cache_dir: Path,
    instance_id: str,
    input_metadata: dict[str, object],
) -> None:
    checkpoint_dir = cache_dir / "instances" / safe_path_component(instance_id)
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "resolution-result.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "resolved_ids": [instance_id],
                "unresolved_ids": [],
                "error_ids": [],
                "status": "resolved",
                "input_metadata": input_metadata,
            }
        ),
        encoding="utf-8",
    )


def test_run_resolution_evaluation_reuses_checkpoint_after_eval_dir_cleanup(tmp_path: Path, monkeypatch) -> None:
    instance_id = "psf__requests-1000"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "resolution-checkpoints" / "verified"
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    input_metadata = postprocess._resolution_instance_input_metadata(
        prediction,
        backend=postprocess._resolution_backend_for_bench("Verified"),
        dataset_name="princeton-nlp/SWE-bench_Verified",
        harness_args=["--timeout", "1800"],
    )
    checkpoint_dir = cache_dir / "instances" / safe_path_component(instance_id)
    checkpoint_dir.mkdir(parents=True)
    checkpoint_summary_path = checkpoint_dir / "resolution-result.json"
    checkpoint_summary_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "resolved_ids": [instance_id],
                "unresolved_ids": [],
                "error_ids": [],
                "status": "resolved",
                "input_metadata": input_metadata,
            }
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path("/python"))
    monkeypatch.setattr(
        postprocess,
        "_run_resolution_command",
        lambda **_kwargs: pytest.fail("cached resolution summary should avoid rerunning the evaluator"),
    )

    summary = postprocess.run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
        cache_dir=cache_dir,
    )

    instance_dir = work_dir / "instances" / safe_path_component(instance_id)
    log_text = (work_dir / "resolution-command.log").read_text(encoding="utf-8")
    assert summary["resolved_ids"] == [instance_id]
    assert checkpoint_summary_path.exists()
    assert not instance_dir.exists()
    assert f"[reuse] {instance_id} -> {checkpoint_summary_path}" in log_text
    assert "[cleanup] removed checkpointed resolution artifacts" in log_text


def test_run_resolution_evaluation_cleans_checkpointed_instance_artifacts(tmp_path: Path, monkeypatch) -> None:
    instance_id = "psf__requests-1000"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "resolution-checkpoints" / "verified"
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path("/python"))

    def fake_run_resolution_command(*, cwd: Path, log_path: Path, **_kwargs):
        (cwd / "large-workspace").mkdir(parents=True)
        (cwd / "large-workspace" / "artifact.bin").write_text("temporary evaluator output", encoding="utf-8")
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": [instance_id], "unresolved_ids": [], "error_ids": []}),
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
        cache_dir=cache_dir,
    )

    checkpoint_summary = cache_dir / "instances" / safe_path_component(instance_id) / "resolution-result.json"
    instance_dir = work_dir / "instances" / safe_path_component(instance_id)
    assert summary["resolved_ids"] == [instance_id]
    assert checkpoint_summary.exists()
    assert not instance_dir.exists()
    assert "[cleanup] removed checkpointed resolution artifacts" in (work_dir / "resolution-command.log").read_text(
        encoding="utf-8"
    )


def test_run_resolution_evaluation_can_keep_checkpointed_instance_artifacts(tmp_path: Path, monkeypatch) -> None:
    instance_id = "psf__requests-1000"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "resolution-checkpoints" / "verified"
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path("/python"))

    def fake_run_resolution_command(*, cwd: Path, log_path: Path, **_kwargs):
        (cwd / "large-workspace").mkdir(parents=True)
        (cwd / "large-workspace" / "artifact.bin").write_text("temporary evaluator output", encoding="utf-8")
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": [instance_id], "unresolved_ids": [], "error_ids": []}),
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return 0, "ok"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
        cache_dir=cache_dir,
        self_clean_resolution_artifacts=False,
    )

    checkpoint_summary = cache_dir / "instances" / safe_path_component(instance_id) / "resolution-result.json"
    instance_dir = work_dir / "instances" / safe_path_component(instance_id)
    assert summary["resolved_ids"] == [instance_id]
    assert checkpoint_summary.exists()
    assert (instance_dir / "large-workspace" / "artifact.bin").exists()
    assert "[cleanup] removed checkpointed resolution artifacts" not in (work_dir / "resolution-command.log").read_text(
        encoding="utf-8"
    )


def test_failed_swebench_return_keeps_artifacts_and_does_not_checkpoint(tmp_path: Path, monkeypatch) -> None:
    instance_id = "psf__requests-1000"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "resolution-checkpoints" / "verified"
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(postprocess, "_swe_bench_python_executable", lambda: Path("/python"))

    def fake_run_resolution_command(*, cwd: Path, log_path: Path, **_kwargs):
        (cwd / "large-workspace").mkdir(parents=True)
        (cwd / "large-workspace" / "artifact.bin").write_text("debug me", encoding="utf-8")
        (cwd / "report.json").write_text(
            json.dumps({"resolved_ids": [instance_id], "unresolved_ids": [], "error_ids": []}),
            encoding="utf-8",
        )
        log_path.write_text("failed after report\n", encoding="utf-8")
        return 1, "failed after report"

    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)

    summary = postprocess.run_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
        cache_dir=cache_dir,
    )

    instance_dir = work_dir / "instances" / safe_path_component(instance_id)
    checkpoint_summary = cache_dir / "instances" / safe_path_component(instance_id) / "resolution-result.json"
    assert summary["_partial_from_error"] is True
    assert summary["resolved_ids"] == [instance_id]
    assert (instance_dir / "large-workspace" / "artifact.bin").exists()
    assert (instance_dir / "resolution-command.log").exists()
    assert not checkpoint_summary.exists()
    assert "[cleanup] removed checkpointed resolution artifacts" not in (work_dir / "resolution-command.log").read_text(
        encoding="utf-8"
    )


def test_failed_poly_return_keeps_artifacts_and_skips_docker_cleanup(tmp_path: Path, monkeypatch) -> None:
    instance_id = "mui__material-ui-42412"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "resolution-checkpoints" / "poly"
    prediction = {"instance_id": instance_id, "model_patch": "diff --git a/a.py b/a.py\n"}
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    def fake_run_resolution_command(*, cwd: Path, log_path: Path, **_kwargs):
        (cwd / "large-workspace").mkdir(parents=True)
        (cwd / "large-workspace" / "artifact.bin").write_text("debug me", encoding="utf-8")
        (cwd / "result.json").write_text(
            json.dumps({"resolved": [instance_id], "not_resolved": [], "error_ids": []}),
            encoding="utf-8",
        )
        log_path.write_text("failed after report\n", encoding="utf-8")
        return 1, "failed after report"

    monkeypatch.setattr(postprocess, "_poly_bench_python_executable", lambda: Path("/python"))
    monkeypatch.setattr(postprocess, "_run_resolution_command", fake_run_resolution_command)
    monkeypatch.setattr(
        postprocess,
        "_cleanup_checkpointed_resolution_docker_images",
        lambda **_kwargs: pytest.fail("docker cleanup must not run after a failed evaluator return"),
    )

    summary = postprocess.run_poly_resolution_evaluation(
        predictions_path=predictions_path,
        dataset_name="AmazonScience/SWE-PolyBench",
        run_id="demo",
        work_dir=work_dir,
        max_workers=1,
        cache_dir=cache_dir,
    )

    instance_dir = work_dir / "instances" / safe_path_component(instance_id)
    checkpoint_summary = cache_dir / "instances" / safe_path_component(instance_id) / "resolution-result.json"
    assert summary["_partial_from_error"] is True
    assert summary["resolved_ids"] == [instance_id]
    assert (instance_dir / "large-workspace" / "artifact.bin").exists()
    assert (instance_dir / "resolution-command.log").exists()
    assert not checkpoint_summary.exists()


def test_checkpointed_multi_docker_cleanup_removes_only_instance_pr_image(tmp_path: Path, monkeypatch) -> None:
    instance_id = "clap-rs__clap-1869"
    cache_dir = tmp_path / "checkpoints" / "multi"
    log_path = tmp_path / "resolution-command.log"
    backend = postprocess._resolution_backend_for_bench("Multi")
    input_metadata = {"backend": "multi-swebench", "prediction_sha256": "demo"}
    removed: list[str] = []
    _write_checkpoint(cache_dir=cache_dir, instance_id=instance_id, input_metadata=input_metadata)

    monkeypatch.setattr(
        postprocess,
        "_docker_image_references",
        lambda: [
            "mswebench/clap-rs_m_clap:pr-1869",
            "mswebench/clap-rs_m_clap:base",
            "contextbench-postprocess:2026-04-23",
        ],
    )
    monkeypatch.setattr(postprocess, "_docker_active_image_references", lambda: set())
    monkeypatch.setattr(postprocess, "_remove_docker_image_ref", lambda image_ref: removed.append(image_ref) or (True, ""))

    cleaned = postprocess._cleanup_checkpointed_resolution_docker_images(
        instance_id=instance_id,
        backend=backend,
        cache_dir=cache_dir,
        input_metadata=input_metadata,
        log_path=log_path,
    )

    assert cleaned == ["mswebench/clap-rs_m_clap:pr-1869"]
    assert removed == ["mswebench/clap-rs_m_clap:pr-1869"]
    assert "base" not in log_path.read_text(encoding="utf-8")


def test_checkpointed_poly_docker_cleanup_removes_matching_instance_images(tmp_path: Path, monkeypatch) -> None:
    instance_id = "mui__material-ui-42412"
    cache_dir = tmp_path / "checkpoints" / "poly"
    log_path = tmp_path / "resolution-command.log"
    backend = postprocess._resolution_backend_for_bench("Poly")
    input_metadata = {"backend": "swe-polybench", "prediction_sha256": "demo"}
    removed: list[str] = []
    _write_checkpoint(cache_dir=cache_dir, instance_id=instance_id, input_metadata=input_metadata)

    monkeypatch.setattr(
        postprocess,
        "_docker_image_references",
        lambda: [
            "polybench_typescript_mui__material-ui-42412:latest",
            "ghcr.io/timesler/swe-polybench.eval.x86_64.mui__material-ui-42412:latest",
            "polybench_typescript_base:latest",
            "polybench_typescript_mui__material-ui-99999:latest",
        ],
    )
    monkeypatch.setattr(postprocess, "_docker_active_image_references", lambda: set())
    monkeypatch.setattr(postprocess, "_remove_docker_image_ref", lambda image_ref: removed.append(image_ref) or (True, ""))

    cleaned = postprocess._cleanup_checkpointed_resolution_docker_images(
        instance_id=instance_id,
        backend=backend,
        cache_dir=cache_dir,
        input_metadata=input_metadata,
        log_path=log_path,
    )

    assert cleaned == [
        "ghcr.io/timesler/swe-polybench.eval.x86_64.mui__material-ui-42412:latest",
        "polybench_typescript_mui__material-ui-42412:latest",
    ]
    assert removed == cleaned


def test_checkpointed_docker_cleanup_skips_active_image(tmp_path: Path, monkeypatch) -> None:
    instance_id = "clap-rs__clap-1869"
    image_ref = "mswebench/clap-rs_m_clap:pr-1869"
    cache_dir = tmp_path / "checkpoints" / "multi"
    log_path = tmp_path / "resolution-command.log"
    backend = postprocess._resolution_backend_for_bench("Multi")
    input_metadata = {"backend": "multi-swebench", "prediction_sha256": "demo"}
    _write_checkpoint(cache_dir=cache_dir, instance_id=instance_id, input_metadata=input_metadata)

    monkeypatch.setattr(postprocess, "_docker_image_references", lambda: [image_ref])
    monkeypatch.setattr(postprocess, "_docker_active_image_references", lambda: {image_ref})
    monkeypatch.setattr(
        postprocess,
        "_remove_docker_image_ref",
        lambda _image_ref: pytest.fail("active image must not be removed"),
    )

    cleaned = postprocess._cleanup_checkpointed_resolution_docker_images(
        instance_id=instance_id,
        backend=backend,
        cache_dir=cache_dir,
        input_metadata=input_metadata,
        log_path=log_path,
    )

    assert cleaned == []
    assert f"[docker-cleanup] skipped active image {image_ref}" in log_path.read_text(encoding="utf-8")


def test_checkpointed_docker_cleanup_respects_toggle(tmp_path: Path, monkeypatch) -> None:
    instance_id = "clap-rs__clap-1869"
    cache_dir = tmp_path / "checkpoints" / "multi"
    log_path = tmp_path / "resolution-command.log"
    backend = postprocess._resolution_backend_for_bench("Multi")
    input_metadata = {"backend": "multi-swebench", "prediction_sha256": "demo"}
    _write_checkpoint(cache_dir=cache_dir, instance_id=instance_id, input_metadata=input_metadata)

    monkeypatch.setattr(
        postprocess,
        "_docker_image_references",
        lambda: pytest.fail("disabled docker cleanup must not query Docker"),
    )

    cleaned = postprocess._cleanup_checkpointed_resolution_docker_images(
        instance_id=instance_id,
        backend=backend,
        cache_dir=cache_dir,
        input_metadata=input_metadata,
        log_path=log_path,
        enabled=False,
    )

    assert cleaned == []
    assert not log_path.exists()
