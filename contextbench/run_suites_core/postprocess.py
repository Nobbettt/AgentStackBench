# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-05-21 from upstream ContextBench.
# Summary of changes: add fork run-suite adapters, parallel resolution evaluation, image prebuilds, and resilient checkpoints.

"""Conversion and evaluation helpers for run suites."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..artifact_sanitization import SanitizationContext, sanitize_text
from ..coding_agents.conversion import (
    ContextPathValidationError,
    convert_run_record,
    load_predictions_from_path,
    record_is_convertible,
    record_with_resolved_artifact_paths,
    resolve_record_path,
)
from ..coding_agents.files import ensure_dir, read_json, read_jsonl, safe_path_component, write_json
from ..coding_agents.verification_quality import analyze_record_quality
from ..evaluate import GoldLoader, aggregate_results, evaluate_instance
from ..extractors import available as treesitter_available
from ..extractors.treesitter import DEF_NODES
from ..parsers import load_pred
from .helpers import stable_json_hash

_BENCH_TO_RESOLUTION_DATASET = {
    "Verified": "princeton-nlp/SWE-bench_Verified",
    "Pro": "ScaleAI/SWE-bench_Pro",
    "Poly": "AmazonScience/SWE-PolyBench",
    "Multi": "bytedance-research/Multi-SWE-Bench",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_PATH_ERRORS_FIELD = "_artifact_path_errors"
_DEFAULT_SWE_BENCH_PYTHON = _REPO_ROOT / ".cache" / "swebench-eval-venv" / "bin" / "python"
_DEFAULT_POLY_BENCH_PYTHON = _REPO_ROOT / ".cache" / "polybench-eval-venv" / "bin" / "python"
_DEFAULT_MULTI_BENCH_PYTHON = _REPO_ROOT / ".cache" / "multibench-eval-venv" / "bin" / "python"
_PRO_BENCH_ROOT = _REPO_ROOT / ".cache" / "probench-eval"
_PRO_BENCH_PYTHON = _REPO_ROOT / ".cache" / "probench-eval-venv" / "bin" / "python"
_PRO_BENCH_EVALUATOR = _PRO_BENCH_ROOT / "swe_bench_pro_eval.py"
_PRO_BENCH_RUN_SCRIPTS = _PRO_BENCH_ROOT / "run_scripts"
_PRO_BENCH_DOCKERFILES = _PRO_BENCH_ROOT / "dockerfiles"
_PRO_BENCH_RAW_SAMPLE_JSONL = _PRO_BENCH_ROOT / "helper_code" / "sweap_eval_full_v2.jsonl"
_PRO_BENCH_DOCKERHUB_USERNAME = "jefzda"
_SWEBENCH_RESOLUTION_WRAPPER = _REPO_ROOT / "contextbench" / "run_suites_resolution_wrappers" / "swebench_wrapper.py"
_POLYBENCH_RESOLUTION_WRAPPER = _REPO_ROOT / "contextbench" / "run_suites_resolution_wrappers" / "polybench.py"
_PROBENCH_RESOLUTION_WRAPPER = _REPO_ROOT / "contextbench" / "run_suites_resolution_wrappers" / "probench.py"
_MULTIBENCH_RESOLUTION_WRAPPER = _REPO_ROOT / "contextbench" / "run_suites_resolution_wrappers" / "multibench.py"
_PRO_REQUIRED_SAMPLE_COLUMNS = (
    "instance_id",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "base_commit",
    "repo",
    "fail_to_pass",
    "pass_to_pass",
)
_POLY_REQUIRED_SAMPLE_COLUMNS = (
    "instance_id",
    "patch",
    "test_patch",
    "repo",
    "base_commit",
    "language",
    "Dockerfile",
    "F2P",
    "P2P",
    "test_command",
    "modified_nodes",
)

_SENSITIVE_ENV_NAME_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)
_RESOLUTION_HEARTBEAT_INTERVAL_SECONDS = 60.0
_RESOLUTION_INPUT_METADATA_VERSION = 1
_POSTPROCESS_REQUIRED_PARSER_LANGUAGES = (
    "c",
    "c_sharp",
    "cpp",
    "go",
    "java",
    "javascript",
    "python",
    "rust",
    "tsx",
    "typescript",
)


def _default_evaluation_workspace_key(out_path: Path) -> str:
    resolved = str(out_path.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return safe_path_component(f"eval-{out_path.stem}-{digest}-pid-{os.getpid()}")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _final_prediction_has_context(pred_data: dict[str, object]) -> bool:
    traj_data = pred_data.get("traj_data")
    if not isinstance(traj_data, dict):
        return False
    return bool(
        traj_data.get("pred_files")
        or traj_data.get("pred_spans")
        or traj_data.get("pred_symbols")
    )


def _assert_evaluation_artifact_consistent(
    *,
    results: list[dict[str, object]],
    predictions_by_instance_id: dict[str, dict[str, object]],
) -> None:
    inconsistent_ids = [
        str(row.get("instance_id"))
        for row in results
        if row.get("error") == "no_context_extracted"
        and _final_prediction_has_context(predictions_by_instance_id.get(str(row.get("instance_id"))) or {})
    ]
    if not inconsistent_ids:
        return
    sample = ", ".join(inconsistent_ids[:10])
    suffix = "" if len(inconsistent_ids) <= 10 else f", ... ({len(inconsistent_ids)} total)"
    raise RuntimeError(
        "Evaluation artifact is inconsistent: predictions contain final context but "
        f"evaluation produced no_context_extracted for {sample}{suffix}. "
        "Rerun evaluation with isolated worktrees; this usually indicates a stale or raced evaluation output."
    )


@dataclass(frozen=True)
class ResolutionBackend:
    backend: str
    dataset_name: str | None
    module_name: str | None
    export_format: str
    run_evaluation: Callable[..., dict[str, object]] | None
    python_executable: Path | None = None
    wrapper_path: Path | None = None
    setup_command: str | None = None
    unsupported_reason: str | None = None
    requires_docker: bool = True


@dataclass(frozen=True)
class ResolutionCommandError(RuntimeError):
    message: str
    exit_code: int
    log_path: str
    tail: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class _ResolutionInstanceResult:
    index: int
    summary: dict[str, object]
    error: ResolutionCommandError | None = None


def _run_resolution_instance_jobs(
    *,
    job_count: int,
    max_workers: int,
    run_one: Callable[[int], _ResolutionInstanceResult],
) -> list[_ResolutionInstanceResult]:
    if job_count <= 0:
        return []
    worker_count = max(1, min(int(max_workers or 1), job_count))
    if worker_count == 1:
        return [run_one(index) for index in range(job_count)]

    results: list[_ResolutionInstanceResult | None] = [None] * job_count
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(run_one, index): index for index in range(job_count)}
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _run_resolution_command(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    log_prefix: str,
    env: dict[str, str] | None = None,
    heartbeat_interval_seconds: float = _RESOLUTION_HEARTBEAT_INTERVAL_SECONDS,
    heartbeat_label: str | None = None,
) -> tuple[int, str]:
    ensure_dir(log_path.parent)
    tail: deque[str] = deque(maxlen=40)
    sanitize_context = SanitizationContext(
        repo_root=_REPO_ROOT,
        suite_dir=log_path.parent,
        task_dir=log_path.parent,
        extra_roots=(cwd,),
    )
    with open(log_path, "w", encoding="utf-8") as log_handle:
        command_text = sanitize_text(" ".join(_redact_command_for_log(command)), context=sanitize_context)
        log_handle.write(f"$ {command_text}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, **env} if env else None,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            try:
                for raw_line in process.stdout:
                    output_queue.put(raw_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_stdout, name="contextbench-resolution-output", daemon=True)
        reader.start()
        started = time.monotonic()
        next_heartbeat = (
            started + heartbeat_interval_seconds
            if heartbeat_interval_seconds > 0
            else None
        )
        label = heartbeat_label or _heartbeat_label_for_command(command)
        while True:
            timeout = None
            if next_heartbeat is not None:
                timeout = max(0.1, next_heartbeat - time.monotonic())
            try:
                raw_line = output_queue.get(timeout=timeout)
            except queue.Empty:
                if process.poll() is None:
                    elapsed = int(time.monotonic() - started)
                    heartbeat = (
                        f"[heartbeat] command={label} elapsed={elapsed}s "
                        "no output; subprocess still running"
                    )
                    tail.append(heartbeat)
                    print(f"{log_prefix} {heartbeat}", flush=True)
                    log_handle.write(heartbeat + "\n")
                    log_handle.flush()
                if heartbeat_interval_seconds > 0:
                    next_heartbeat = time.monotonic() + heartbeat_interval_seconds
                continue
            if raw_line is None:
                break
            line = sanitize_text(raw_line.rstrip("\n"), context=sanitize_context)
            tail.append(line)
            print(f"{log_prefix} {line}", flush=True)
            log_handle.write(line + ("\n" if raw_line.endswith("\n") else ""))
            log_handle.flush()
            if heartbeat_interval_seconds > 0:
                next_heartbeat = time.monotonic() + heartbeat_interval_seconds
        returncode = process.wait()
        reader.join(timeout=1)
    return returncode, "\n".join(tail)


def _heartbeat_label_for_command(command: list[str]) -> str:
    for token in command:
        name = Path(str(token)).name
        if name in {"swebench_wrapper.py", "polybench.py", "probench.py", "multibench.py"}:
            return name
    return Path(str(command[0])).name if command else "command"


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SENSITIVE_ENV_NAME_PARTS)


def _redact_env_assignment(value: str) -> str:
    if "=" not in value:
        return value
    key, raw_value = value.split("=", 1)
    if _is_sensitive_env_name(key):
        return f"{key}=<redacted>"
    return f"{key}={raw_value}"


def _redact_command_for_log(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next_env_value = False
    for token in command:
        if redact_next_env_value:
            redacted.append(_redact_env_assignment(token))
            redact_next_env_value = False
            continue
        redacted.append(_redact_env_assignment(token))
        if token in {"-e", "--env", "--environment"}:
            redact_next_env_value = True
    return redacted


def _task_results_for_source_dir(source_dir: Path) -> Path | None:
    candidates = [
        source_dir / "task-results.jsonl",
        source_dir.parent / "task-results.jsonl",
        source_dir.parent.parent / "task-results.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_agent_record_id(record: dict[str, object], row: dict[str, object]) -> str:
    return str(
        record.get("original_inst_id")
        or row.get("original_inst_id")
        or record.get("instance_id")
        or row.get("instance_id")
        or ""
    ).strip()


def _record_matches_agent(record: dict[str, object], expected_agent: str) -> bool:
    raw_agent = str(record.get("agent") or "").strip().lower()
    return not raw_agent or raw_agent == expected_agent


def _normalize_model_patch_for_resolution(raw_patch: object) -> str:
    patch = str(raw_patch or "")
    if not patch.strip():
        return ""
    return patch if patch.endswith("\n") else f"{patch}\n"


def _read_model_patch_for_resolution(record: dict[str, object]) -> str:
    return _normalize_model_patch_for_resolution(record.get("model_patch"))


def _task_result_rows_for_source_dir(source_dir: Path) -> list[dict[str, object]]:
    task_results = _task_results_for_source_dir(source_dir) if source_dir.exists() else None
    if task_results is None:
        return []
    return [row for row in read_jsonl(task_results) if isinstance(row, dict)]


def _swe_bench_python_executable() -> Path:
    return _DEFAULT_SWE_BENCH_PYTHON


def _poly_bench_python_executable() -> Path:
    return _DEFAULT_POLY_BENCH_PYTHON


def _pro_bench_python_executable() -> Path:
    return _PRO_BENCH_PYTHON


def _multi_bench_python_executable() -> Path:
    return _DEFAULT_MULTI_BENCH_PYTHON


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return path if path.is_absolute() else path.absolute()


def _module_available_with_python(module_name: str | None, python_executable: Path) -> bool:
    if not module_name or not python_executable.exists():
        return False
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
            module_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _docker_image_available(image: str | None) -> bool:
    if not image:
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _docker_image_id(image: str | None) -> str | None:
    if not image:
        return None
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    image_id = (result.stdout or "").strip()
    return image_id or None


def _docker_image_platform(image: str | None) -> str | None:
    if not image:
        return None
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    platform = (result.stdout or "").strip()
    return platform or None


def _postprocess_image_supports_evaluation(image: str | None) -> tuple[bool, str]:
    if not image:
        return False, "Postprocess Docker image is not configured."
    languages = ",".join(_POSTPROCESS_REQUIRED_PARSER_LANGUAGES)
    script = (
        "from contextbench.extractors.treesitter import _get_parser_for_lang; "
        "import sys; "
        f"languages={languages!r}.split(','); "
        "missing=[lang for lang in languages if _get_parser_for_lang(lang) is None]; "
        "print('missing tree-sitter parsers: ' + ', '.join(missing) if missing else 'tree-sitter parsers ok'); "
        "sys.exit(1 if missing else 0)"
    )
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", image, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, output


def _docker_host_socket_path() -> Path | None:
    for candidate in (Path("/var/run/docker.sock"), Path.home() / ".docker" / "run" / "docker.sock"):
        if candidate.exists():
            return candidate
    return None


def _unsupported_resolution_backend(bench: str, *, dataset_name: str | None, reason: str) -> ResolutionBackend:
    return ResolutionBackend(
        backend="unsupported",
        dataset_name=dataset_name,
        module_name=None,
        export_format="jsonl-instance-patch",
        run_evaluation=None,
        setup_command=None,
        unsupported_reason=f"{bench}: {reason}",
        requires_docker=False,
    )


def collect_resolution_predictions(
    *,
    source_dir: Path,
    expected_agent: str,
    bench: str,
) -> dict[str, object]:
    rows = [
        row
        for row in _task_result_rows_for_source_dir(source_dir)
        if str(row.get("bench") or "").strip() == bench
    ]
    prediction_count = 0
    missing_patch_count = 0
    skipped_ineligible_count = 0
    skipped_ineligible_reasons: Counter[str] = Counter()
    predictions: list[dict[str, object]] = []
    no_patch_ids: list[str] = []
    selected_ids: list[str] = []
    task_results = _task_results_for_source_dir(source_dir) if source_dir.exists() else None
    for row in rows:
        record_path = resolve_record_path(
            row.get("record_path"),
            task_results_path=task_results,
            source_dir=source_dir,
        )
        if record_path is None:
            skipped_ineligible_count += 1
            skipped_ineligible_reasons["record_path_missing"] += 1
            continue
        record = read_json(record_path)
        if not isinstance(record, dict):
            skipped_ineligible_count += 1
            skipped_ineligible_reasons["record_not_object"] += 1
            continue
        if not _record_matches_agent(record, expected_agent):
            skipped_ineligible_count += 1
            skipped_ineligible_reasons["record_agent_mismatch"] += 1
            continue
        eligibility_error = _resolution_prediction_ineligibility_reason(row=row, record=record)
        if eligibility_error is not None:
            skipped_ineligible_count += 1
            skipped_ineligible_reasons[eligibility_error] += 1
            continue
        prediction_id = _normalize_agent_record_id(record, row)
        if not prediction_id:
            skipped_ineligible_count += 1
            skipped_ineligible_reasons["prediction_id_missing"] += 1
            continue
        selected_ids.append(prediction_id)
        model_patch = _read_model_patch_for_resolution(record)
        if not model_patch:
            missing_patch_count += 1
            no_patch_ids.append(prediction_id)
            continue
        predictions.append(
            {
                "instance_id": prediction_id,
                "model_patch": model_patch,
                "model_name_or_path": str(record.get("agent") or expected_agent),
            }
        )
        prediction_count += 1
    task_count = len(rows)
    return {
        "bench": bench,
        "task_count": task_count,
        "prediction_count": prediction_count,
        "missing_patch_count": missing_patch_count,
        "skipped_ineligible_count": skipped_ineligible_count,
        "skipped_ineligible_reasons": dict(sorted(skipped_ineligible_reasons.items())),
        "coverage_of_attempted_tasks": (prediction_count / task_count) if task_count else 0.0,
        "is_partial": bool(task_count and prediction_count < task_count),
        "scope": "resolution_predictions",
        "predictions": predictions,
        "prediction_ids": [str(prediction.get("instance_id") or "").strip() for prediction in predictions if str(prediction.get("instance_id") or "").strip()],
        "selected_ids": selected_ids,
        "no_patch_ids": no_patch_ids,
    }


def _resolution_prediction_ineligibility_reason(*, row: dict[str, object], record: dict[str, object]) -> str | None:
    row_status = str(row.get("status") or "").strip().lower()
    if row_status and row_status != "completed":
        return f"task_result_status_{row_status}"
    if bool(row.get("timeout")):
        return "task_result_timeout"
    if "ok" in row and row.get("ok") is not True:
        return "task_result_not_ok"

    record_status = str(record.get("status") or "").strip().lower()
    if record_status != "completed":
        return "record_status_missing" if not record_status else f"record_status_{record_status}"
    if bool(record.get("timeout")):
        return "record_timeout"
    if record.get("ok") is not True:
        return "record_not_ok"
    return None


def _write_resolution_predictions_jsonl(predictions: list[dict[str, object]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False))
            handle.write("\n")


def _write_pro_resolution_predictions_json(
    predictions: list[dict[str, object]],
    out_path: Path,
    *,
    expected_agent: str,
) -> None:
    ensure_dir(out_path.parent)
    payload = [
        {
            "instance_id": str(prediction.get("instance_id") or "").strip(),
            "patch": _normalize_model_patch_for_resolution(prediction.get("model_patch")),
            "prefix": str(prediction.get("model_name_or_path") or expected_agent),
        }
        for prediction in predictions
        if str(prediction.get("instance_id") or "").strip()
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_multi_instance_id(instance_id: str) -> tuple[str, str, int]:
    value = str(instance_id or "").strip()
    if "__" not in value or "-" not in value:
        raise RuntimeError(f"Invalid Multi-SWE-Bench instance id: {instance_id!r}")
    try:
        org_repo, number_text = value.rsplit("-", 1)
        org, repo = org_repo.split("__", 1)
        number = int(number_text)
    except Exception as exc:
        raise RuntimeError(f"Invalid Multi-SWE-Bench instance id: {instance_id!r}") from exc
    if not org or not repo:
        raise RuntimeError(f"Invalid Multi-SWE-Bench instance id: {instance_id!r}")
    return org, repo, number


def _multi_patch_id(*, org: str, repo: str, number: int) -> str:
    return f"{org}__{repo}-{number}"


def _multi_context_id_from_report_id(value: object) -> str:
    text = str(value or "").strip()
    if "__" in text and "-" in text:
        _parse_multi_instance_id(text)
        return text
    if "/" not in text or ":pr-" not in text:
        raise RuntimeError(f"Invalid Multi-SWE-Bench report id: {text!r}")
    org_repo, number_text = text.rsplit(":pr-", 1)
    org, repo = org_repo.split("/", 1)
    return _multi_patch_id(org=org, repo=repo, number=int(number_text))


def _write_multi_resolution_predictions_jsonl(
    predictions: list[dict[str, object]],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for prediction in predictions:
            instance_id = str(prediction.get("instance_id") or "").strip()
            if not instance_id:
                continue
            org, repo, number = _parse_multi_instance_id(instance_id)
            handle.write(
                json.dumps(
                    {
                        "org": org,
                        "repo": repo,
                        "number": number,
                        "fix_patch": _normalize_model_patch_for_resolution(prediction.get("model_patch")),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def _resolution_predictions_path(*, predictions_root: Path, bench: str, backend: ResolutionBackend) -> Path:
    extension = "json" if backend.backend == "swebench-pro" else "jsonl"
    return predictions_root / f"{bench.lower()}-{backend.backend}.{extension}"


def _resolution_run_id(
    *,
    eval_root: Path,
    suite_name: str,
    variant_name: str,
    bench: str,
    run_suffix: str | None = None,
    resume_existing: bool = False,
) -> str:
    prefix = safe_path_component(f"{suite_name}-{variant_name}-{bench}-resolution")
    bench_root = eval_root / bench.lower()
    if resume_existing and bench_root.exists():
        candidates = [path for path in bench_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
        if candidates:
            latest = max(candidates, key=lambda path: path.stat().st_mtime)
            return latest.name
    suffix = safe_path_component(run_suffix or str(time.time_ns()))
    return f"{prefix}-{suffix}"


def _write_backend_resolution_predictions(
    *,
    predictions: list[dict[str, object]],
    out_path: Path,
    backend: ResolutionBackend,
    expected_agent: str,
) -> None:
    if backend.backend == "swebench-pro":
        _write_pro_resolution_predictions_json(predictions, out_path, expected_agent=expected_agent)
        return
    if backend.backend == "multi-swebench":
        _write_multi_resolution_predictions_jsonl(predictions, out_path)
        return
    _write_resolution_predictions_jsonl(predictions, out_path)


def export_resolution_predictions(
    *,
    source_dir: Path,
    expected_agent: str,
    bench: str,
    out_path: Path,
) -> dict[str, object]:
    summary = collect_resolution_predictions(
        source_dir=source_dir,
        expected_agent=expected_agent,
        bench=bench,
    )
    _write_resolution_predictions_jsonl(list(summary.get("predictions") or []), out_path)
    summary = dict(summary)
    summary["predictions_path"] = str(out_path)
    summary.pop("predictions", None)
    return summary


def _find_resolution_report_payload(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        if "resolved_ids" in value or "unresolved_ids" in value or "error_ids" in value:
            return value
        for child in value.values():
            found = _find_resolution_report_payload(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_resolution_report_payload(child)
            if found is not None:
                return found
    return None


def _load_resolution_report(report_root: Path) -> dict[str, object]:
    candidates = sorted(report_root.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.name in {"resolution-error.json", "resolution-result.json"}:
            continue
        try:
            payload = read_json(candidate)
        except Exception:
            continue
        found = _find_resolution_report_payload(payload)
        if found is not None:
            found = dict(found)
            found["report_path"] = str(candidate)
            return found
    raise RuntimeError(f"Unable to locate a SWE-bench resolution report under {report_root}")


def _load_poly_resolution_report(work_dir: Path) -> dict[str, object]:
    report_path = work_dir / "result.json"
    if not report_path.exists():
        raise RuntimeError(f"Unable to locate SWE-PolyBench result.json under {work_dir}")
    payload = read_json(report_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"SWE-PolyBench result.json must contain an object: {report_path}")

    resolved_ids = [str(item).strip() for item in (payload.get("resolved") or []) if str(item).strip()]
    unresolved_ids = [str(item).strip() for item in (payload.get("not_resolved") or []) if str(item).strip()]
    error_ids = [str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()]
    return {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "resolved_count": int(payload.get("total_resolved") or len(resolved_ids)),
        "total_instances": int(payload.get("total_instances") or len(resolved_ids) + len(unresolved_ids) + len(error_ids)),
        "total_resolved": int(payload.get("total_resolved") or len(resolved_ids)),
        "total_unresolved": int(payload.get("total_unresolved") or len(unresolved_ids)),
        "empty_patch_instances": int(payload.get("total_empty_patch_instances") or 0),
        "generated_ids": [str(item).strip() for item in (payload.get("generation") or []) if str(item).strip()],
        "no_generation_ids": [str(item).strip() for item in (payload.get("no_generation") or []) if str(item).strip()],
        "patch_applied_ids": [str(item).strip() for item in (payload.get("patch_applied") or []) if str(item).strip()],
        "with_logs_ids": [str(item).strip() for item in (payload.get("with_logs") or []) if str(item).strip()],
        "report_path": str(report_path),
    }


def _load_dataset_rows(dataset_name: str, *, split: str) -> tuple[list[dict[str, object]], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Resolution dataset filtering requires the 'datasets' package in the run-suite Python environment."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    column_names = [str(column) for column in getattr(dataset, "column_names", [])]
    rows = [dict(row) for row in dataset]
    return rows, column_names


def _coerce_poly_csv_value(column: str, value: object) -> object:
    if value is None:
        return ""
    if column == "modified_nodes" and not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if column in {"F2P", "P2P"} and not isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_poly_dataset_csv(
    *,
    dataset_name: str,
    instance_ids: list[str],
    out_path: Path,
) -> None:
    if not instance_ids:
        raise RuntimeError("SWE-PolyBench dataset export requires at least one prediction instance_id.")

    rows, column_names = _load_dataset_rows(dataset_name, split="test")
    wanted = {instance_id for instance_id in instance_ids if instance_id}
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id in wanted and instance_id not in by_id:
            by_id[instance_id] = row

    missing_ids = [instance_id for instance_id in instance_ids if instance_id and instance_id not in by_id]
    if missing_ids:
        raise RuntimeError(
            "SWE-PolyBench dataset is missing selected instances: "
            + ", ".join(missing_ids[:10])
            + (f" ... and {len(missing_ids) - 10} more" if len(missing_ids) > 10 else "")
        )

    missing_columns: dict[str, list[str]] = {}
    for instance_id in instance_ids:
        row = by_id.get(instance_id)
        if row is None:
            continue
        missing = [column for column in _POLY_REQUIRED_SAMPLE_COLUMNS if column not in row]
        if missing:
            missing_columns[instance_id] = missing
    if missing_columns:
        details = "; ".join(
            f"{instance_id}: {', '.join(columns)}"
            for instance_id, columns in list(missing_columns.items())[:5]
        )
        raise RuntimeError(f"SWE-PolyBench dataset is missing required columns: {details}")

    fieldnames: list[str] = []
    for column in column_names:
        if column not in fieldnames:
            fieldnames.append(column)
    for column in _POLY_REQUIRED_SAMPLE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance_id in instance_ids:
            row = by_id[instance_id]
            writer.writerow(
                {
                    column: _coerce_poly_csv_value(column, row.get(column))
                    for column in fieldnames
                }
            )


def _write_multi_dataset_jsonl(
    *,
    dataset_name: str,
    instance_ids: list[str],
    out_path: Path,
) -> None:
    if not instance_ids:
        raise RuntimeError("Multi-SWE-Bench dataset export requires at least one prediction instance_id.")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "Multi-SWE-Bench dataset filtering requires the 'huggingface_hub' package in the run-suite Python environment."
        ) from exc

    wanted = {instance_id for instance_id in instance_ids if instance_id}
    by_id: dict[str, dict[str, object]] = {}
    repo_files = list_repo_files(dataset_name, repo_type="dataset")
    candidate_paths: list[str] = []
    for instance_id in instance_ids:
        org, repo, _number = _parse_multi_instance_id(instance_id)
        repo_file_name = f"{org}__{repo}_dataset.jsonl"
        matches = [path for path in repo_files if path.endswith(f"/{repo_file_name}")]
        if not matches:
            raise RuntimeError(f"Multi-SWE-Bench dataset repo has no JSONL shard for selected instance: {instance_id}")
        for match in matches:
            if match not in candidate_paths:
                candidate_paths.append(match)

    for dataset_file in candidate_paths:
        local_path = Path(hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=dataset_file))
        for raw_line in local_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                continue
            org = str(row.get("org") or "").strip()
            repo = str(row.get("repo") or "").strip()
            number = row.get("number")
            instance_id = str(row.get("instance_id") or "").strip()
            if not instance_id and org and repo and number not in (None, ""):
                try:
                    instance_id = _multi_patch_id(org=org, repo=repo, number=int(number))
                except Exception:
                    instance_id = ""
            if instance_id in wanted and instance_id not in by_id:
                by_id[instance_id] = row
                if len(by_id) == len(wanted):
                    break
        if len(by_id) == len(wanted):
            break

    missing_ids = [instance_id for instance_id in instance_ids if instance_id and instance_id not in by_id]
    if missing_ids:
        raise RuntimeError(
            "Multi-SWE-Bench dataset is missing selected instances: "
            + ", ".join(missing_ids[:10])
            + (f" ... and {len(missing_ids) - 10} more" if len(missing_ids) > 10 else "")
        )

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for instance_id in instance_ids:
            row = dict(by_id[instance_id])
            org = str(row.get("org") or "").strip()
            repo = str(row.get("repo") or "").strip()
            number = row.get("number")
            if not row.get("instance_id") and org and repo and number not in (None, ""):
                row["instance_id"] = _multi_patch_id(org=org, repo=repo, number=int(number))
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_resolution_error_summary(*, work_dir: Path, payload: dict[str, object]) -> str:
    ensure_dir(work_dir)
    path = work_dir / "resolution-error.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return str(path)


def _append_text_log(log_path: Path, text: str) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _write_reused_resolution_log(*, log_path: Path, instance_id: str, summary_path: Path) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"[reuse] {instance_id} -> {summary_path}\n")
        handle.write("Existing non-error resolution summary reused; evaluator was not rerun for this instance.\n")


def _resolution_instance_dir(work_dir: Path, instance_id: str) -> Path:
    return work_dir / "instances" / safe_path_component(instance_id)


def _resolution_instance_summary_path(instance_dir: Path) -> Path:
    return instance_dir / "resolution-result.json"


def _resolution_instance_diagnostics_path(instance_dir: Path) -> Path:
    return instance_dir / "resolution-diagnostics.json"


def _resolution_checkpoint_instance_dir(cache_dir: Path, instance_id: str) -> Path:
    return cache_dir / "instances" / safe_path_component(instance_id)


def _read_resolution_artifact_text(path: Path, *, max_chars: int = 200_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _resolution_artifact_tail(path_value: object, *, max_chars: int = 20_000) -> str | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    text = _read_resolution_artifact_text(path, max_chars=max_chars)
    return text or None


def _safe_json_object(path_value: object) -> dict[str, object] | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_resolution_instance_diagnostics(
    *,
    summary: dict[str, object],
    instance_dir: Path,
    artifacts_retained: bool,
    cleanup_note: str | None = None,
) -> dict[str, object]:
    report_payload = _safe_json_object(summary.get("report_path"))
    diagnostic: dict[str, object] = {
        "schema_version": 1,
        "instance_id": str(summary.get("instance_id") or instance_dir.name),
        "status": str(summary.get("status") or ""),
        "resolved_ids": list(summary.get("resolved_ids") or []),
        "unresolved_ids": list(summary.get("unresolved_ids") or []),
        "error_ids": list(summary.get("error_ids") or []),
        "evaluation_error_ids": list(summary.get("evaluation_error_ids") or []),
        "test_timeout_ids": list(summary.get("test_timeout_ids") or []),
        "infra_error_ids": list(summary.get("infra_error_ids") or []),
        "resolution_failure_kind": summary.get("resolution_failure_kind"),
        "resolution_failure_is_evaluated": summary.get("resolution_failure_is_evaluated"),
        "error_detail": summary.get("error_detail"),
        "log_path": summary.get("log_path"),
        "report_path": summary.get("report_path"),
        "artifacts_retained": artifacts_retained,
        "input_metadata": summary.get("input_metadata"),
        "log_tail": _resolution_artifact_tail(summary.get("log_path")),
    }
    if report_payload is not None:
        diagnostic["report_keys"] = sorted(str(key) for key in report_payload.keys())
        for key in (
            "resolved_ids",
            "unresolved_ids",
            "error_ids",
            "evaluation_error_ids",
            "test_timeout_ids",
            "infra_error_ids",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ):
            if key in report_payload:
                diagnostic[f"report_{key}"] = report_payload.get(key)
    if cleanup_note:
        diagnostic["cleanup_note"] = cleanup_note
        diagnostic["cleanup_recorded_at_unix"] = time.time()
    return {key: value for key, value in diagnostic.items() if value not in (None, "", [])}


def _write_resolution_instance_diagnostics(
    instance_dir: Path,
    summary: dict[str, object],
    *,
    artifacts_retained: bool,
    cleanup_note: str | None = None,
) -> dict[str, object]:
    diagnostics_path = _resolution_instance_diagnostics_path(instance_dir)
    diagnostic = _build_resolution_instance_diagnostics(
        summary=summary,
        instance_dir=instance_dir,
        artifacts_retained=artifacts_retained,
        cleanup_note=cleanup_note,
    )
    diagnostic["diagnostics_path"] = str(diagnostics_path)
    write_json(diagnostics_path, diagnostic)
    return diagnostic


def _swebench_timeout_from_input_metadata(input_metadata: dict[str, object]) -> int | None:
    harness_args = input_metadata.get("harness_args")
    if not isinstance(harness_args, list):
        return None
    args = [str(item) for item in harness_args]
    for index, arg in enumerate(args):
        if arg == "--timeout" and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return None
        if arg.startswith("--timeout="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _swebench_max_test_runtime_seconds(log_text: str) -> float | None:
    runtimes: list[float] = []
    for match in re.finditer(r"Test runtime:\s*([0-9][0-9_,]*(?:\.[0-9]+)?)\s+seconds", log_text):
        try:
            runtimes.append(float(match.group(1).replace("_", "").replace(",", "")))
        except ValueError:
            continue
    return max(runtimes) if runtimes else None


def _swebench_error_is_patch_test_timeout(instance_dir: Path, *, timeout_seconds: int | None = None) -> bool:
    run_log_text = "\n".join(
        _read_resolution_artifact_text(path)
        for path in sorted(instance_dir.rglob("run_instance.log"))
    )
    test_output_text = "\n".join(
        _read_resolution_artifact_text(path)
        for path in sorted(instance_dir.rglob("test_output.txt"))
    )
    combined = f"{run_log_text}\n{test_output_text}"
    if not combined.strip():
        return False
    patch_applied = "Applied Patch:" in run_log_text or "Applied patch" in run_log_text
    test_started = (
        ">>>>> Start Test Output" in test_output_text
        or "Test output for " in run_log_text
        or "Test runtime:" in run_log_text
    )
    explicit_timeout = "Test timed out after" in combined or "Timeout error:" in combined
    stopped_after_timeout = False
    if timeout_seconds is not None:
        test_runtime = _swebench_max_test_runtime_seconds(run_log_text)
        stopped_after_timeout = (
            test_runtime is not None
            and test_runtime >= timeout_seconds
            and "container" in run_log_text
            and "is not running" in run_log_text
        )
    timed_out = explicit_timeout or stopped_after_timeout
    return patch_applied and test_started and timed_out


def _normalize_swebench_instance_summary(
    summary: dict[str, object],
    *,
    instance_dir: Path,
) -> dict[str, object]:
    input_metadata = summary.get("input_metadata")
    if not isinstance(input_metadata, dict) or input_metadata.get("backend") != "swebench":
        return summary
    instance_id = str(summary.get("instance_id") or instance_dir.name).strip()
    if not instance_id:
        return summary
    error_ids = [str(item).strip() for item in (summary.get("error_ids") or []) if str(item).strip()]
    if instance_id not in set(error_ids):
        return summary
    timeout_seconds = _swebench_timeout_from_input_metadata(input_metadata)
    if not _swebench_error_is_patch_test_timeout(instance_dir, timeout_seconds=timeout_seconds):
        return summary

    resolved_ids = [str(item).strip() for item in (summary.get("resolved_ids") or []) if str(item).strip()]
    unresolved_ids = [str(item).strip() for item in (summary.get("unresolved_ids") or []) if str(item).strip()]
    infra_error_ids = [item for item in error_ids if item != instance_id]
    if instance_id not in set(unresolved_ids):
        unresolved_ids.append(instance_id)

    normalized = dict(summary)
    normalized["resolved_ids"] = resolved_ids
    normalized["unresolved_ids"] = unresolved_ids
    normalized["error_ids"] = infra_error_ids
    normalized["status"] = "unresolved"
    normalized["evaluation_error_ids"] = sorted(set([*(normalized.get("evaluation_error_ids") or []), instance_id]))
    normalized["test_timeout_ids"] = sorted(set([*(normalized.get("test_timeout_ids") or []), instance_id]))
    normalized["infra_error_ids"] = infra_error_ids
    normalized["resolution_failure_kind"] = "test_timeout"
    normalized["resolution_failure_is_evaluated"] = True
    normalized["error_detail"] = normalized.get("error_detail") or (
        "SWE-bench test command timed out after the patch was applied and test execution started; "
        "counting as an evaluated unresolved patch result."
    )
    return normalized


def _read_resolution_instance_summary(
    instance_dir: Path,
    *,
    expected_input_metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    path = _resolution_instance_summary_path(instance_dir)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    if expected_input_metadata is not None and payload.get("input_metadata") != expected_input_metadata:
        return None
    payload = _normalize_swebench_instance_summary(payload, instance_dir=instance_dir)
    instance_id = str(payload.get("instance_id") or instance_dir.name).strip()
    resolved_ids = {str(item).strip() for item in (payload.get("resolved_ids") or []) if str(item).strip()}
    unresolved_ids = {str(item).strip() for item in (payload.get("unresolved_ids") or []) if str(item).strip()}
    error_ids = {str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()}
    status = str(payload.get("status") or "").strip().lower()
    if status == "error":
        return None
    if status == "resolved" and instance_id not in resolved_ids:
        return None
    if status == "unresolved" and instance_id not in unresolved_ids:
        return None
    return payload


def _read_reusable_resolution_instance_summary(
    instance_dir: Path,
    *,
    cache_dir: Path | None = None,
    expected_input_metadata: dict[str, object] | None = None,
) -> tuple[dict[str, object], Path] | None:
    local = _read_resolution_instance_summary(
        instance_dir,
        expected_input_metadata=expected_input_metadata,
    )
    if local is not None:
        return local, _resolution_instance_summary_path(instance_dir)
    if cache_dir is None:
        return None
    instance_id = instance_dir.name
    checkpoint_dir = _resolution_checkpoint_instance_dir(cache_dir, instance_id)
    cached = _read_resolution_instance_summary(
        checkpoint_dir,
        expected_input_metadata=expected_input_metadata,
    )
    if cached is None:
        return None
    return cached, _resolution_instance_summary_path(checkpoint_dir)


def _status_for_instance_report(instance_id: str, resolved_ids: list[str], unresolved_ids: list[str], error_ids: list[str]) -> str:
    id_set = {str(instance_id).strip()}
    if id_set & set(resolved_ids):
        return "resolved"
    if id_set & set(error_ids):
        return "error"
    if id_set & set(unresolved_ids):
        return "unresolved"
    return "error"


def _write_resolution_instance_summary(
    instance_dir: Path,
    payload: dict[str, object],
    *,
    cache_dir: Path | None = None,
) -> None:
    payload["diagnostics_path"] = str(_resolution_instance_diagnostics_path(instance_dir))
    payload["artifacts_retained"] = True
    local_payload = dict(payload)
    write_json(_resolution_instance_summary_path(instance_dir), local_payload)
    _write_resolution_instance_diagnostics(
        instance_dir,
        local_payload,
        artifacts_retained=True,
    )
    if cache_dir is not None:
        instance_id = str(payload.get("instance_id") or instance_dir.name).strip()
        if instance_id:
            checkpoint_dir = _resolution_checkpoint_instance_dir(cache_dir, instance_id)
            checkpoint_payload = dict(payload)
            checkpoint_payload["diagnostics_path"] = str(_resolution_instance_diagnostics_path(checkpoint_dir))
            checkpoint_payload["artifacts_retained"] = True
            write_json(_resolution_instance_summary_path(checkpoint_dir), checkpoint_payload)
            _write_resolution_instance_diagnostics(
                checkpoint_dir,
                checkpoint_payload,
                artifacts_retained=True,
            )


def _cleanup_checkpointed_resolution_instance(
    instance_dir: Path,
    *,
    cache_dir: Path | None,
    input_metadata: dict[str, object],
    log_path: Path,
    enabled: bool = True,
) -> bool:
    if not enabled or cache_dir is None or not instance_dir.exists():
        return False
    instance_id = instance_dir.name
    checkpoint_dir = _resolution_checkpoint_instance_dir(cache_dir, instance_id)
    checkpoint_path = _resolution_instance_summary_path(checkpoint_dir)
    if not checkpoint_path.exists():
        return False
    reusable = _read_resolution_instance_summary(
        checkpoint_dir,
        expected_input_metadata=input_metadata,
    )
    if reusable is None:
        return False
    cleanup_note = f"Removed checkpointed resolution artifact directory {instance_dir}"
    checkpoint_payload = dict(reusable)
    checkpoint_payload["artifacts_retained"] = False
    checkpoint_payload["diagnostics_path"] = str(_resolution_instance_diagnostics_path(checkpoint_dir))
    checkpoint_payload["artifact_cleanup"] = {
        "removed_instance_dir": str(instance_dir),
        "recorded_at_unix": time.time(),
    }
    _write_resolution_instance_diagnostics(
        checkpoint_dir,
        checkpoint_payload,
        artifacts_retained=False,
        cleanup_note=cleanup_note,
    )
    write_json(_resolution_instance_summary_path(checkpoint_dir), checkpoint_payload)
    shutil.rmtree(instance_dir)
    _append_text_log(log_path, f"[cleanup] removed checkpointed resolution artifacts for {instance_id}")
    return True


def _mark_resolution_summary_artifacts_cleaned(
    summary: dict[str, object],
    *,
    cache_dir: Path | None,
    instance_id: str,
) -> None:
    summary["artifacts_retained"] = False
    if cache_dir is not None:
        checkpoint_dir = _resolution_checkpoint_instance_dir(cache_dir, instance_id)
        summary["diagnostics_path"] = str(_resolution_instance_diagnostics_path(checkpoint_dir))


def _docker_image_references() -> list[str]:
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _docker_active_image_references() -> set[str]:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Image}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _multi_resolution_docker_image_ref(instance_id: str) -> str | None:
    repo_part, separator, number = instance_id.rpartition("-")
    if not separator or not number.isdigit() or "__" not in repo_part:
        return None
    org, repo = repo_part.split("__", 1)
    org = org.strip().lower()
    repo = repo.strip().lower()
    if not org or not repo:
        return None
    return f"mswebench/{org}_m_{repo}:pr-{number}"


def _swebench_resolution_docker_image_ref(instance_id: str) -> str:
    instance = str(instance_id or "").strip().lower()
    if not instance:
        raise RuntimeError("SWE-bench resolution image requires a non-empty instance_id.")
    return f"sweb.eval.x86_64.{instance}:latest"


def _poly_resolution_docker_image_ref(*, instance_id: str, language: str) -> str:
    instance = str(instance_id or "").strip().lower()
    lang = str(language or "").strip().lower()
    if not instance or not lang:
        raise RuntimeError("SWE-PolyBench resolution image requires instance_id and language.")
    return f"polybench_{lang}_{instance}"


def _repo_name_from_task(task: dict[str, object]) -> str:
    repo_name = str(task.get("repo") or "").strip()
    if repo_name:
        return repo_name
    repo_url = str(task.get("repo_url") or "").strip()
    if not repo_url:
        raise RuntimeError(f"Task is missing repo metadata: {task.get('instance_id') or task.get('original_inst_id')}")
    repo = repo_url.rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "/" not in repo:
        raise RuntimeError(f"Task repo_url is not owner/name-addressable: {repo_url!r}")
    return "/".join(repo.rsplit("/", 2)[-2:])


def resolution_instance_id_from_task(task: dict[str, object]) -> str:
    instance_id = str(task.get("original_inst_id") or task.get("instance_id") or "").strip()
    if not instance_id:
        raise RuntimeError("Task is missing original_inst_id/instance_id for resolution image selection.")
    return instance_id


def resolution_image_ref_for_task(task: dict[str, object]) -> str:
    bench = str(task.get("bench") or task.get("source") or "Verified").strip() or "Verified"
    instance_id = resolution_instance_id_from_task(task)
    if bench == "Verified":
        return _swebench_resolution_docker_image_ref(instance_id)
    if bench == "Poly":
        return _poly_resolution_docker_image_ref(
            instance_id=instance_id,
            language=str(task.get("language") or ""),
        )
    if bench == "Pro":
        return _pro_dockerhub_image_uri(instance_id, _repo_name_from_task(task))
    if bench == "Multi":
        image_ref = _multi_resolution_docker_image_ref(instance_id)
        if not image_ref:
            raise RuntimeError(f"Invalid Multi-SWE-Bench instance id for image selection: {instance_id!r}")
        return image_ref
    raise RuntimeError(f"No resolution image mapping is configured for bench {bench!r}")


def _resolution_docker_image_refs_for_instance(*, backend: str, instance_id: str) -> list[str]:
    image_refs = set(_docker_image_references())
    if backend == "swebench":
        image_ref = _swebench_resolution_docker_image_ref(instance_id)
        return [image_ref] if image_ref in image_refs else []
    if backend == "multi-swebench":
        image_ref = _multi_resolution_docker_image_ref(instance_id)
        return [image_ref] if image_ref and image_ref in image_refs else []
    if backend == "swe-polybench":
        ghcr_ref = f"ghcr.io/timesler/swe-polybench.eval.x86_64.{instance_id}:latest"
        poly_suffix = f"_{instance_id}:latest"
        refs = [
            ref
            for ref in image_refs
            if ref == ghcr_ref or (ref.startswith("polybench_") and ref.endswith(poly_suffix))
        ]
        return sorted(refs)
    return []


def _remove_docker_image_ref(image_ref: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "image", "rm", image_ref],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "docker image rm timed out"
    except OSError as exc:
        return False, str(exc)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, output


def _cleanup_checkpointed_resolution_docker_images(
    *,
    instance_id: str,
    backend: ResolutionBackend,
    cache_dir: Path | None,
    input_metadata: dict[str, object],
    log_path: Path,
    enabled: bool = True,
) -> list[str]:
    if not enabled or cache_dir is None:
        return []
    checkpoint_dir = _resolution_checkpoint_instance_dir(cache_dir, instance_id)
    reusable = _read_resolution_instance_summary(
        checkpoint_dir,
        expected_input_metadata=input_metadata,
    )
    if reusable is None:
        return []

    active_image_refs = _docker_active_image_references()
    removed_refs: list[str] = []
    for image_ref in _resolution_docker_image_refs_for_instance(
        backend=backend.backend,
        instance_id=instance_id,
    ):
        if image_ref in active_image_refs:
            _append_text_log(log_path, f"[docker-cleanup] skipped active image {image_ref}")
            continue
        removed, detail = _remove_docker_image_ref(image_ref)
        if removed:
            removed_refs.append(image_ref)
            _append_text_log(log_path, f"[docker-cleanup] removed image {image_ref}")
        else:
            _append_text_log(log_path, f"[docker-cleanup] skipped image {image_ref}: {detail}")
    return removed_refs


def _resolution_prediction_metadata_payload(
    prediction: dict[str, object],
    *,
    backend: ResolutionBackend,
) -> dict[str, object]:
    if backend.backend == "swebench-pro":
        return {
            "instance_id": str(prediction.get("instance_id") or "").strip(),
            "patch": _normalize_model_patch_for_resolution(prediction.get("patch", prediction.get("model_patch"))),
            "prefix": str(prediction.get("prefix", prediction.get("model_name_or_path")) or ""),
        }
    if backend.backend == "multi-swebench":
        if prediction.get("org") is not None or prediction.get("repo") is not None or prediction.get("number") is not None:
            return {
                "org": str(prediction.get("org") or "").strip(),
                "repo": str(prediction.get("repo") or "").strip(),
                "number": int(prediction.get("number") or 0),
                "fix_patch": _normalize_model_patch_for_resolution(prediction.get("fix_patch", prediction.get("model_patch"))),
            }
        instance_id = str(prediction.get("instance_id") or "").strip()
        org, repo, number = _parse_multi_instance_id(instance_id)
        return {
            "org": org,
            "repo": repo,
            "number": number,
            "fix_patch": _normalize_model_patch_for_resolution(prediction.get("fix_patch", prediction.get("model_patch"))),
        }
    return {
        "instance_id": str(prediction.get("instance_id") or "").strip(),
        "model_patch": _normalize_model_patch_for_resolution(prediction.get("model_patch")),
        "model_name_or_path": str(prediction.get("model_name_or_path") or ""),
    }


def _resolution_instance_input_metadata(
    prediction: dict[str, object],
    *,
    backend: ResolutionBackend,
    dataset_name: str,
    harness_args: list[str] | None,
) -> dict[str, object]:
    return {
        "schema_version": _RESOLUTION_INPUT_METADATA_VERSION,
        "backend": backend.backend,
        "dataset_name": dataset_name,
        "harness_args": [str(item) for item in (harness_args or [])],
        "prediction_sha256": stable_json_hash(
            _resolution_prediction_metadata_payload(prediction, backend=backend)
        ),
    }


def _extend_resolution_ids(target: list[str], values: object) -> None:
    seen = set(target)
    for value in values or []:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        target.append(value)


def _aggregate_instance_resolution_results(
    *,
    instance_summaries: list[dict[str, object]],
    report_path: Path,
) -> dict[str, object]:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    evaluation_error_ids: list[str] = []
    test_timeout_ids: list[str] = []
    infra_error_ids: list[str] = []
    instance_diagnostics: list[dict[str, object]] = []
    for summary in instance_summaries:
        _extend_resolution_ids(resolved_ids, summary.get("resolved_ids") or [])
        _extend_resolution_ids(unresolved_ids, summary.get("unresolved_ids") or [])
        _extend_resolution_ids(error_ids, summary.get("error_ids") or [])
        _extend_resolution_ids(evaluation_error_ids, summary.get("evaluation_error_ids") or [])
        _extend_resolution_ids(test_timeout_ids, summary.get("test_timeout_ids") or [])
        _extend_resolution_ids(infra_error_ids, summary.get("infra_error_ids") or [])
        instance_diagnostics.append(
            {
                "instance_id": summary.get("instance_id"),
                "status": summary.get("status"),
                "diagnostics_path": summary.get("diagnostics_path"),
                "artifacts_retained": summary.get("artifacts_retained"),
                "resolution_failure_kind": summary.get("resolution_failure_kind"),
            }
        )

    aggregate_payload = {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "resolved_count": len(resolved_ids),
        "total_instances": len(resolved_ids) + len(unresolved_ids) + len(error_ids),
    }
    if evaluation_error_ids:
        aggregate_payload["evaluation_error_ids"] = evaluation_error_ids
    if test_timeout_ids:
        aggregate_payload["test_timeout_ids"] = test_timeout_ids
    if infra_error_ids:
        aggregate_payload["infra_error_ids"] = infra_error_ids
    if instance_diagnostics:
        aggregate_payload["instance_diagnostics"] = instance_diagnostics

    aggregate = {**aggregate_payload, "report_path": str(report_path)}
    ensure_dir(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return aggregate


def _write_single_resolution_prediction(
    *,
    prediction: dict[str, object],
    out_path: Path,
    backend: ResolutionBackend,
) -> None:
    if backend.backend == "swebench-pro":
        payload = [
            {
                "instance_id": str(prediction.get("instance_id") or "").strip(),
                "patch": _normalize_model_patch_for_resolution(prediction.get("patch", prediction.get("model_patch"))),
                "prefix": str(prediction.get("model_name_or_path") or ""),
            }
        ]
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return
    if backend.backend == "multi-swebench":
        _write_multi_resolution_predictions_jsonl([prediction], out_path)
        return
    _write_resolution_predictions_jsonl([prediction], out_path)


def _swebench_harness_args_with_timeout(
    harness_args: list[str] | None,
    *,
    swebench_timeout: int,
) -> list[str]:
    args = [str(item) for item in (harness_args or [])]
    for arg in args:
        if arg == "--timeout" or arg.startswith("--timeout="):
            raise ValueError(
                "postprocess.resolve_harness_args must not include --timeout for SWE-bench; "
                "use postprocess.swebench_timeout instead."
            )
    return [*args, "--timeout", str(int(swebench_timeout))]


def run_resolution_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    swebench_timeout: int = 1800,
    cache_dir: Path | None = None,
    self_clean_resolution_artifacts: bool = True,
    self_clean_resolution_docker_images: bool = True,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    cache_dir = cache_dir.resolve() if cache_dir is not None else None
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    effective_harness_args = _swebench_harness_args_with_timeout(
        harness_args,
        swebench_timeout=swebench_timeout,
    )
    backend = _resolution_backend_for_bench("Verified")
    valid_predictions = [
        prediction
        for prediction in prediction_rows
        if isinstance(prediction, dict) and str(prediction.get("instance_id") or "").strip()
    ]

    log_lock = threading.Lock()

    def append_log(text: str) -> None:
        with log_lock:
            _append_text_log(log_path, text)

    def run_one(job_index: int) -> _ResolutionInstanceResult:
        prediction = valid_predictions[job_index]
        instance_id = str(prediction.get("instance_id") or "").strip()
        display_index = job_index + 1
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=effective_harness_args,
        )
        reusable = _read_reusable_resolution_instance_summary(
            instance_dir,
            cache_dir=cache_dir,
            expected_input_metadata=input_metadata,
        )
        if reusable is not None:
            existing, reused_summary_path = reusable
            _write_resolution_instance_summary(instance_dir, existing, cache_dir=cache_dir)
            append_log(f"[reuse] {instance_id} -> {reused_summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=reused_summary_path,
            )
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    existing,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            return _ResolutionInstanceResult(index=job_index, summary=existing)

        instance_predictions_path = instance_dir / "predictions.jsonl"
        _write_single_resolution_prediction(
            prediction=prediction,
            out_path=instance_predictions_path,
            backend=backend,
        )
        instance_log_path = instance_dir / "resolution-command.log"
        instance_run_id = f"{run_id}--{safe_path_component(instance_id)}"
        command = [
            str(_swe_bench_python_executable()),
            str(_SWEBENCH_RESOLUTION_WRAPPER),
            "--dataset_name",
            dataset_name,
            "--predictions_path",
            str(instance_predictions_path),
            "--max_workers",
            "1",
            "--run_id",
            instance_run_id,
            "--report_dir",
            str(instance_dir),
            *effective_harness_args,
        ]
        append_log(f"[run] {display_index}/{len(valid_predictions)} {instance_id}")
        print(
            f"[resolution:{run_id}] starting backend=swebench instance={instance_id} predictions={instance_predictions_path}",
            flush=True,
        )
        returncode, tail = _run_resolution_command(
            command=command,
            cwd=instance_dir,
            log_path=instance_log_path,
            log_prefix=f"[resolution:{run_id}]",
            env=env,
        )
        try:
            instance_report = _load_resolution_report(instance_dir)
            resolved_ids = [str(item).strip() for item in (instance_report.get("resolved_ids") or []) if str(item).strip()]
            unresolved_ids = [str(item).strip() for item in (instance_report.get("unresolved_ids") or []) if str(item).strip()]
            error_ids = [str(item).strip() for item in (instance_report.get("error_ids") or []) if str(item).strip()]
            if instance_id in set(resolved_ids):
                status = "resolved"
            elif instance_id in set(error_ids):
                status = "error"
            elif instance_id in set(unresolved_ids):
                status = "unresolved"
            else:
                status = "error"
                error_ids = error_ids or [instance_id]
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": error_ids,
                "log_path": str(instance_log_path),
                "report_path": str(instance_report.get("report_path") or ""),
                "status": status,
                "input_metadata": input_metadata,
            }
            instance_summary = _normalize_swebench_instance_summary(
                instance_summary,
                instance_dir=instance_dir,
            )
        except Exception:
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": [instance_id],
                "log_path": str(instance_log_path),
                "status": "error",
                "input_metadata": input_metadata,
            }
        successful_return = returncode == 0
        _write_resolution_instance_summary(
            instance_dir,
            instance_summary,
            cache_dir=cache_dir if successful_return else None,
        )
        if successful_return:
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    instance_summary,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
        error = None
        if not successful_return:
            error = ResolutionCommandError(
                message=(
                    f"SWE-bench harness failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )
        return _ResolutionInstanceResult(index=job_index, summary=instance_summary, error=error)

    instance_results = _run_resolution_instance_jobs(
        job_count=len(valid_predictions),
        max_workers=max_workers,
        run_one=run_one,
    )
    instance_summaries = [result.summary for result in instance_results]
    first_error = next((result.error for result in instance_results if result.error is not None), None)
    summary = _aggregate_instance_resolution_results(
        instance_summaries=instance_summaries,
        report_path=work_dir / "report.json",
    )
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_swe_bench_python_executable())
    summary["wrapper_path"] = str(_SWEBENCH_RESOLUTION_WRAPPER)
    summary["swebench_timeout"] = int(swebench_timeout)
    if first_error is not None:
        summary["error_detail"] = str(first_error)
        summary["exit_code"] = first_error.exit_code
        summary["tail"] = first_error.tail
        summary["_partial_from_error"] = True
    return summary


def run_poly_resolution_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    self_clean_resolution_artifacts: bool = True,
    self_clean_resolution_docker_images: bool = True,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    cache_dir = cache_dir.resolve() if cache_dir is not None else None
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    dataset_subset_path = work_dir / "dataset-subset.csv"
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    backend = _resolution_backend_for_bench("Poly")
    valid_predictions = [
        prediction
        for prediction in prediction_rows
        if isinstance(prediction, dict) and str(prediction.get("instance_id") or "").strip()
    ]
    log_lock = threading.Lock()

    def append_log(text: str) -> None:
        with log_lock:
            _append_text_log(log_path, text)

    def run_one(job_index: int) -> _ResolutionInstanceResult:
        prediction = valid_predictions[job_index]
        instance_id = str(prediction.get("instance_id") or "").strip()
        display_index = job_index + 1
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        reusable = _read_reusable_resolution_instance_summary(
            instance_dir,
            cache_dir=cache_dir,
            expected_input_metadata=input_metadata,
        )
        if reusable is not None:
            existing, reused_summary_path = reusable
            _write_resolution_instance_summary(instance_dir, existing, cache_dir=cache_dir)
            append_log(f"[reuse] {instance_id} -> {reused_summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=reused_summary_path,
            )
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    existing,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            _cleanup_checkpointed_resolution_docker_images(
                instance_id=instance_id,
                backend=backend,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_docker_images,
            )
            return _ResolutionInstanceResult(index=job_index, summary=existing)
        instance_predictions_path = instance_dir / "predictions.jsonl"
        _write_single_resolution_prediction(
            prediction=prediction,
            out_path=instance_predictions_path,
            backend=backend,
        )
        result_root = instance_dir / "evaluation_results"
        ensure_dir(result_root)
        instance_log_path = instance_dir / "resolution-command.log"
        command = [
            str(_poly_bench_python_executable()),
            str(_POLYBENCH_RESOLUTION_WRAPPER),
            "--dataset-name",
            dataset_name,
            "--predictions-path",
            str(instance_predictions_path),
            "--result-path",
            str(result_root),
            "--num-threads",
            "1",
            *(harness_args or []),
        ]
        append_log(f"[run] {display_index}/{len(valid_predictions)} {instance_id}")
        print(
            f"[resolution:{run_id}] starting backend=swe-polybench instance={instance_id} predictions={instance_predictions_path}",
            flush=True,
        )
        returncode, tail = _run_resolution_command(
            command=command,
            cwd=instance_dir,
            log_path=instance_log_path,
            log_prefix=f"[resolution:{run_id}]",
            env=env,
        )
        try:
            instance_report = _load_poly_resolution_report(instance_dir)
            resolved_ids = [str(item).strip() for item in (instance_report.get("resolved_ids") or []) if str(item).strip()]
            unresolved_ids = [str(item).strip() for item in (instance_report.get("unresolved_ids") or []) if str(item).strip()]
            error_ids = [str(item).strip() for item in (instance_report.get("error_ids") or []) if str(item).strip()]
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": error_ids,
                "log_path": str(instance_log_path),
                "report_path": str(instance_report.get("report_path") or ""),
                "status": _status_for_instance_report(instance_id, resolved_ids, unresolved_ids, error_ids),
                "input_metadata": input_metadata,
            }
        except Exception:
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": [instance_id],
                "log_path": str(instance_log_path),
                "status": "error",
                "input_metadata": input_metadata,
            }
        successful_return = returncode == 0
        _write_resolution_instance_summary(
            instance_dir,
            instance_summary,
            cache_dir=cache_dir if successful_return else None,
        )
        if successful_return:
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    instance_summary,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            _cleanup_checkpointed_resolution_docker_images(
                instance_id=instance_id,
                backend=backend,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_docker_images,
            )
        error = None
        if not successful_return:
            error = ResolutionCommandError(
                message=(
                    f"SWE-PolyBench evaluator failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )
        return _ResolutionInstanceResult(index=job_index, summary=instance_summary, error=error)

    instance_results = _run_resolution_instance_jobs(
        job_count=len(valid_predictions),
        max_workers=max_workers,
        run_one=run_one,
    )
    instance_summaries = [result.summary for result in instance_results]
    first_error = next((result.error for result in instance_results if result.error is not None), None)
    summary = _aggregate_instance_resolution_results(
        instance_summaries=instance_summaries,
        report_path=work_dir / "result.json",
    )
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_poly_bench_python_executable())
    summary["wrapper_path"] = str(_POLYBENCH_RESOLUTION_WRAPPER)
    summary["dataset_subset_path"] = str(dataset_subset_path)
    if first_error is not None:
        summary["error_detail"] = str(first_error)
        summary["exit_code"] = first_error.exit_code
        summary["tail"] = first_error.tail
        summary["_partial_from_error"] = True
    return summary


def _coerce_pro_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return repr(value)
    return value


def _normalized_pro_raw_sample_row(row: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in row.items():
        normalized_key = str(key)
        if normalized_key in {"FAIL_TO_PASS", "PASS_TO_PASS"}:
            normalized_key = normalized_key.lower()
        normalized[normalized_key] = value
    return normalized


def _write_pro_raw_sample_csv(
    *,
    raw_sample_jsonl: Path,
    instance_ids: list[str],
    out_path: Path,
) -> None:
    if not raw_sample_jsonl.exists():
        raise FileNotFoundError(f"SWE-bench Pro raw sample snapshot not found: {raw_sample_jsonl}")
    if not instance_ids:
        raise RuntimeError("SWE-bench Pro raw sample export requires at least one prediction instance_id.")

    wanted = {instance_id for instance_id in instance_ids if instance_id}
    by_id: dict[str, dict[str, object]] = {}
    for raw_row in read_jsonl(raw_sample_jsonl):
        if not isinstance(raw_row, dict):
            continue
        row = _normalized_pro_raw_sample_row(raw_row)
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id in wanted and instance_id not in by_id:
            by_id[instance_id] = row

    missing_ids = [instance_id for instance_id in instance_ids if instance_id and instance_id not in by_id]
    if missing_ids:
        raise RuntimeError(
            "SWE-bench Pro raw sample snapshot is missing selected instances: "
            + ", ".join(missing_ids[:10])
            + (f" ... and {len(missing_ids) - 10} more" if len(missing_ids) > 10 else "")
        )

    missing_columns: dict[str, list[str]] = {}
    for instance_id in instance_ids:
        row = by_id.get(instance_id)
        if row is None:
            continue
        missing = [column for column in _PRO_REQUIRED_SAMPLE_COLUMNS if column not in row]
        if missing:
            missing_columns[instance_id] = missing
    if missing_columns:
        details = "; ".join(
            f"{instance_id}: {', '.join(columns)}"
            for instance_id, columns in list(missing_columns.items())[:5]
        )
        raise RuntimeError(f"SWE-bench Pro raw sample snapshot is missing required columns: {details}")

    fieldnames: list[str] = []
    for column in _PRO_REQUIRED_SAMPLE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for instance_id in instance_ids:
        for column in by_id[instance_id].keys():
            if column not in fieldnames:
                fieldnames.append(column)

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance_id in instance_ids:
            row = by_id[instance_id]
            writer.writerow({column: _coerce_pro_csv_value(row.get(column)) for column in fieldnames})


def _load_pro_prediction_ids(predictions_path: Path) -> list[str]:
    payload = read_json(predictions_path)
    if not isinstance(payload, list):
        raise RuntimeError(f"SWE-bench Pro predictions must be a JSON list: {predictions_path}")
    instance_ids = [
        str(row.get("instance_id") or "").strip()
        for row in payload
        if isinstance(row, dict) and str(row.get("instance_id") or "").strip()
    ]
    if len(instance_ids) != len(payload):
        raise RuntimeError(f"SWE-bench Pro predictions contain rows without instance_id: {predictions_path}")
    return instance_ids


def _load_jsonl_prediction_ids(predictions_path: Path) -> list[str]:
    rows = read_jsonl(predictions_path)
    instance_ids = [
        str(row.get("instance_id") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("instance_id") or "").strip()
    ]
    if len(instance_ids) != len(rows):
        raise RuntimeError(f"Resolution predictions contain JSONL rows without instance_id: {predictions_path}")
    return instance_ids


def _load_multi_prediction_ids(predictions_path: Path) -> list[str]:
    rows = read_jsonl(predictions_path)
    instance_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        org = str(row.get("org") or "").strip()
        repo = str(row.get("repo") or "").strip()
        number = row.get("number")
        if org and repo and number not in (None, ""):
            instance_ids.append(_multi_patch_id(org=org, repo=repo, number=int(number)))
    if len(instance_ids) != len(rows):
        raise RuntimeError(f"Multi-SWE-Bench predictions contain rows without org/repo/number: {predictions_path}")
    return instance_ids


def _load_multi_resolution_report(result_root: Path) -> dict[str, object]:
    report_path = result_root / "final_report.json"
    if not report_path.exists():
        raise RuntimeError(f"Unable to locate Multi-SWE-Bench final_report.json under {result_root}")
    payload = read_json(report_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Multi-SWE-Bench final_report.json must contain an object: {report_path}")
    resolved_ids = [_multi_context_id_from_report_id(item) for item in (payload.get("resolved_ids") or []) if str(item).strip()]
    unresolved_ids = [_multi_context_id_from_report_id(item) for item in (payload.get("unresolved_ids") or []) if str(item).strip()]
    error_ids = [_multi_context_id_from_report_id(item) for item in (payload.get("error_ids") or []) if str(item).strip()]
    return {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "resolved_count": int(payload.get("resolved_instances") or len(resolved_ids)),
        "total_instances": int(payload.get("total_instances") or len(resolved_ids) + len(unresolved_ids) + len(error_ids)),
        "completed_instances": int(payload.get("completed_instances") or len(resolved_ids) + len(unresolved_ids)),
        "incomplete_instances": int(payload.get("incomplete_instances") or len(error_ids)),
        "empty_patch_instances": int(payload.get("empty_patch_instances") or 0),
        "submitted_ids": [_multi_context_id_from_report_id(item) for item in (payload.get("submitted_ids") or []) if str(item).strip()],
        "completed_ids": [_multi_context_id_from_report_id(item) for item in (payload.get("completed_ids") or []) if str(item).strip()],
        "incomplete_ids": [_multi_context_id_from_report_id(item) for item in (payload.get("incomplete_ids") or []) if str(item).strip()],
        "empty_patch_ids": [_multi_context_id_from_report_id(item) for item in (payload.get("empty_patch_ids") or []) if str(item).strip()],
        "report_path": str(report_path),
    }


def _load_pro_resolution_report(result_root: Path) -> dict[str, object]:
    report_path = result_root / "eval_results.json"
    if not report_path.exists():
        raise RuntimeError(f"Unable to locate SWE-bench Pro eval_results.json under {result_root}")
    payload = read_json(report_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"SWE-bench Pro eval_results.json must contain an object: {report_path}")
    resolved_ids = sorted(str(instance_id) for instance_id, resolved in payload.items() if bool(resolved))
    unresolved_ids = sorted(str(instance_id) for instance_id, resolved in payload.items() if not bool(resolved))
    return {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "resolved_count": len(resolved_ids),
        "report_path": str(report_path),
    }


def run_pro_resolution_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    self_clean_resolution_artifacts: bool = True,
    self_clean_resolution_docker_images: bool = True,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    raw_sample_path = work_dir / "raw-sample.csv"
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_payload = read_json(predictions_path)
    if not isinstance(prediction_payload, list):
        raise RuntimeError(f"SWE-bench Pro predictions must be a JSON list: {predictions_path}")
    valid_predictions = [
        prediction
        for prediction in prediction_payload
        if isinstance(prediction, dict) and str(prediction.get("instance_id") or "").strip()
    ]
    backend = _resolution_backend_for_bench("Pro")
    log_lock = threading.Lock()

    def append_log(text: str) -> None:
        with log_lock:
            _append_text_log(log_path, text)

    def run_one(job_index: int) -> _ResolutionInstanceResult:
        prediction = valid_predictions[job_index]
        instance_id = str(prediction.get("instance_id") or "").strip()
        display_index = job_index + 1
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        reusable = _read_reusable_resolution_instance_summary(
            instance_dir,
            cache_dir=cache_dir,
            expected_input_metadata=input_metadata,
        )
        if reusable is not None:
            existing, reused_summary_path = reusable
            _write_resolution_instance_summary(instance_dir, existing, cache_dir=cache_dir)
            append_log(f"[reuse] {instance_id} -> {reused_summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=reused_summary_path,
            )
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    existing,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            return _ResolutionInstanceResult(index=job_index, summary=existing)

        instance_predictions_path = instance_dir / "predictions.json"
        _write_single_resolution_prediction(
            prediction=prediction,
            out_path=instance_predictions_path,
            backend=backend,
        )
        result_root = instance_dir / "evaluation_results"
        ensure_dir(result_root)
        instance_log_path = instance_dir / "resolution-command.log"
        command = [
            str(_pro_bench_python_executable()),
            str(_PROBENCH_RESOLUTION_WRAPPER),
            "--patch_path",
            str(instance_predictions_path),
            "--output_dir",
            str(result_root),
            "--num_workers",
            "1",
            "--dockerhub_username",
            _PRO_BENCH_DOCKERHUB_USERNAME,
            "--use_local_docker",
            *(harness_args or []),
        ]
        if cache_dir is not None:
            command.extend(["--cache_dir", str(cache_dir.resolve())])
        if cache_dir is not None and self_clean_resolution_artifacts:
            command.append("--self-clean-resolution-artifacts")
        append_log(f"[run] {display_index}/{len(valid_predictions)} {instance_id}")
        print(
            f"[resolution:{run_id}] starting backend=swebench-pro instance={instance_id} predictions={instance_predictions_path}",
            flush=True,
        )
        command_env = {**(env or {}), "CONTEXTBENCH_PROBENCH_ROOT": str(_PRO_BENCH_ROOT)}
        returncode, tail = _run_resolution_command(
            command=command,
            cwd=instance_dir,
            log_path=instance_log_path,
            log_prefix=f"[resolution:{run_id}]",
            env=command_env,
        )
        try:
            instance_report = _load_pro_resolution_report(result_root)
            resolved_ids = [str(item).strip() for item in (instance_report.get("resolved_ids") or []) if str(item).strip()]
            unresolved_ids = [str(item).strip() for item in (instance_report.get("unresolved_ids") or []) if str(item).strip()]
            error_ids = [str(item).strip() for item in (instance_report.get("error_ids") or []) if str(item).strip()]
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": error_ids,
                "log_path": str(instance_log_path),
                "report_path": str(instance_report.get("report_path") or ""),
                "status": _status_for_instance_report(instance_id, resolved_ids, unresolved_ids, error_ids),
                "input_metadata": input_metadata,
            }
        except Exception:
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": [instance_id],
                "log_path": str(instance_log_path),
                "status": "error",
                "input_metadata": input_metadata,
            }
        successful_return = returncode == 0
        _write_resolution_instance_summary(
            instance_dir,
            instance_summary,
            cache_dir=cache_dir if successful_return else None,
        )
        if successful_return:
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    instance_summary,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
        error = None
        if not successful_return:
            error = ResolutionCommandError(
                message=(
                    f"SWE-bench Pro evaluator failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )
        return _ResolutionInstanceResult(index=job_index, summary=instance_summary, error=error)

    # Pro evaluator Docker images are not currently instance-addressable in a way
    # that is safe to prune here; artifact checkpointing still works per instance.
    del self_clean_resolution_docker_images
    instance_results = _run_resolution_instance_jobs(
        job_count=len(valid_predictions),
        max_workers=max_workers,
        run_one=run_one,
    )
    instance_summaries = [result.summary for result in instance_results]
    first_error = next((result.error for result in instance_results if result.error is not None), None)
    summary = _aggregate_instance_resolution_results(
        instance_summaries=instance_summaries,
        report_path=work_dir / "evaluation_results" / "eval_results.json",
    )
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_pro_bench_python_executable())
    summary["wrapper_path"] = str(_PROBENCH_RESOLUTION_WRAPPER)
    summary["raw_sample_path"] = str(raw_sample_path)
    if first_error is not None:
        summary["error_detail"] = str(first_error)
        summary["exit_code"] = first_error.exit_code
        summary["tail"] = first_error.tail
        summary["_partial_from_error"] = True
    return summary


def run_multi_resolution_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    self_clean_resolution_artifacts: bool = True,
    self_clean_resolution_docker_images: bool = True,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    cache_dir = cache_dir.resolve() if cache_dir is not None else None
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    backend = _resolution_backend_for_bench("Multi")
    valid_predictions = [
        prediction
        for prediction in prediction_rows
        if isinstance(prediction, dict)
        and str(prediction.get("org") or "").strip()
        and str(prediction.get("repo") or "").strip()
        and prediction.get("number") not in (None, "")
    ]
    log_lock = threading.Lock()

    def append_log(text: str) -> None:
        with log_lock:
            _append_text_log(log_path, text)

    def run_one(job_index: int) -> _ResolutionInstanceResult:
        prediction = valid_predictions[job_index]
        instance_id = _multi_patch_id(
            org=str(prediction.get("org") or "").strip(),
            repo=str(prediction.get("repo") or "").strip(),
            number=int(prediction.get("number") or 0),
        )
        display_index = job_index + 1
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        reusable = _read_reusable_resolution_instance_summary(
            instance_dir,
            cache_dir=cache_dir,
            expected_input_metadata=input_metadata,
        )
        if reusable is not None:
            existing, reused_summary_path = reusable
            _write_resolution_instance_summary(instance_dir, existing, cache_dir=cache_dir)
            append_log(f"[reuse] {instance_id} -> {reused_summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=reused_summary_path,
            )
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    existing,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            _cleanup_checkpointed_resolution_docker_images(
                instance_id=instance_id,
                backend=backend,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_docker_images,
            )
            return _ResolutionInstanceResult(index=job_index, summary=existing)

        instance_predictions_path = instance_dir / "predictions.jsonl"
        _write_multi_resolution_predictions_jsonl(
            [
                {
                    "instance_id": instance_id,
                    "model_patch": prediction.get("fix_patch"),
                }
            ],
            instance_predictions_path,
        )
        instance_dataset_path = instance_dir / "dataset.jsonl"
        _write_multi_dataset_jsonl(
            dataset_name=dataset_name,
            instance_ids=[instance_id],
            out_path=instance_dataset_path,
        )
        result_root = instance_dir / "evaluation_results"
        repo_root = instance_dir / "repos"
        log_root = instance_dir / "logs"
        for path in (result_root, repo_root, log_root):
            ensure_dir(path)
        instance_log_path = instance_dir / "resolution-command.log"
        command = [
            str(_multi_bench_python_executable()),
            str(_MULTIBENCH_RESOLUTION_WRAPPER),
            "--predictions-path",
            str(instance_predictions_path),
            "--dataset-path",
            str(instance_dataset_path),
            "--output-dir",
            str(result_root),
            "--repo-dir",
            str(repo_root),
            "--log-dir",
            str(log_root),
            "--max-workers",
            "1",
            *(harness_args or []),
        ]
        append_log(f"[run] {display_index}/{len(valid_predictions)} {instance_id}")
        print(
            f"[resolution:{run_id}] starting backend=multi-swebench instance={instance_id} predictions={instance_predictions_path}",
            flush=True,
        )
        returncode, tail = _run_resolution_command(
            command=command,
            cwd=instance_dir,
            log_path=instance_log_path,
            log_prefix=f"[resolution:{run_id}]",
            env=env,
        )
        try:
            instance_report = _load_multi_resolution_report(result_root)
            resolved_ids = [str(item).strip() for item in (instance_report.get("resolved_ids") or []) if str(item).strip()]
            unresolved_ids = [str(item).strip() for item in (instance_report.get("unresolved_ids") or []) if str(item).strip()]
            error_ids = [str(item).strip() for item in (instance_report.get("error_ids") or []) if str(item).strip()]
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": error_ids,
                "log_path": str(instance_log_path),
                "report_path": str(instance_report.get("report_path") or ""),
                "status": _status_for_instance_report(instance_id, resolved_ids, unresolved_ids, error_ids),
                "input_metadata": input_metadata,
            }
        except Exception:
            instance_summary = {
                "instance_id": instance_id,
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": [instance_id],
                "log_path": str(instance_log_path),
                "status": "error",
                "input_metadata": input_metadata,
            }
        successful_return = returncode == 0
        _write_resolution_instance_summary(
            instance_dir,
            instance_summary,
            cache_dir=cache_dir if successful_return else None,
        )
        if successful_return:
            if _cleanup_checkpointed_resolution_instance(
                instance_dir,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_artifacts,
            ):
                _mark_resolution_summary_artifacts_cleaned(
                    instance_summary,
                    cache_dir=cache_dir,
                    instance_id=instance_id,
                )
            _cleanup_checkpointed_resolution_docker_images(
                instance_id=instance_id,
                backend=backend,
                cache_dir=cache_dir,
                input_metadata=input_metadata,
                log_path=log_path,
                enabled=self_clean_resolution_docker_images,
            )
        error = None
        if not successful_return:
            error = ResolutionCommandError(
                message=(
                    f"Multi-SWE-Bench evaluator failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )
        return _ResolutionInstanceResult(index=job_index, summary=instance_summary, error=error)

    instance_results = _run_resolution_instance_jobs(
        job_count=len(valid_predictions),
        max_workers=max_workers,
        run_one=run_one,
    )
    instance_summaries = [result.summary for result in instance_results]
    first_error = next((result.error for result in instance_results if result.error is not None), None)
    summary = _aggregate_instance_resolution_results(
        instance_summaries=instance_summaries,
        report_path=work_dir / "final_report.json",
    )
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_multi_bench_python_executable())
    summary["wrapper_path"] = str(_MULTIBENCH_RESOLUTION_WRAPPER)
    if first_error is not None:
        summary["error_detail"] = str(first_error)
        summary["exit_code"] = first_error.exit_code
        summary["tail"] = first_error.tail
        summary["_partial_from_error"] = True
    return summary


def _pro_dockerhub_image_uri(instance_id: str, repo_name: str) -> str:
    if "/" not in repo_name:
        raise RuntimeError(f"SWE-bench Pro raw sample repo is not owner/name: {repo_name!r}")
    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    tag_part = instance_id.replace("instance_", "")
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if tag_part.endswith("-vnan"):
            tag_part = tag_part[:-5]
    elif tag_part.endswith("-vnan"):
        tag_part = tag_part[:-5]
    tag = f"{repo_base}.{repo_name_only}-{tag_part}"
    if len(tag) > 128:
        tag = tag[:128]
    return f"{_PRO_BENCH_DOCKERHUB_USERNAME}/sweap-images:{tag}"


def _prepare_swebench_resolution_images(
    *,
    predictions_path: Path,
    dataset_name: str,
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    instance_ids = _load_jsonl_prediction_ids(predictions_path)
    log_path = work_dir / "image-prebuild-command.log"
    if not instance_ids:
        return {"status": "skipped", "reason": "no predictions", "instance_count": 0}
    command = [
        str(_swe_bench_python_executable()),
        "-m",
        "swebench.harness.prepare_images",
        "--dataset_name",
        dataset_name,
        "--split",
        "test",
        "--instance_ids",
        *instance_ids,
        "--max_workers",
        str(max(1, int(max_workers))),
        "--force_rebuild",
        "false",
        "--tag",
        "latest",
        "--env_image_tag",
        "latest",
        "--open_file_limit",
        "4096",
    ]
    returncode, tail = _run_resolution_command(
        command=command,
        cwd=work_dir,
        log_path=log_path,
        log_prefix="[resolution-image-prebuild:swebench]",
        env=env,
    )
    if returncode != 0:
        raise ResolutionCommandError(
            message=f"SWE-bench image preparation failed for {dataset_name}: {tail.strip()}\nFull log: {log_path}",
            exit_code=returncode,
            log_path=str(log_path),
            tail=tail,
        )
    return {
        "status": "completed",
        "backend": "swebench",
        "dataset_name": dataset_name,
        "instance_count": len(instance_ids),
        "max_workers": max(1, int(max_workers)),
        "log_path": str(log_path),
    }


def _prepare_poly_resolution_images(
    *,
    predictions_path: Path,
    dataset_name: str,
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    instance_ids = _load_jsonl_prediction_ids(predictions_path)
    if not instance_ids:
        return {"status": "skipped", "reason": "no predictions", "instance_count": 0}
    result_root = work_dir / "poly-image-prebuild"
    log_path = work_dir / "image-prebuild-command.log"
    command = [
        str(_poly_bench_python_executable()),
        str(_POLYBENCH_RESOLUTION_WRAPPER),
        "--dataset-name",
        dataset_name,
        "--predictions-path",
        str(predictions_path),
        "--result-path",
        str(result_root),
        "--num-threads",
        str(max(1, int(max_workers))),
        "--prepare-images-only",
    ]
    returncode, tail = _run_resolution_command(
        command=command,
        cwd=work_dir,
        log_path=log_path,
        log_prefix="[resolution-image-prebuild:poly]",
        env=env,
    )
    if returncode != 0:
        raise ResolutionCommandError(
            message=f"SWE-PolyBench image preparation failed for {dataset_name}: {tail.strip()}\nFull log: {log_path}",
            exit_code=returncode,
            log_path=str(log_path),
            tail=tail,
        )
    return {
        "status": "completed",
        "backend": "swe-polybench",
        "dataset_name": dataset_name,
        "instance_count": len(instance_ids),
        "max_workers": max(1, int(max_workers)),
        "log_path": str(log_path),
    }


def _prepare_pro_resolution_images(
    *,
    predictions_path: Path,
    dataset_name: str,
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    del dataset_name
    prediction_ids = _load_pro_prediction_ids(predictions_path)
    if not prediction_ids:
        return {"status": "skipped", "reason": "no predictions", "instance_count": 0}
    _write_pro_raw_sample_csv(
        raw_sample_jsonl=_PRO_BENCH_RAW_SAMPLE_JSONL,
        instance_ids=prediction_ids,
        out_path=work_dir / "image-prebuild-raw-sample.csv",
    )
    by_id: dict[str, dict[str, object]] = {}
    wanted = set(prediction_ids)
    for raw_row in read_jsonl(_PRO_BENCH_RAW_SAMPLE_JSONL):
        if not isinstance(raw_row, dict):
            continue
        row = _normalized_pro_raw_sample_row(raw_row)
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id in wanted and instance_id not in by_id:
            by_id[instance_id] = row
    missing_raw_rows = [instance_id for instance_id in prediction_ids if instance_id not in by_id]
    if missing_raw_rows:
        raise RuntimeError(
            "SWE-bench Pro raw sample snapshot is missing selected instances: "
            + ", ".join(missing_raw_rows[:10])
            + (f" ... and {len(missing_raw_rows) - 10} more" if len(missing_raw_rows) > 10 else "")
        )
    images = [
        {
            "instance_id": instance_id,
            "image": _pro_dockerhub_image_uri(instance_id, str(by_id[instance_id].get("repo") or "")),
        }
        for instance_id in prediction_ids
    ]
    image_root = work_dir / "pro-image-prebuild"
    ensure_dir(image_root)

    def pull_one(index: int) -> _ResolutionInstanceResult:
        row = images[index]
        instance_id = str(row["instance_id"])
        image = str(row["image"])
        log_path = image_root / f"{safe_path_component(instance_id)}.log"
        command = ["docker", "pull", image]
        returncode, tail = _run_resolution_command(
            command=command,
            cwd=image_root,
            log_path=log_path,
            log_prefix="[resolution-image-prebuild:pro]",
            env=env,
        )
        summary = {
            "instance_id": instance_id,
            "image": image,
            "status": "completed" if returncode == 0 else "failed",
            "log_path": str(log_path),
        }
        error = None
        if returncode != 0:
            error = ResolutionCommandError(
                message=f"SWE-bench Pro image pull failed for {instance_id}: {tail.strip()}\nFull log: {log_path}",
                exit_code=returncode,
                log_path=str(log_path),
                tail=tail,
            )
        return _ResolutionInstanceResult(index=index, summary=summary, error=error)

    results = _run_resolution_instance_jobs(
        job_count=len(images),
        max_workers=max_workers,
        run_one=pull_one,
    )
    first_error = next((result.error for result in results if result.error is not None), None)
    summary = {
        "status": "completed" if first_error is None else "failed",
        "backend": "swebench-pro",
        "instance_count": len(images),
        "max_workers": max(1, int(max_workers)),
        "images": [result.summary for result in results],
    }
    write_json(image_root / "summary.json", summary)
    if first_error is not None:
        raise first_error
    return {**summary, "summary_path": str(image_root / "summary.json")}


def _prepare_multi_resolution_images(
    *,
    predictions_path: Path,
    dataset_name: str,
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    instance_ids = _load_multi_prediction_ids(predictions_path)
    if not instance_ids:
        return {"status": "skipped", "reason": "no predictions", "instance_count": 0}
    image_root = work_dir / "multi-image-prebuild"
    dataset_path = image_root / "dataset.jsonl"
    _write_multi_dataset_jsonl(
        dataset_name=dataset_name,
        instance_ids=instance_ids,
        out_path=dataset_path,
    )
    result_root = image_root / "evaluation_results"
    repo_root = image_root / "repos"
    log_root = image_root / "logs"
    for path in (result_root, repo_root, log_root):
        ensure_dir(path)
    log_path = work_dir / "image-prebuild-command.log"
    command = [
        str(_multi_bench_python_executable()),
        str(_MULTIBENCH_RESOLUTION_WRAPPER),
        "--predictions-path",
        str(predictions_path),
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(result_root),
        "--repo-dir",
        str(repo_root),
        "--log-dir",
        str(log_root),
        "--max-workers",
        str(max(1, int(max_workers))),
        "--prepare-images-only",
    ]
    returncode, tail = _run_resolution_command(
        command=command,
        cwd=image_root,
        log_path=log_path,
        log_prefix="[resolution-image-prebuild:multi]",
        env=env,
    )
    if returncode != 0:
        raise ResolutionCommandError(
            message=f"Multi-SWE-Bench image preparation failed for {dataset_name}: {tail.strip()}\nFull log: {log_path}",
            exit_code=returncode,
            log_path=str(log_path),
            tail=tail,
        )
    return {
        "status": "completed",
        "backend": "multi-swebench",
        "dataset_name": dataset_name,
        "instance_count": len(instance_ids),
        "max_workers": max(1, int(max_workers)),
        "log_path": str(log_path),
    }


def prepare_resolution_images_for_bench(
    *,
    bench: str,
    predictions_path: Path,
    dataset_name: str,
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    ensure_dir(work_dir)
    if bench == "Verified":
        return _prepare_swebench_resolution_images(
            predictions_path=predictions_path,
            dataset_name=dataset_name,
            work_dir=work_dir,
            max_workers=max_workers,
            env=env,
        )
    if bench == "Poly":
        return _prepare_poly_resolution_images(
            predictions_path=predictions_path,
            dataset_name=dataset_name,
            work_dir=work_dir,
            max_workers=max_workers,
            env=env,
        )
    if bench == "Pro":
        return _prepare_pro_resolution_images(
            predictions_path=predictions_path,
            dataset_name=dataset_name,
            work_dir=work_dir,
            max_workers=max_workers,
            env=env,
        )
    if bench == "Multi":
        return _prepare_multi_resolution_images(
            predictions_path=predictions_path,
            dataset_name=dataset_name,
            work_dir=work_dir,
            max_workers=max_workers,
            env=env,
        )
    raise RuntimeError(f"No resolution image preparation hook is configured for bench {bench!r}")


def prepare_resolution_images_for_tasks(
    *,
    tasks: list[dict[str, object]],
    work_dir: Path,
    max_workers: int,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    work_dir = work_dir.resolve()
    ensure_dir(work_dir)
    if not tasks:
        return {"status": "skipped", "reason": "no tasks", "task_count": 0, "benches": {}, "images": []}

    tasks_by_bench: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        bench = str(task.get("bench") or task.get("source") or "Verified").strip() or "Verified"
        tasks_by_bench.setdefault(bench, []).append(task)

    per_bench: dict[str, dict[str, object]] = {}
    images: list[dict[str, object]] = []
    worker_count = max(1, int(max_workers or 1))
    for bench, bench_tasks in sorted(tasks_by_bench.items()):
        backend = _resolution_backend_for_bench(bench)
        status, message = _resolution_backend_availability(backend)
        if status != "available":
            raise RuntimeError(
                f"Resolution runtime image preparation requires backend {backend.backend!r} for {bench}, "
                f"but it is {status}: {message}"
            )
        predictions = [
            {
                "instance_id": resolution_instance_id_from_task(task),
                "model_patch": "",
                "model_name_or_path": "contextbench-runtime-image-prebuild",
            }
            for task in bench_tasks
        ]
        bench_work_dir = work_dir / safe_path_component(bench.lower())
        predictions_path = _resolution_predictions_path(
            predictions_root=bench_work_dir,
            bench=bench,
            backend=backend,
        )
        _write_backend_resolution_predictions(
            predictions=predictions,
            out_path=predictions_path,
            backend=backend,
            expected_agent="contextbench-runtime-image-prebuild",
        )
        bench_summary = prepare_resolution_images_for_bench(
            bench=bench,
            predictions_path=predictions_path,
            dataset_name=str(backend.dataset_name or ""),
            work_dir=bench_work_dir / "image-prebuild",
            max_workers=worker_count,
            env=env,
        )
        bench_summary["predictions_path"] = str(predictions_path)
        per_bench[bench] = bench_summary
        images.extend(
            {
                "bench": bench,
                "instance_id": resolution_instance_id_from_task(task),
                "image": resolution_image_ref_for_task(task),
            }
            for task in bench_tasks
        )

    return {
        "status": "completed",
        "scope": "resolution_runtime_images",
        "task_count": len(tasks),
        "bench_count": len(per_bench),
        "max_workers": worker_count,
        "benches": per_bench,
        "images": images,
    }


def _resolution_backend_for_bench(bench: str) -> ResolutionBackend:
    dataset_name = _BENCH_TO_RESOLUTION_DATASET.get(bench)
    if bench == "Verified":
        return ResolutionBackend(
            backend="swebench",
            dataset_name=dataset_name,
            module_name="swebench.harness.run_evaluation",
            export_format="jsonl-instance-patch",
            run_evaluation=run_resolution_evaluation,
            python_executable=_swe_bench_python_executable(),
            wrapper_path=_SWEBENCH_RESOLUTION_WRAPPER,
            setup_command="python3 -m contextbench.run_suites_setup swebench",
        )
    if bench == "Poly":
        return ResolutionBackend(
            backend="swe-polybench",
            dataset_name=dataset_name,
            module_name="poly_bench_evaluation.run_evaluation",
            export_format="jsonl-instance-patch",
            run_evaluation=run_poly_resolution_evaluation,
            python_executable=_poly_bench_python_executable(),
            wrapper_path=_POLYBENCH_RESOLUTION_WRAPPER,
            setup_command="python3 -m contextbench.run_suites_setup polybench",
        )
    if bench == "Pro":
        return ResolutionBackend(
            backend="swebench-pro",
            dataset_name=dataset_name,
            module_name=None,
            export_format="json-list-instance-patch-prefix",
            run_evaluation=run_pro_resolution_evaluation,
            python_executable=_pro_bench_python_executable(),
            wrapper_path=_PROBENCH_RESOLUTION_WRAPPER,
            setup_command="python3 -m contextbench.run_suites_setup probench",
        )
    if bench == "Multi":
        return ResolutionBackend(
            backend="multi-swebench",
            dataset_name=dataset_name,
            module_name="multi_swe_bench",
            export_format="jsonl-org-repo-number-fix-patch",
            run_evaluation=run_multi_resolution_evaluation,
            python_executable=_multi_bench_python_executable(),
            wrapper_path=_MULTIBENCH_RESOLUTION_WRAPPER,
            setup_command="python3 -m contextbench.run_suites_setup multibench",
        )
    return _unsupported_resolution_backend(
        bench,
        dataset_name=dataset_name,
        reason="No resolution backend is configured for this bench.",
    )


def _resolution_backend_availability(backend: ResolutionBackend) -> tuple[str, str | None]:
    if backend.unsupported_reason:
        return "unsupported_backend", backend.unsupported_reason
    if backend.requires_docker and not _docker_available():
        return (
            "backend_unavailable",
            f"Resolution backend '{backend.backend}' requires a reachable Docker daemon.",
        )
    if backend.python_executable is not None and not backend.python_executable.exists():
        setup_hint = backend.setup_command or "python3 -m contextbench.run_suites_setup resolution-envs"
        return (
            "backend_unavailable",
            f"Resolution backend '{backend.backend}' requires evaluator Python '{backend.python_executable}'. Run '{setup_hint}'.",
        )
    if backend.wrapper_path is not None and not backend.wrapper_path.exists():
        return (
            "backend_unavailable",
            f"Resolution backend '{backend.backend}' requires wrapper script '{backend.wrapper_path}'.",
        )
    if backend.module_name and backend.python_executable is not None and not _module_available_with_python(
        backend.module_name,
        backend.python_executable,
    ):
        setup_hint = backend.setup_command or "python3 -m contextbench.run_suites_setup resolution-envs"
        return (
            "backend_unavailable",
            f"Resolution backend '{backend.backend}' requires Python module '{backend.module_name}' in '{backend.python_executable}'. Run '{setup_hint}'.",
        )
    if backend.backend == "swebench-pro":
        setup_hint = backend.setup_command or "python3 -m contextbench.run_suites_setup probench"
        for required_path in (_PRO_BENCH_EVALUATOR, _PRO_BENCH_RUN_SCRIPTS, _PRO_BENCH_DOCKERFILES, _PRO_BENCH_RAW_SAMPLE_JSONL):
            if not required_path.exists():
                return (
                    "backend_unavailable",
                    f"Resolution backend '{backend.backend}' requires '{required_path}'. Run '{setup_hint}'.",
                )
    return "available", None


def describe_resolution_backend_support(benches: list[str]) -> list[dict[str, object]]:
    descriptions: list[dict[str, object]] = []
    for bench in benches:
        backend = _resolution_backend_for_bench(bench)
        status, message = _resolution_backend_availability(backend)
        descriptions.append(
            {
                "bench": bench,
                "backend": backend.backend,
                "status": status,
                "dataset_name": backend.dataset_name,
                "message": message,
                "python_executable": str(backend.python_executable) if backend.python_executable else None,
                "wrapper_path": str(backend.wrapper_path) if backend.wrapper_path else None,
            }
        )
    return descriptions


def evaluate_resolution_for_suite(
    *,
    source_dir: Path,
    expected_agent: str,
    suite_name: str,
    variant_name: str,
    work_dir: Path,
    max_workers: int,
    prebuild_images: bool = False,
    prebuild_workers: int | None = None,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    swebench_timeout: int = 1800,
    self_clean_resolution_artifacts: bool = True,
    self_clean_resolution_docker_images: bool = True,
    run_suffix: str | None = None,
    resume_existing_resolution: bool = False,
    clean_resolution_artifacts: bool = False,
) -> dict[str, object]:
    task_rows = _task_result_rows_for_source_dir(source_dir)
    benches = sorted({str(row.get("bench") or "").strip() for row in task_rows if str(row.get("bench") or "").strip()})
    predictions_root = work_dir / "resolution-exports"
    eval_root = work_dir / "resolution-eval"
    checkpoint_root = work_dir / "resolution-checkpoints"
    if clean_resolution_artifacts and not resume_existing_resolution:
        for path in (predictions_root, eval_root):
            if path.exists():
                shutil.rmtree(path)
    ensure_dir(predictions_root)
    ensure_dir(eval_root)

    per_bench: dict[str, dict[str, object]] = {}
    total_tasks = 0
    total_predictions = 0
    evaluated_task_count = 0
    evaluated_prediction_count = 0
    total_resolved = 0
    successful_benches: list[str] = []
    failed_benches: list[str] = []
    unsupported_benches: list[str] = []
    supported_benches: list[str] = []
    partial_benches: list[str] = []
    for bench in benches:
        backend = _resolution_backend_for_bench(bench)
        availability_status, availability_message = _resolution_backend_availability(backend)
        export_summary = collect_resolution_predictions(
            source_dir=source_dir,
            expected_agent=expected_agent,
            bench=bench,
        )
        task_count = int(export_summary["task_count"])
        prediction_count = int(export_summary["prediction_count"])
        missing_patch_count = int(export_summary["missing_patch_count"])
        skipped_ineligible_count = int(export_summary["skipped_ineligible_count"])
        no_patch_ids = [str(item).strip() for item in (export_summary.get("no_patch_ids") or []) if str(item).strip()]
        total_tasks += task_count
        total_predictions += prediction_count
        bench_is_partial = bool(task_count and skipped_ineligible_count)
        if bench_is_partial:
            partial_benches.append(bench)
        run_id = _resolution_run_id(
            eval_root=eval_root,
            suite_name=suite_name,
            variant_name=variant_name,
            bench=bench,
            run_suffix=run_suffix,
            resume_existing=resume_existing_resolution,
        )
        bench_eval_dir = eval_root / bench.lower() / run_id
        bench_checkpoint_dir = checkpoint_root / bench.lower()
        bench_summary: dict[str, object] = {
            "bench": bench,
            "backend": backend.backend,
            "dataset_name": backend.dataset_name,
            "status": None,
            "task_count": task_count,
            "prediction_count": prediction_count,
            "resolved_count": 0,
            "pass_at_1": None,
            "prediction_ids": list(export_summary.get("prediction_ids") or []),
            "selected_ids": list(export_summary.get("selected_ids") or []),
            "resolved_ids": [],
            "unresolved_ids": [],
            "no_patch_ids": no_patch_ids,
            "unknown_ids": [],
            "coverage_of_attempted_tasks": export_summary["coverage_of_attempted_tasks"],
            "is_partial": bench_is_partial,
            "missing_patch_count": missing_patch_count,
            "skipped_ineligible_count": skipped_ineligible_count,
            "skipped_ineligible_reasons": export_summary["skipped_ineligible_reasons"],
            "error_detail": None,
            "predictions_path": None,
            "evaluation_dir": str(bench_eval_dir.resolve()),
            "error_summary_path": None,
            "log_path": None,
            "scope": "resolution_predictions",
        }

        if availability_status == "unsupported_backend":
            bench_summary["status"] = "unsupported_backend"
            bench_summary["error_detail"] = availability_message
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            unsupported_benches.append(bench)
            per_bench[bench] = bench_summary
            continue
        supported_benches.append(bench)
        if prediction_count <= 0 and no_patch_ids and len(no_patch_ids) == task_count and not bench_is_partial:
            evaluated_count = len(no_patch_ids)
            evaluated_task_count += evaluated_count
            evaluated_prediction_count += evaluated_count
            successful_benches.append(bench)
            bench_summary["status"] = "completed"
            bench_summary["unresolved_ids"] = no_patch_ids
            bench_summary["evaluated_task_count"] = evaluated_count
            bench_summary["resolved_count"] = 0
            bench_summary["pass_at_1"] = 0.0 if task_count else None
            bench_summary["pass_at_1_on_evaluated"] = 0.0 if evaluated_count else None
            bench_summary["pass_at_1_on_selected"] = 0.0 if task_count else None
            bench_summary["is_partial"] = False
            per_bench[bench] = bench_summary
            continue
        if prediction_count <= 0:
            bench_summary["status"] = "no_predictions"
            bench_summary["error_detail"] = "No patch-producing predictions were available for this bench."
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            failed_benches.append(bench)
            per_bench[bench] = bench_summary
            continue
        if availability_status == "backend_unavailable":
            bench_summary["status"] = "backend_unavailable"
            bench_summary["error_detail"] = availability_message
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            failed_benches.append(bench)
            per_bench[bench] = bench_summary
            continue
        predictions_path = _resolution_predictions_path(
            predictions_root=predictions_root,
            bench=bench,
            backend=backend,
        )
        _write_backend_resolution_predictions(
            predictions=list(export_summary.get("predictions") or []),
            out_path=predictions_path,
            backend=backend,
            expected_agent=expected_agent,
        )
        bench_summary["predictions_path"] = str(predictions_path)
        if prebuild_images:
            bench_summary["image_prebuild"] = prepare_resolution_images_for_bench(
                bench=bench,
                predictions_path=predictions_path,
                dataset_name=str(backend.dataset_name or ""),
                work_dir=bench_eval_dir / "image-prebuild",
                max_workers=prebuild_workers or max_workers,
                env=env,
            )
        stale_error_path = bench_eval_dir / "resolution-error.json"
        if stale_error_path.exists():
            stale_error_path.unlink()

        try:
            run_kwargs = {
                "predictions_path": predictions_path,
                "dataset_name": str(backend.dataset_name or ""),
                "run_id": run_id,
                "work_dir": bench_eval_dir,
                "max_workers": max_workers,
                "harness_args": harness_args,
                "env": env,
                "cache_dir": bench_checkpoint_dir,
                "self_clean_resolution_artifacts": self_clean_resolution_artifacts,
                "self_clean_resolution_docker_images": self_clean_resolution_docker_images,
            }
            if backend.backend == "swebench":
                run_kwargs["swebench_timeout"] = int(swebench_timeout)
            resolution_summary = backend.run_evaluation(**run_kwargs) if backend.run_evaluation is not None else {}
        except ResolutionCommandError as exc:
            partial_summary: dict[str, object] | None = None
            try:
                if backend.backend == "swebench":
                    partial_summary = _load_resolution_report(bench_eval_dir)
                elif backend.backend == "swe-polybench":
                    partial_summary = _load_poly_resolution_report(bench_eval_dir)
                elif backend.backend == "swebench-pro":
                    partial_summary = _load_pro_resolution_report(bench_eval_dir)
                elif backend.backend == "multi-swebench":
                    partial_summary = _load_multi_resolution_report(bench_eval_dir)
            except Exception:
                partial_summary = None

            if partial_summary is None:
                bench_summary["status"] = "failed"
                bench_summary["error_detail"] = str(exc)
                bench_summary["log_path"] = exc.log_path
                bench_summary["error_summary_path"] = _write_resolution_error_summary(
                    work_dir=bench_eval_dir,
                    payload={
                        **dict(bench_summary),
                        "exit_code": exc.exit_code,
                        "tail": exc.tail,
                    },
                )
                failed_benches.append(bench)
                per_bench[bench] = bench_summary
                continue

            resolution_summary = dict(partial_summary)
            resolution_summary["_partial_from_error"] = True
            resolution_summary["log_path"] = exc.log_path
            resolution_summary["error_detail"] = str(exc)
            resolution_summary["exit_code"] = exc.exit_code
            resolution_summary["tail"] = exc.tail
        except Exception as exc:
            bench_summary["status"] = "failed"
            bench_summary["error_detail"] = str(exc)
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            failed_benches.append(bench)
            per_bench[bench] = bench_summary
            continue

        if bool(resolution_summary.get("_partial_from_error")):
            bench_summary.update(resolution_summary)
            bench_summary["status"] = "failed"
            bench_summary["is_partial"] = True
            bench_summary["pass_at_1"] = None
            bench_summary["error_detail"] = bench_summary.get("error_detail") or (
                "Resolution backend returned a partial report after a nonzero evaluator exit."
            )
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            failed_benches.append(bench)
            if bench not in partial_benches:
                partial_benches.append(bench)
            per_bench[bench] = bench_summary
            continue

        resolved_ids = [str(item).strip() for item in (resolution_summary.get("resolved_ids") or []) if str(item).strip()]
        raw_unresolved_ids = [str(item).strip() for item in (resolution_summary.get("unresolved_ids") or []) if str(item).strip()]
        unresolved_ids = list(raw_unresolved_ids)
        unresolved_seen = set(unresolved_ids)
        for no_patch_id in no_patch_ids:
            if no_patch_id not in unresolved_seen:
                unresolved_ids.append(no_patch_id)
                unresolved_seen.add(no_patch_id)
        error_ids = [str(item).strip() for item in (resolution_summary.get("error_ids") or []) if str(item).strip()]
        submitted_ids = [str(item).strip() for item in (bench_summary.get("prediction_ids") or []) if str(item).strip()]
        unknown_ids = sorted(set(submitted_ids) - set(resolved_ids) - set(unresolved_ids) - set(error_ids))
        backend_reported_ids = set(resolved_ids) | set(raw_unresolved_ids) | set(error_ids)
        extra_ids = sorted(backend_reported_ids - set(submitted_ids))
        attempted_ids = sorted(set(resolved_ids) | set(unresolved_ids) | set(error_ids))
        evaluated_count = len(attempted_ids)
        if evaluated_count <= 0:
            evaluated_count = int(resolution_summary.get("total_instances") or 0)

        total_instances_mismatch = (
            "total_instances" in resolution_summary
            and int(resolution_summary.get("total_instances") or 0) != prediction_count
        )
        if error_ids or unknown_ids or extra_ids or total_instances_mismatch:
            bench_summary.update(resolution_summary)
            bench_summary["status"] = "failed"
            bench_summary["error_ids"] = error_ids
            bench_summary["unknown_ids"] = unknown_ids
            bench_summary["extra_reported_ids"] = extra_ids
            if not bench_summary.get("error_detail"):
                details: list[str] = []
                if error_ids:
                    details.append(
                        "backend error ids: "
                        + ", ".join(error_ids[:5])
                        + (f" ... and {len(error_ids) - 5} more" if len(error_ids) > 5 else "")
                    )
                if unknown_ids:
                    details.append(
                        "missing evaluator result ids: "
                        + ", ".join(unknown_ids[:5])
                        + (f" ... and {len(unknown_ids) - 5} more" if len(unknown_ids) > 5 else "")
                    )
                if extra_ids:
                    details.append(
                        "unexpected evaluator result ids: "
                        + ", ".join(extra_ids[:5])
                        + (f" ... and {len(extra_ids) - 5} more" if len(extra_ids) > 5 else "")
                    )
                if total_instances_mismatch:
                    details.append(
                        "total_instances mismatch: "
                        f"reported={resolution_summary.get('total_instances')} predictions={prediction_count}"
                    )
                bench_summary["error_detail"] = "Resolution backend did not produce complete valid coverage (" + "; ".join(details) + ")."
            bench_summary["log_path"] = resolution_summary.get("log_path")
            bench_summary["error_summary_path"] = _write_resolution_error_summary(
                work_dir=bench_eval_dir,
                payload=dict(bench_summary),
            )
            failed_benches.append(bench)
            per_bench[bench] = bench_summary
            continue

        resolved_count = int(resolution_summary.get("resolved_count") or len(resolved_ids))
        bench_partial = bench_is_partial or bool(error_ids or unknown_ids)
        pass_at_1_on_evaluated = (resolved_count / evaluated_count) if evaluated_count else None
        pass_at_1_on_selected = (resolved_count / task_count) if task_count else None
        total_resolved += resolved_count
        evaluated_task_count += evaluated_count
        evaluated_prediction_count += evaluated_count
        successful_benches.append(bench)
        bench_summary.update(resolution_summary)
        bench_summary["status"] = "completed"
        bench_summary["resolved_ids"] = resolved_ids
        bench_summary["unresolved_ids"] = unresolved_ids
        bench_summary["no_patch_ids"] = no_patch_ids
        bench_summary["unknown_ids"] = unknown_ids
        bench_summary["error_ids"] = error_ids
        bench_summary["evaluated_task_count"] = evaluated_count
        bench_summary["resolved_count"] = resolved_count
        bench_summary["pass_at_1"] = None if bench_partial else pass_at_1_on_selected
        bench_summary["pass_at_1_on_evaluated"] = pass_at_1_on_evaluated
        bench_summary["pass_at_1_on_selected"] = pass_at_1_on_selected
        bench_summary["is_partial"] = bench_partial
        bench_summary["log_path"] = resolution_summary.get("log_path")
        if bench_summary["is_partial"] and bench not in partial_benches:
            partial_benches.append(bench)
        per_bench[bench] = bench_summary

    completed_without_partial = bool(benches and len(successful_benches) == len(benches) and not partial_benches)
    if completed_without_partial:
        overall_status = "completed"
    elif successful_benches:
        overall_status = "partial"
    else:
        overall_status = "failed"
    pass_at_1_on_evaluated = (total_resolved / evaluated_task_count) if evaluated_task_count else None
    pass_at_1_on_selected = (total_resolved / total_tasks) if total_tasks else None

    return {
        "status": overall_status,
        "backend": "mixed",
        "task_count": total_tasks,
        "prediction_count": total_predictions,
        "evaluated_task_count": evaluated_task_count,
        "evaluated_prediction_count": evaluated_prediction_count,
        "resolved_count": total_resolved,
        "pass_at_1": pass_at_1_on_selected if completed_without_partial else None,
        "pass_at_1_on_evaluated": pass_at_1_on_evaluated,
        "pass_at_1_on_selected": pass_at_1_on_selected,
        "coverage_of_attempted_tasks": (total_predictions / total_tasks) if total_tasks else 0.0,
        "evaluated_coverage_of_attempted_tasks": (evaluated_prediction_count / total_tasks) if total_tasks else 0.0,
        "is_partial": bool(benches and (len(successful_benches) < len(benches) or partial_benches)),
        "partial_benches": partial_benches,
        "scope": "resolution_predictions",
        "supported_benches": supported_benches,
        "successful_benches": successful_benches,
        "failed_benches": failed_benches,
        "unsupported_benches": unsupported_benches,
        "per_bench": per_bench,
        "predictions_dir": str(predictions_root),
        "evaluation_dir": str(eval_root),
    }


def _verification_quality_paths(out_path: Path) -> tuple[Path, Path, Path]:
    root = out_path.parent
    return (
        root / "verification-quality.jsonl",
        root / "verification-quality-summary.json",
        root / "verification-quality.csv",
    )


def _write_verification_quality_artifacts(*, out_path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    jsonl_path, summary_path, csv_path = _verification_quality_paths(out_path)
    _write_jsonl_atomic(jsonl_path, rows)

    strongest_counts = Counter(
        str(((row.get("verification_quality") or {}).get("strongest_verification") if isinstance(row.get("verification_quality"), dict) else "") or "unknown")
        for row in rows
    )
    syntax_only_count = sum(
        1
        for row in rows
        if isinstance(row.get("verification_quality"), dict)
        and bool(row["verification_quality"].get("syntax_only"))
    )
    successful_runtime_count = sum(
        1
        for row in rows
        if isinstance(row.get("verification_quality"), dict)
        and bool(row["verification_quality"].get("successful_runtime_verification"))
    )
    environment_limited_count = sum(
        1
        for row in rows
        if isinstance(row.get("verification_quality"), dict)
        and bool(row["verification_quality"].get("environment_limited"))
    )
    added_regression_test_count = sum(
        1
        for row in rows
        if isinstance(row.get("regression_test"), dict)
        and bool(row["regression_test"].get("added_regression_test"))
    )
    added_regression_test_not_run_count = sum(
        1
        for row in rows
        if isinstance(row.get("regression_test"), dict)
        and bool(row["regression_test"].get("added_regression_test"))
        and row["regression_test"].get("regression_tests_run") is False
    )
    summary = {
        "schema_version": 1,
        "scope": "verification_quality",
        "record_count": len(rows),
        "strongest_verification_counts": dict(sorted(strongest_counts.items())),
        "successful_runtime_verification_count": successful_runtime_count,
        "syntax_only_count": syntax_only_count,
        "environment_limited_count": environment_limited_count,
        "added_regression_test_count": added_regression_test_count,
        "added_regression_test_not_run_count": added_regression_test_not_run_count,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }
    write_json(summary_path, summary)

    fieldnames = [
        "instance_id",
        "original_inst_id",
        "bench",
        "record_path",
        "strongest_verification",
        "successful_runtime_verification",
        "syntax_only",
        "environment_limited",
        "commands_total",
        "failed_commands_total",
        "added_regression_test",
        "regression_tests_run",
        "added_test_files",
    ]
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            verification = row.get("verification_quality") if isinstance(row.get("verification_quality"), dict) else {}
            regression = row.get("regression_test") if isinstance(row.get("regression_test"), dict) else {}
            writer.writerow(
                {
                    "instance_id": row.get("instance_id"),
                    "original_inst_id": row.get("original_inst_id"),
                    "bench": row.get("bench"),
                    "record_path": row.get("record_path"),
                    "strongest_verification": verification.get("strongest_verification"),
                    "successful_runtime_verification": verification.get("successful_runtime_verification"),
                    "syntax_only": verification.get("syntax_only"),
                    "environment_limited": verification.get("environment_limited"),
                    "commands_total": verification.get("commands_total"),
                    "failed_commands_total": verification.get("failed_commands_total"),
                    "added_regression_test": regression.get("added_regression_test"),
                    "regression_tests_run": regression.get("regression_tests_run"),
                    "added_test_files": "|".join(str(item) for item in (regression.get("added_test_files") or [])),
                }
            )
    summary["summary_path"] = str(summary_path)
    return summary


def convert_records_to_jsonl(*, source_dir: Path, expected_agent: str, out_path: Path) -> dict[str, object]:
    started_at = time.time()
    print(f"[postprocess] converting {expected_agent} records from {source_dir} -> {out_path}", flush=True)
    ensure_dir(out_path.parent)
    summary: dict[str, object] = {
        "scope": "converted_predictions",
        "selected_task_count": None,
        "record_count": 0,
        "convertible_record_count": 0,
        "prediction_count": 0,
        "missing_record_path_count": 0,
        "nonconvertible_record_count": 0,
        "input_error_count": 0,
        "conversion_error_count": 0,
        "conversion_errors": [],
        "coverage_of_attempted_tasks": None,
        "missing_prediction_count": None,
        "is_partial": False,
    }
    task_results = _task_results_for_source_dir(source_dir) if source_dir.exists() else None
    verification_quality_rows: list[dict[str, object]] = []
    if task_results is not None:
        rows = [row for row in read_jsonl(task_results) if isinstance(row, dict)]
        total = len(rows)
        summary["selected_task_count"] = total
        progress_every = 10 if total >= 50 else 1
        with open(out_path, "w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                record_path_value = row.get("record_path")
                record_path = resolve_record_path(
                    record_path_value,
                    task_results_path=task_results,
                    source_dir=source_dir,
                )
                if record_path is None:
                    summary["missing_record_path_count"] = int(summary["missing_record_path_count"] or 0) + 1
                    continue
                instance_id = row.get("instance_id") or row.get("original_inst_id") or record_path.stem
                print(f"[postprocess] converting record {index}/{total}: {instance_id}", flush=True)
                record = record_with_resolved_artifact_paths(
                    read_json(record_path),
                    record_path=record_path,
                    require_existing_artifacts=True,
                )
                summary["record_count"] = int(summary["record_count"] or 0) + 1
                if _record_matches_agent(record, expected_agent):
                    quality_payload = analyze_record_quality(record)
                    verification_quality_rows.append(
                        {
                            "schema_version": 1,
                            "instance_id": str(record.get("instance_id") or row.get("instance_id") or instance_id),
                            "original_inst_id": str(record.get("original_inst_id") or row.get("original_inst_id") or ""),
                            "bench": str(record.get("bench") or row.get("bench") or ""),
                            "record_path": str(record_path),
                            **quality_payload,
                        }
                    )
                if not record_is_convertible(record, expected_agent=expected_agent):
                    summary["nonconvertible_record_count"] = int(summary["nonconvertible_record_count"] or 0) + 1
                    if index % progress_every == 0 or index == total:
                        print(f"[postprocess] conversion progress {index}/{total}", flush=True)
                    continue
                summary["convertible_record_count"] = int(summary["convertible_record_count"] or 0) + 1
                artifact_path_errors = record.get(_ARTIFACT_PATH_ERRORS_FIELD)
                if isinstance(artifact_path_errors, list) and artifact_path_errors:
                    summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                    summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
                    errors = summary.setdefault("conversion_errors", [])
                    if isinstance(errors, list):
                        errors.append(
                            {
                                "instance_id": str(record.get("instance_id") or instance_id),
                                "record_path": str(record_path),
                                "error": "missing_artifact_path",
                                "artifact_paths": artifact_path_errors,
                            }
                        )
                    if index % progress_every == 0 or index == total:
                        print(f"[postprocess] conversion progress {index}/{total}", flush=True)
                    continue
                try:
                    converted = convert_run_record(record)
                except ContextPathValidationError as exc:
                    summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                    summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
                    errors = summary.setdefault("conversion_errors", [])
                    if isinstance(errors, list):
                        errors.append(
                            {
                                "instance_id": exc.instance_id,
                                "record_path": str(record_path),
                                "error": "invalid_predicted_context_path",
                                "invalid_paths": exc.invalid_paths,
                            }
                        )
                    if index % progress_every == 0 or index == total:
                        print(f"[postprocess] conversion progress {index}/{total}", flush=True)
                    continue
                except Exception as exc:
                    summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                    summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
                    errors = summary.setdefault("conversion_errors", [])
                    if isinstance(errors, list):
                        errors.append(
                            {
                                "instance_id": str(instance_id),
                                "record_path": str(record_path),
                                "error": str(exc),
                            }
                        )
                    if index % progress_every == 0 or index == total:
                        print(f"[postprocess] conversion progress {index}/{total}", flush=True)
                    continue
                handle.write(json.dumps(converted, ensure_ascii=False))
                handle.write("\n")
                summary["prediction_count"] = int(summary["prediction_count"] or 0) + 1
                if index % progress_every == 0 or index == total:
                    print(f"[postprocess] conversion progress {index}/{total}", flush=True)
        task_count = int(summary["selected_task_count"] or 0)
        prediction_count = int(summary["prediction_count"] or 0)
        summary["coverage_of_attempted_tasks"] = (prediction_count / task_count) if task_count else 0.0
        summary["missing_prediction_count"] = max(task_count - prediction_count, 0)
        summary["is_partial"] = bool(task_count and prediction_count < task_count)
    else:
        predictions = load_predictions_from_path(source_dir, expected_agent=expected_agent) if source_dir.exists() else []
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        count = len(predictions)
        summary["record_count"] = count
        summary["convertible_record_count"] = count
        summary["prediction_count"] = count
    quality_summary = _write_verification_quality_artifacts(
        out_path=out_path,
        rows=verification_quality_rows,
    )
    summary["verification_quality_path"] = quality_summary["jsonl_path"]
    summary["verification_quality_summary_path"] = quality_summary["summary_path"]
    summary["verification_quality_csv_path"] = quality_summary["csv_path"]
    summary["verification_quality"] = {
        key: value
        for key, value in quality_summary.items()
        if key not in {"jsonl_path", "summary_path", "csv_path"}
    }
    print(
        f"[postprocess] conversion complete: wrote {int(summary['prediction_count'] or 0)} predictions in {time.time() - started_at:.1f}s",
        flush=True,
    )
    return summary


def evaluate_prediction_file(
    *,
    gold_path: Path,
    pred_path: Path,
    cache_dir: Path,
    out_path: Path,
    selected_task_count: int | None = None,
    workspace_key: str | None = None,
    tmp_root: Path | None = None,
) -> dict[str, object]:
    if not treesitter_available():
        raise RuntimeError("Tree-sitter is not available for evaluation")

    started_at = time.time()
    gold_loader = GoldLoader(str(gold_path))
    pred_rows = load_pred(str(pred_path))
    predictions_by_instance_id = {
        str(row.get("instance_id") or row.get("original_inst_id")): row
        for row in pred_rows
        if row.get("instance_id") or row.get("original_inst_id")
    }
    effective_workspace_key = workspace_key or _default_evaluation_workspace_key(out_path)
    total_rows = len(pred_rows)
    progress_every = 10 if total_rows >= 50 else 1
    print(
        f"[postprocess] evaluating predictions from {pred_path} against {gold_path} ({total_rows} rows)",
        flush=True,
    )
    results: list[dict[str, object]] = []
    for index, pred_data in enumerate(pred_rows, start=1):
        instance_id = pred_data.get("instance_id") or pred_data.get("original_inst_id")
        if not instance_id:
            continue
        gold_ctx = gold_loader.get(instance_id)
        if not gold_ctx:
            results.append({"instance_id": instance_id, "error": "missing_gold"})
            if index % progress_every == 0 or index == total_rows:
                print(f"[postprocess] evaluation progress {index}/{total_rows}", flush=True)
            continue
        results.append(
            evaluate_instance(
                instance_id,
                gold_ctx,
                pred_data,
                str(cache_dir),
                workspace_key=effective_workspace_key,
                tmp_root=str(tmp_root) if tmp_root is not None else None,
            )
        )
        if index % progress_every == 0 or index == total_rows:
            print(f"[postprocess] evaluation progress {index}/{total_rows}", flush=True)

    _assert_evaluation_artifact_consistent(
        results=results,
        predictions_by_instance_id=predictions_by_instance_id,
    )
    _write_jsonl_atomic(out_path, results)

    error_counts = dict(Counter(str(row.get("error")) for row in results if row.get("error")))
    summary = aggregate_results(results)
    summary["error_counts"] = error_counts
    summary["error_count"] = sum(int(value) for value in error_counts.values())
    summary["has_errors"] = bool(error_counts)
    summary["scope"] = "converted_predictions"
    summary["prediction_count"] = total_rows
    summary["evaluated_prediction_count"] = len(results)
    summary["selected_task_count"] = selected_task_count
    summary["coverage_of_attempted_tasks"] = (
        (len(results) / selected_task_count) if selected_task_count else None
    )
    summary["missing_prediction_count"] = (
        max(selected_task_count - len(results), 0) if selected_task_count is not None else None
    )
    summary["is_partial"] = bool(selected_task_count and len(results) < selected_task_count) or bool(error_counts)
    print(
        f"[postprocess] evaluation complete: {summary.get('num_valid')}/{summary.get('num_total')} valid in {time.time() - started_at:.1f}s",
        flush=True,
    )
    return summary
