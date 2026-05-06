# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: resolve task-result record paths relative to suite artifacts before postprocess conversion and add optional SWE-bench-style patch resolution evaluation.

"""Conversion and evaluation helpers for run suites."""

from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
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
    task_results = _task_results_for_source_dir(source_dir) if source_dir.exists() else None
    for row in rows:
        record_path = resolve_record_path(
            row.get("record_path"),
            task_results_path=task_results,
            source_dir=source_dir,
        )
        if record_path is None:
            continue
        record = read_json(record_path)
        if not isinstance(record, dict) or not _record_matches_agent(record, expected_agent):
            continue
        eligibility_error = _resolution_prediction_ineligibility_reason(row=row, record=record)
        if eligibility_error is not None:
            skipped_ineligible_count += 1
            skipped_ineligible_reasons[eligibility_error] += 1
            continue
        prediction_id = _normalize_agent_record_id(record, row)
        if not prediction_id:
            continue
        model_patch = _read_model_patch_for_resolution(record)
        if not model_patch:
            missing_patch_count += 1
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


def _status_for_instance_report(instance_id: str, resolved_ids: list[str], unresolved_ids: list[str], error_ids: list[str]) -> str:
    id_set = {str(instance_id).strip()}
    if id_set & set(resolved_ids):
        return "resolved"
    if id_set & set(error_ids):
        return "error"
    if id_set & set(unresolved_ids):
        return "unresolved"
    return "error"


def _write_resolution_instance_summary(instance_dir: Path, payload: dict[str, object]) -> None:
    write_json(_resolution_instance_summary_path(instance_dir), payload)


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


def _aggregate_instance_resolution_results(
    *,
    instance_summaries: list[dict[str, object]],
    report_path: Path,
) -> dict[str, object]:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    for summary in instance_summaries:
        for target, values in (
            (resolved_ids, summary.get("resolved_ids") or []),
            (unresolved_ids, summary.get("unresolved_ids") or []),
            (error_ids, summary.get("error_ids") or []),
        ):
            seen = set(target)
            for value in values:
                value = str(value).strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                target.append(value)
    aggregate = {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "resolved_count": len(resolved_ids),
        "total_instances": len(resolved_ids) + len(unresolved_ids) + len(error_ids),
        "report_path": str(report_path),
    }
    ensure_dir(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": error_ids,
                "resolved_count": len(resolved_ids),
                "total_instances": len(resolved_ids) + len(unresolved_ids) + len(error_ids),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
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
                "patch": _normalize_model_patch_for_resolution(prediction.get("model_patch")),
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


def run_resolution_evaluation(
    *,
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    instance_summaries: list[dict[str, object]] = []
    first_error: ResolutionCommandError | None = None
    backend = _resolution_backend_for_bench("Verified")
    for index, prediction in enumerate(prediction_rows, start=1):
        if not isinstance(prediction, dict):
            continue
        instance_id = str(prediction.get("instance_id") or "").strip()
        if not instance_id:
            continue
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        summary_path = _resolution_instance_summary_path(instance_dir)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        existing = _read_resolution_instance_summary(instance_dir, expected_input_metadata=input_metadata)
        if existing is not None:
            instance_summaries.append(existing)
            _append_text_log(log_path, f"[reuse] {instance_id} -> {summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=summary_path,
            )
            continue

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
            *(harness_args or []),
        ]
        _append_text_log(log_path, f"[run] {index}/{len(prediction_rows)} {instance_id}")
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
        _write_resolution_instance_summary(instance_dir, instance_summary)
        instance_summaries.append(instance_summary)
        if returncode != 0 and first_error is None:
            first_error = ResolutionCommandError(
                message=(
                    f"SWE-bench harness failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )

    summary = _aggregate_instance_resolution_results(
        instance_summaries=instance_summaries,
        report_path=work_dir / "report.json",
    )
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_swe_bench_python_executable())
    summary["wrapper_path"] = str(_SWEBENCH_RESOLUTION_WRAPPER)
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
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    dataset_subset_path = work_dir / "dataset-subset.csv"
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    instance_summaries: list[dict[str, object]] = []
    first_error: ResolutionCommandError | None = None
    backend = _resolution_backend_for_bench("Poly")
    for index, prediction in enumerate(prediction_rows, start=1):
        if not isinstance(prediction, dict):
            continue
        instance_id = str(prediction.get("instance_id") or "").strip()
        if not instance_id:
            continue
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        existing = _read_resolution_instance_summary(instance_dir, expected_input_metadata=input_metadata)
        if existing is not None:
            instance_summaries.append(existing)
            summary_path = _resolution_instance_summary_path(instance_dir)
            _append_text_log(log_path, f"[reuse] {instance_id} -> {summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=summary_path,
            )
            continue
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
        _append_text_log(log_path, f"[run] {index}/{len(prediction_rows)} {instance_id}")
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
        _write_resolution_instance_summary(instance_dir, instance_summary)
        instance_summaries.append(instance_summary)
        if returncode != 0 and first_error is None:
            first_error = ResolutionCommandError(
                message=(
                    f"SWE-PolyBench evaluator failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )

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
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    result_root = work_dir / "evaluation_results"
    ensure_dir(result_root)
    raw_sample_path = work_dir / "raw-sample.csv"
    log_path = work_dir / "resolution-command.log"
    command = [
        str(_pro_bench_python_executable()),
        str(_PROBENCH_RESOLUTION_WRAPPER),
        "--patch_path",
        str(predictions_path),
        "--output_dir",
        str(result_root),
        "--num_workers",
        str(max_workers),
        "--dockerhub_username",
        _PRO_BENCH_DOCKERHUB_USERNAME,
        "--use_local_docker",
        *(harness_args or []),
    ]
    print(
        f"[resolution:{run_id}] starting backend=swebench-pro dataset={dataset_name} predictions={predictions_path}",
        flush=True,
    )
    command_env = {**(env or {}), "CONTEXTBENCH_PROBENCH_ROOT": str(_PRO_BENCH_ROOT)}
    returncode, tail = _run_resolution_command(
        command=command,
        cwd=work_dir,
        log_path=log_path,
        log_prefix=f"[resolution:{run_id}]",
        env=command_env,
    )
    if returncode != 0:
        raise ResolutionCommandError(
            message=(
                f"SWE-bench Pro evaluator failed for {dataset_name} ({run_id}): "
                f"{tail.strip()}\nFull log: {log_path}"
            ),
            exit_code=returncode,
            log_path=str(log_path),
            tail=tail,
        )
    summary = _load_pro_resolution_report(result_root)
    summary["dataset_name"] = dataset_name
    summary["run_id"] = run_id
    summary["log_path"] = str(log_path)
    summary["python_executable"] = str(_pro_bench_python_executable())
    summary["wrapper_path"] = str(_PROBENCH_RESOLUTION_WRAPPER)
    summary["raw_sample_path"] = str(raw_sample_path)
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
) -> dict[str, object]:
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Resolution predictions not found: {predictions_path}")
    ensure_dir(work_dir)
    log_path = work_dir / "resolution-command.log"
    if log_path.exists():
        log_path.unlink()
    prediction_rows = read_jsonl(predictions_path)
    instance_summaries: list[dict[str, object]] = []
    first_error: ResolutionCommandError | None = None
    backend = _resolution_backend_for_bench("Multi")
    for index, prediction in enumerate(prediction_rows, start=1):
        if not isinstance(prediction, dict):
            continue
        instance_id = _multi_patch_id(
            org=str(prediction.get("org") or "").strip(),
            repo=str(prediction.get("repo") or "").strip(),
            number=int(prediction.get("number") or 0),
        )
        if not instance_id:
            continue
        instance_dir = _resolution_instance_dir(work_dir, instance_id)
        input_metadata = _resolution_instance_input_metadata(
            prediction,
            backend=backend,
            dataset_name=dataset_name,
            harness_args=harness_args,
        )
        existing = _read_resolution_instance_summary(instance_dir, expected_input_metadata=input_metadata)
        if existing is not None:
            instance_summaries.append(existing)
            summary_path = _resolution_instance_summary_path(instance_dir)
            _append_text_log(log_path, f"[reuse] {instance_id} -> {summary_path}")
            _write_reused_resolution_log(
                log_path=instance_dir / "resolution-command.log",
                instance_id=instance_id,
                summary_path=summary_path,
            )
            continue

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
            str(max_workers),
            *(harness_args or []),
        ]
        _append_text_log(log_path, f"[run] {index}/{len(prediction_rows)} {instance_id}")
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
        _write_resolution_instance_summary(instance_dir, instance_summary)
        instance_summaries.append(instance_summary)
        if returncode != 0 and first_error is None:
            first_error = ResolutionCommandError(
                message=(
                    f"Multi-SWE-Bench evaluator failed for {dataset_name} ({run_id}): "
                    f"{tail.strip()}\nFull log: {instance_log_path}"
                ),
                exit_code=returncode,
                log_path=str(instance_log_path),
                tail=tail,
            )

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
    harness_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    run_suffix: str | None = None,
    resume_existing_resolution: bool = False,
    clean_resolution_artifacts: bool = False,
) -> dict[str, object]:
    task_rows = _task_result_rows_for_source_dir(source_dir)
    benches = sorted({str(row.get("bench") or "").strip() for row in task_rows if str(row.get("bench") or "").strip()})
    predictions_root = work_dir / "resolution-exports"
    eval_root = work_dir / "resolution-eval"
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
        total_tasks += task_count
        total_predictions += prediction_count
        bench_is_partial = bool(task_count and prediction_count < task_count)
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
            "resolved_ids": [],
            "unresolved_ids": [],
            "unknown_ids": [],
            "coverage_of_attempted_tasks": export_summary["coverage_of_attempted_tasks"],
            "is_partial": bench_is_partial,
            "missing_patch_count": export_summary["missing_patch_count"],
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
        stale_error_path = bench_eval_dir / "resolution-error.json"
        if stale_error_path.exists():
            stale_error_path.unlink()

        try:
            resolution_summary = backend.run_evaluation(
                predictions_path=predictions_path,
                dataset_name=str(backend.dataset_name or ""),
                run_id=run_id,
                work_dir=bench_eval_dir,
                max_workers=max_workers,
                harness_args=harness_args,
                env=env,
            ) if backend.run_evaluation is not None else {}
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
        error_ids = [str(item).strip() for item in (resolution_summary.get("error_ids") or []) if str(item).strip()]
        submitted_ids = [str(item).strip() for item in (bench_summary.get("prediction_ids") or []) if str(item).strip()]
        unknown_ids = sorted(set(submitted_ids) - set(resolved_ids) - set(unresolved_ids) - set(error_ids))
        extra_ids = sorted((set(resolved_ids) | set(unresolved_ids) | set(error_ids)) - set(submitted_ids))
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
        evaluated_prediction_count += min(prediction_count, evaluated_count)
        successful_benches.append(bench)
        bench_summary.update(resolution_summary)
        bench_summary["status"] = "completed"
        bench_summary["resolved_ids"] = resolved_ids
        bench_summary["unresolved_ids"] = unresolved_ids
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
) -> dict[str, object]:
    if not treesitter_available():
        raise RuntimeError("Tree-sitter is not available for evaluation")

    started_at = time.time()
    gold_loader = GoldLoader(str(gold_path))
    pred_rows = load_pred(str(pred_path))
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
        results.append(evaluate_instance(instance_id, gold_ctx, pred_data, str(cache_dir)))
        if index % progress_every == 0 or index == total_rows:
            print(f"[postprocess] evaluation progress {index}/{total_rows}", flush=True)

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

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
