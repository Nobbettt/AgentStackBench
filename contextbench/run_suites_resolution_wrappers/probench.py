
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROBENCH_ROOT = Path(os.environ.get("CONTEXTBENCH_PROBENCH_ROOT", "/opt/probench"))
PROBENCH_SCRIPT = PROBENCH_ROOT / "swe_bench_pro_eval.py"
PROBENCH_RAW_SAMPLE_JSONL = PROBENCH_ROOT / "helper_code" / "sweap_eval_full_v2.jsonl"
PROBENCH_RUN_SCRIPTS = PROBENCH_ROOT / "run_scripts"
PRO_REQUIRED_SAMPLE_COLUMNS = (
    "instance_id",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "base_commit",
    "repo",
    "fail_to_pass",
    "pass_to_pass",
)
INPUT_METADATA_VERSION = 1


def _normalized_pro_raw_sample_row(row: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in row.items():
        normalized_key = str(key)
        if normalized_key in {"FAIL_TO_PASS", "PASS_TO_PASS"}:
            normalized_key = normalized_key.lower()
        normalized[normalized_key] = value
    return normalized


def _coerce_pro_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return repr(value)
    return value


def _load_prediction_ids(predictions_path: Path) -> list[str]:
    return [str(row.get("instance_id") or "").strip() for row in _load_prediction_rows(predictions_path)]


def _load_prediction_rows(predictions_path: Path) -> list[dict[str, object]]:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"SWE-bench Pro predictions must be a JSON list: {predictions_path}")
    rows = [row for row in payload if isinstance(row, dict) and str(row.get("instance_id") or "").strip()]
    if len(rows) != len(payload):
        raise RuntimeError(f"SWE-bench Pro predictions contain rows without instance_id: {predictions_path}")
    return rows


def _write_prediction_rows(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned or "instance"


def _load_eval_results(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): bool(value) for key, value in payload.items()}


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prediction_metadata(row: dict[str, object]) -> dict[str, object]:
    payload = {
        "instance_id": str(row.get("instance_id") or "").strip(),
        "patch": str(row.get("patch") or ""),
        "prefix": str(row.get("prefix") or ""),
    }
    return {
        "schema_version": INPUT_METADATA_VERSION,
        "backend": "swebench-pro",
        "instance_id": payload["instance_id"],
        "prediction_sha256": _stable_json_hash(payload),
    }


def _metadata_path(instance_dir: Path) -> Path:
    return instance_dir / "resolution-input.json"


def _read_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_matching_metadata(instance_dir: Path, expected_metadata: dict[str, object]) -> bool:
    return _read_metadata(_metadata_path(instance_dir)) == expected_metadata


def _write_metadata(instance_dir: Path, metadata: dict[str, object]) -> None:
    path = _metadata_path(instance_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_instance_error(
    *,
    instance_dir: Path,
    instance_id: str,
    exit_code: int | None,
    detail: str,
) -> Path:
    error_path = instance_dir / "evaluation-error.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "exit_code": exit_code,
                "detail": detail,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return error_path


def _write_pro_raw_sample_csv(*, instance_ids: list[str], out_path: Path) -> None:
    wanted = {instance_id for instance_id in instance_ids if instance_id}
    by_id: dict[str, dict[str, object]] = {}
    for raw_row in [json.loads(line) for line in PROBENCH_RAW_SAMPLE_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]:
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
        missing = [column for column in PRO_REQUIRED_SAMPLE_COLUMNS if column not in row]
        if missing:
            missing_columns[instance_id] = missing
    if missing_columns:
        details = "; ".join(
            f"{instance_id}: {', '.join(columns)}"
            for instance_id, columns in list(missing_columns.items())[:5]
        )
        raise RuntimeError(f"SWE-bench Pro raw sample snapshot is missing required columns: {details}")

    fieldnames: list[str] = []
    for column in PRO_REQUIRED_SAMPLE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for instance_id in instance_ids:
        for column in by_id[instance_id].keys():
            if column not in fieldnames:
                fieldnames.append(column)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance_id in instance_ids:
            row = by_id[instance_id]
            writer.writerow({column: _coerce_pro_csv_value(row.get(column)) for column in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench Pro evaluation from the repo-local host evaluator.")
    parser.add_argument("--patch_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--dockerhub_username", required=True)
    parser.add_argument("--use_local_docker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_path = args.patch_path.resolve()
    output_dir = args.output_dir.resolve()
    work_dir = output_dir.parent

    prediction_rows = _load_prediction_rows(patch_path)
    aggregate_results: dict[str, bool] = _load_eval_results(output_dir / "eval_results.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(prediction_rows)
    errors: list[dict[str, object]] = []
    for index, row in enumerate(prediction_rows, start=1):
        instance_id = str(row.get("instance_id") or "").strip()
        instance_dir = work_dir / "instances" / _safe_component(instance_id)
        instance_output_dir = instance_dir / "evaluation_results"
        input_metadata = _prediction_metadata(row)
        if instance_id in aggregate_results:
            if _has_matching_metadata(instance_dir, input_metadata):
                print(f"[probench-wrapper] reusing existing result for {instance_id}", flush=True)
                continue
            aggregate_results.pop(instance_id, None)
        instance_results = _load_eval_results(instance_output_dir / "eval_results.json")
        if instance_id in instance_results and _has_matching_metadata(instance_dir, input_metadata):
            print(f"[probench-wrapper] reusing existing result for {instance_id}", flush=True)
            aggregate_results[instance_id] = instance_results[instance_id]
            (output_dir / "eval_results.json").write_text(
                json.dumps(aggregate_results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            continue

        instance_patch_path = instance_dir / "predictions.json"
        instance_raw_sample_path = instance_dir / "raw-sample.csv"
        _write_prediction_rows([row], instance_patch_path)
        _write_pro_raw_sample_csv(instance_ids=[instance_id], out_path=instance_raw_sample_path)

        command = [
            sys.executable,
            str(PROBENCH_SCRIPT),
            "--raw_sample_path",
            str(instance_raw_sample_path),
            "--patch_path",
            str(instance_patch_path),
            "--output_dir",
            str(instance_output_dir),
            "--scripts_dir",
            str(PROBENCH_RUN_SCRIPTS),
            "--num_workers",
            "1",
            "--dockerhub_username",
            str(args.dockerhub_username),
        ]
        if args.use_local_docker:
            command.append("--use_local_docker")

        print(f"[probench-wrapper] evaluating instance {index}/{total}: {instance_id}", flush=True)
        completed = subprocess.run(command, check=False, cwd=str(PROBENCH_ROOT))
        instance_results = _load_eval_results(instance_output_dir / "eval_results.json")
        if completed.returncode != 0:
            detail = (
                f"SWE-bench Pro evaluator exited {completed.returncode} for {instance_id}; "
                "not recording this as unresolved."
            )
            error_path = _write_instance_error(
                instance_dir=instance_dir,
                instance_id=instance_id,
                exit_code=completed.returncode,
                detail=detail,
            )
            errors.append(
                {
                    "instance_id": instance_id,
                    "exit_code": completed.returncode,
                    "error_path": str(error_path),
                    "detail": detail,
                }
            )
            print(f"[probench-wrapper] {detail}", flush=True)
        elif instance_id in instance_results:
            aggregate_results[instance_id] = instance_results[instance_id]
            _write_metadata(instance_dir, input_metadata)
        else:
            detail = (
                f"SWE-bench Pro evaluator produced no eval_results entry for {instance_id}; "
                "not recording this as unresolved."
            )
            error_path = _write_instance_error(
                instance_dir=instance_dir,
                instance_id=instance_id,
                exit_code=None,
                detail=detail,
            )
            errors.append(
                {
                    "instance_id": instance_id,
                    "exit_code": None,
                    "error_path": str(error_path),
                    "detail": detail,
                }
            )
            print(f"[probench-wrapper] {detail}", flush=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps(aggregate_results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if errors:
        (output_dir / "eval_errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    (output_dir / "eval_results.json").write_text(
        json.dumps(aggregate_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
