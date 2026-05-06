
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import re
import shutil
from pathlib import Path


POLY_REQUIRED_SAMPLE_COLUMNS = (
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

def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned or "instance"


def _load_dataset_rows(dataset_name: str) -> tuple[list[dict[str, object]], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="test")
    column_names = [str(column) for column in getattr(dataset, "column_names", [])]
    rows = [dict(row) for row in dataset]
    return rows, column_names


def _load_jsonl_prediction_rows(predictions_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Resolution predictions contain a non-object JSONL row: {predictions_path}")
            instance_id = str(payload.get("instance_id") or "").strip()
            if not instance_id:
                raise RuntimeError(f"Resolution predictions contain JSONL rows without instance_id: {predictions_path}")
            rows.append(payload)
    return rows


def _write_prediction_rows(predictions_path: Path, rows: list[dict[str, object]]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


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


def _write_poly_dataset_csv(*, rows_by_id: dict[str, dict[str, object]], instance_ids: list[str], out_path: Path) -> None:
    missing_ids = [instance_id for instance_id in instance_ids if instance_id and instance_id not in rows_by_id]
    if missing_ids:
        raise RuntimeError(
            "SWE-PolyBench dataset is missing selected instances: "
            + ", ".join(missing_ids[:10])
            + (f" ... and {len(missing_ids) - 10} more" if len(missing_ids) > 10 else "")
        )

    missing_columns: dict[str, list[str]] = {}
    for instance_id in instance_ids:
        row = rows_by_id.get(instance_id)
        if row is None:
            continue
        missing = [column for column in POLY_REQUIRED_SAMPLE_COLUMNS if column not in row]
        if missing:
            missing_columns[instance_id] = missing
    if missing_columns:
        details = "; ".join(
            f"{instance_id}: {', '.join(columns)}"
            for instance_id, columns in list(missing_columns.items())[:5]
        )
        raise RuntimeError(f"SWE-PolyBench dataset is missing required columns: {details}")

    fieldnames: list[str] = []
    first_row = rows_by_id[instance_ids[0]]
    for column in first_row.keys():
        if column not in fieldnames:
            fieldnames.append(str(column))
    for column in POLY_REQUIRED_SAMPLE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance_id in instance_ids:
            row = rows_by_id[instance_id]
            writer.writerow({column: _coerce_poly_csv_value(column, row.get(column)) for column in fieldnames})


def _poly_image_id(*, instance_id: str, language: str) -> str:
    return f"polybench_{language.lower()}_{instance_id.lower()}"


def _poly_container_name(*, instance_id: str, language: str) -> str:
    return f"container_{_poly_image_id(instance_id=instance_id, language=language)}"


def _cleanup_stale_poly_containers(
    selected_rows: list[dict[str, object]],
    *,
    client: object | None = None,
) -> None:
    if client is None:
        import docker

        client = docker.from_env()

    container_api = getattr(client, "containers")
    for row in selected_rows:
        instance_id = str(row.get("instance_id") or "").strip()
        language = str(row.get("language") or "").strip()
        if not instance_id or not language:
            continue
        container_name = _poly_container_name(instance_id=instance_id, language=language)
        try:
            container = container_api.get(container_name)
        except Exception as exc:
            error_name = exc.__class__.__name__
            if error_name == "NotFound":
                continue
            raise
        if hasattr(container, "reload"):
            container.reload()
        status = getattr(container, "status", None)
        if status is None:
            attrs = getattr(container, "attrs", {}) or {}
            status = attrs.get("State", {}).get("Status")
        print(
            f"[polybench-preflight] removing reserved evaluator container {container_name} (status={status or 'unknown'})",
            flush=True,
        )
        container.remove(force=True)


def _load_single_result(result_path: Path) -> dict[str, object]:
    if not result_path.exists():
        raise RuntimeError(f"SWE-PolyBench did not write result.json: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"SWE-PolyBench result.json must contain an object: {result_path}")
    return payload


def _result_has_errors(payload: dict[str, object], *, instance_id: str) -> bool:
    return instance_id in {str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()}


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _error_result_payload(instance_id: str) -> dict[str, object]:
    return {
        "resolved": [],
        "not_resolved": [],
        "error_ids": [instance_id],
        "total_empty_patch_instances": 0,
        "generation": [],
        "no_generation": [],
        "patch_applied": [],
        "with_logs": [],
    }


def _write_aggregate_result(results: list[dict[str, object]], out_path: Path) -> None:
    aggregate_payload = _aggregate_poly_results(results)
    out_path.write_text(
        json.dumps(aggregate_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_diagnostic_files(*, source_dir: Path, work_dir: Path) -> None:
    for directory_name in ("build_logs", "run_logs_python", "run_logs_javascript", "run_logs_typescript", "run_logs_java"):
        source = source_dir / directory_name
        if not source.exists():
            continue
        target = work_dir / directory_name
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        target.append(value)


def _aggregate_poly_results(results: list[dict[str, object]]) -> dict[str, object]:
    resolved: list[str] = []
    not_resolved: list[str] = []
    generation: list[str] = []
    no_generation: list[str] = []
    patch_applied: list[str] = []
    with_logs: list[str] = []
    error_ids: list[str] = []
    total_empty_patch_instances = 0

    for payload in results:
        resolved.extend([str(item).strip() for item in (payload.get("resolved") or []) if str(item).strip()])
        not_resolved.extend([str(item).strip() for item in (payload.get("not_resolved") or []) if str(item).strip()])
        generation.extend([str(item).strip() for item in (payload.get("generation") or []) if str(item).strip()])
        no_generation.extend([str(item).strip() for item in (payload.get("no_generation") or []) if str(item).strip()])
        patch_applied.extend([str(item).strip() for item in (payload.get("patch_applied") or []) if str(item).strip()])
        with_logs.extend([str(item).strip() for item in (payload.get("with_logs") or []) if str(item).strip()])
        error_ids.extend([str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()])
        total_empty_patch_instances += int(payload.get("total_empty_patch_instances") or 0)

    return {
        "resolved": resolved,
        "not_resolved": not_resolved,
        "error_ids": error_ids,
        "total_instances": len(resolved) + len(not_resolved) + len(error_ids),
        "total_resolved": len(resolved),
        "total_unresolved": len(not_resolved),
        "total_empty_patch_instances": total_empty_patch_instances,
        "generation": generation,
        "no_generation": no_generation,
        "patch_applied": patch_applied,
        "with_logs": with_logs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-PolyBench evaluation from the repo-local host evaluator.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import poly_bench_evaluation.run_evaluation as run_evaluation

    predictions_path = args.predictions_path.resolve()
    result_path = args.result_path.resolve()
    work_dir = result_path.parent
    dataset_subset_path = work_dir / "dataset-subset.csv"

    prediction_rows = _load_jsonl_prediction_rows(predictions_path)
    instance_ids = [str(row.get("instance_id") or "").strip() for row in prediction_rows]
    dataset_rows, _column_names = _load_dataset_rows(str(args.dataset_name))
    rows_by_id: dict[str, dict[str, object]] = {}
    for row in dataset_rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id and instance_id not in rows_by_id:
            rows_by_id[instance_id] = row
    _write_poly_dataset_csv(rows_by_id=rows_by_id, instance_ids=instance_ids, out_path=dataset_subset_path)
    selected_rows = [rows_by_id[instance_id] for instance_id in instance_ids]
    _cleanup_stale_poly_containers(selected_rows)

    results: list[dict[str, object]] = []
    total = len(prediction_rows)
    for index, row in enumerate(prediction_rows, start=1):
        instance_id = str(row.get("instance_id") or "").strip()
        instance_slug = _safe_component(instance_id)
        instance_dir = work_dir / "instances" / instance_slug
        instance_dir.mkdir(parents=True, exist_ok=True)

        single_predictions_path = instance_dir / "predictions.jsonl"
        single_dataset_path = instance_dir / "dataset-subset.csv"
        _write_prediction_rows(single_predictions_path, [row])
        _write_poly_dataset_csv(rows_by_id=rows_by_id, instance_ids=[instance_id], out_path=single_dataset_path)
        _cleanup_stale_poly_containers([rows_by_id[instance_id]])

        existing_result_path = instance_dir / "result.json"
        if existing_result_path.exists():
            existing_payload = _load_single_result(existing_result_path)
            if _result_has_errors(existing_payload, instance_id=instance_id):
                print(f"[polybench-wrapper] retrying previous error result for {instance_id}", flush=True)
            else:
                print(f"[polybench-wrapper] replacing existing result for {instance_id}", flush=True)
            existing_result_path.unlink()

        print(f"[polybench-wrapper] evaluating instance {index}/{total}: {instance_id}", flush=True)
        try:
            with _temporary_cwd(instance_dir):
                run_evaluation.evaluate_predictions(
                    dataset_path=str(single_dataset_path),
                    predictions_path=str(single_predictions_path),
                    result_path=str(instance_dir / "evaluation_results"),
                    num_threads=1,
                    evaluate_gold=False,
                    repo_path=str(work_dir / "repos"),
                    delete_image=False,
                    skip_existing=True,
                    retrieval_metrics_only=False,
                    node_retrieval_metrics=False,
                )
        except Exception as exc:
            _copy_diagnostic_files(source_dir=instance_dir, work_dir=work_dir)
            error_payload = _error_result_payload(instance_id)
            (instance_dir / "result.json").write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results.append(error_payload)
            _write_aggregate_result(results, work_dir / "result.json")
            print(
                f"[polybench-wrapper] evaluator raised for {instance_id}: {exc}; recording error and continuing",
                flush=True,
            )
            continue
        _copy_diagnostic_files(source_dir=instance_dir, work_dir=work_dir)
        results.append(_load_single_result(instance_dir / "result.json"))
        _write_aggregate_result(results, work_dir / "result.json")

    _write_aggregate_result(results, work_dir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
