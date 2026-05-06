# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: resolve task-result record paths robustly, tighten provenance source attribution, and normalize file coverage from spans and symbols.

"""Conversion helpers from coding-agent records to ContextBench trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from ..agents.registry import get_coding_agent_adapter, normalize_coding_agent_name
from ..parsers.trajectory import effective_file_list
from .files import read_json, read_jsonl
from .records import merge_span_maps, normalize_retrieval_steps, normalize_span_map, normalize_symbol_map
from .trace_inference import merge_retrieval_steps
from .types import SpanProvenanceMap, SymbolMap, SymbolProvenanceMap, TrajectoryData

_ARTIFACT_PATH_ERRORS_FIELD = "_artifact_path_errors"


def _empty_conversion_summary() -> dict[str, object]:
    return {
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


def _finalize_conversion_summary(summary: dict[str, object]) -> dict[str, object]:
    task_count = summary.get("selected_task_count")
    if isinstance(task_count, int):
        prediction_count = int(summary.get("prediction_count") or 0)
        summary["coverage_of_attempted_tasks"] = (prediction_count / task_count) if task_count else 0.0
        summary["missing_prediction_count"] = max(task_count - prediction_count, 0)
        summary["is_partial"] = bool(task_count and prediction_count < task_count)
    else:
        summary["coverage_of_attempted_tasks"] = None
        summary["missing_prediction_count"] = None
        summary["is_partial"] = False
    return summary


def record_is_convertible(record: dict[str, object], expected_agent: str | None = None) -> bool:
    if not isinstance(record, dict):
        return False
    raw_agent = str(record.get("agent") or "").strip().lower()
    agent = normalize_coding_agent_name(raw_agent) or raw_agent
    if expected_agent:
        normalized_expected = normalize_coding_agent_name(expected_agent) or str(expected_agent).strip().lower()
        if agent and agent != normalized_expected:
            return False
    final_output = record.get("final_output")
    return isinstance(final_output, dict)


def _parser_for_agent(agent: str):
    try:
        return get_coding_agent_adapter(agent).create_parser()
    except ValueError:
        return None


def _find_task_results_path(source: Path) -> Path | None:
    """Locate the nearest task-results manifest for a run-suite variant tree."""
    candidates = [
        source / "task-results.jsonl",
        source.parent / "task-results.jsonl",
        source.parent.parent / "task-results.jsonl",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def resolve_record_path(
    record_path_value: object,
    *,
    task_results_path: Path | None = None,
    source_dir: Path | None = None,
) -> Path | None:
    raw_path = str(record_path_value or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    bases: list[Path] = []
    if task_results_path is not None:
        bases.append(task_results_path.parent)
    if source_dir is not None:
        bases.extend([source_dir, source_dir.parent])

    seen: set[Path] = set()
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    if candidate.exists():
        return candidate
    return None


def _resolve_record_sidecar_path(path_value: object, *, record_path: Path) -> str | None:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    sidecar_candidates: list[Path] = []
    if candidate.is_absolute():
        sidecar_candidates.append(record_path.parent / candidate.name)
    else:
        sidecar_candidates.append(record_path.parent / candidate)
        sidecar_candidates.append(record_path.parent / candidate.name)

    seen: set[Path] = set()
    for sidecar in sidecar_candidates:
        resolved = sidecar.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)
    return None


def record_with_resolved_artifact_paths(
    record: dict[str, object],
    *,
    record_path: Path,
    require_existing_artifacts: bool = False,
) -> dict[str, object]:
    """Make sidecar artifact paths portable across host and postprocess container mounts."""
    resolved_record = dict(record)
    path_errors: list[dict[str, str]] = []
    for field in ("raw_response_path", "diff_path"):
        original_value = resolved_record.get(field)
        resolved_path = _resolve_record_sidecar_path(original_value, record_path=record_path)
        if resolved_path is not None:
            resolved_record[field] = resolved_path
        elif require_existing_artifacts and str(original_value or "").strip():
            path_errors.append({"field": field, "path": str(original_value)})
    if path_errors:
        resolved_record[_ARTIFACT_PATH_ERRORS_FIELD] = path_errors
    return resolved_record


def records_with_resolved_aggregate_artifact_paths(
    records: Iterable[object],
    *,
    aggregate_path: Path,
) -> list[object]:
    resolved: list[object] = []
    for record in records:
        if isinstance(record, dict):
            resolved.append(
                record_with_resolved_artifact_paths(
                    record,
                    record_path=aggregate_path,
                    require_existing_artifacts=True,
                )
            )
        else:
            resolved.append(record)
    return resolved


def _merge_source_lists(*values: list[str]) -> list[str]:
    merged: list[str] = []
    for items in values:
        for item in items:
            if item not in merged:
                merged.append(item)
    return merged


class ContextPathValidationError(ValueError):
    def __init__(self, *, instance_id: str, invalid_paths: list[str]) -> None:
        self.instance_id = instance_id
        self.invalid_paths = sorted({path for path in invalid_paths if path})
        detail = ", ".join(self.invalid_paths[:5])
        if len(self.invalid_paths) > 5:
            detail += f" ... and {len(self.invalid_paths) - 5} more"
        super().__init__(
            f"Invalid predicted context paths for {instance_id or '<unknown>'}: {detail}"
        )


class ContextIdentityValidationError(ValueError):
    def __init__(self, *, record_instance_id: str, record_original_inst_id: str, reported_task_id: str) -> None:
        self.record_instance_id = record_instance_id
        self.record_original_inst_id = record_original_inst_id
        self.reported_task_id = reported_task_id
        super().__init__(
            "Agent-reported task_id does not match the runner record identity: "
            f"task_id={reported_task_id!r}, instance_id={record_instance_id!r}, "
            f"original_inst_id={record_original_inst_id!r}"
        )


def _workspace_path_for_record(record: dict[str, object]) -> Path | None:
    raw_value = str(record.get("workspace_path") or "").strip()
    return Path(raw_value) if raw_value else None


def _clean_relative_context_path(value: str) -> str:
    normalized = str(value).strip().strip("'\"").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        return ""
    posix_path = PurePosixPath(normalized)
    if posix_path.is_absolute() or normalized in {".", ".."}:
        return ""
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        return ""
    return posix_path.as_posix()


def _normalize_context_path(
    value: object,
    *,
    workspace_path: Path | None,
    invalid_paths: list[str],
    strict: bool,
) -> str:
    raw_value = str(value or "").strip().strip("'\"")
    if not raw_value:
        return ""
    candidate = Path(raw_value)
    if candidate.is_absolute():
        if workspace_path is None:
            if strict:
                invalid_paths.append(raw_value)
            return ""
        try:
            relative = candidate.resolve(strict=False).relative_to(workspace_path.resolve(strict=False))
        except Exception:
            if strict:
                invalid_paths.append(raw_value)
            return ""
        cleaned = _clean_relative_context_path(relative.as_posix())
        if cleaned:
            return cleaned
        if strict:
            invalid_paths.append(raw_value)
        return ""
    cleaned = _clean_relative_context_path(raw_value)
    if cleaned:
        return cleaned
    if strict:
        invalid_paths.append(raw_value)
    return ""


def _normalize_context_file_list(
    values: object,
    *,
    workspace_path: Path | None,
    invalid_paths: list[str],
    strict: bool,
) -> list[str]:
    if not isinstance(values, list):
        return []
    files = {
        normalized
        for item in values
        if (normalized := _normalize_context_path(item, workspace_path=workspace_path, invalid_paths=invalid_paths, strict=strict))
    }
    return sorted(files)


def _normalize_context_span_map(
    value: object,
    *,
    workspace_path: Path | None,
    invalid_paths: list[str],
    strict: bool,
) -> dict[str, list[dict[str, int]]]:
    normalized = normalize_span_map(value)
    result: dict[str, list[dict[str, int]]] = {}
    for file_path, spans in normalized.items():
        cleaned = _normalize_context_path(file_path, workspace_path=workspace_path, invalid_paths=invalid_paths, strict=strict)
        if cleaned:
            result.setdefault(cleaned, []).extend(spans)
    return result


def _normalize_context_symbol_map(
    value: object,
    *,
    workspace_path: Path | None,
    invalid_paths: list[str],
    strict: bool,
) -> SymbolMap:
    normalized = normalize_symbol_map(value)
    result: SymbolMap = {}
    for file_path, names in normalized.items():
        cleaned = _normalize_context_path(file_path, workspace_path=workspace_path, invalid_paths=invalid_paths, strict=strict)
        if cleaned:
            result.setdefault(cleaned, []).extend(names)
    return {file_path: sorted(set(names)) for file_path, names in result.items() if names}


def _normalize_context_steps(
    steps: object,
    *,
    workspace_path: Path | None,
    invalid_paths: list[str],
    strict: bool,
) -> list[dict[str, object]]:
    normalized_steps = normalize_retrieval_steps(steps)
    result: list[dict[str, object]] = []
    for step in normalized_steps:
        files = _normalize_context_file_list(
            step.get("files", []),
            workspace_path=workspace_path,
            invalid_paths=invalid_paths,
            strict=strict,
        )
        spans = _normalize_context_span_map(
            step.get("spans", {}),
            workspace_path=workspace_path,
            invalid_paths=invalid_paths,
            strict=strict,
        )
        symbols = _normalize_context_symbol_map(
            step.get("symbols", {}),
            workspace_path=workspace_path,
            invalid_paths=invalid_paths,
            strict=strict,
        )
        effective_files = sorted(effective_file_list(files=files, spans=spans, symbols=symbols))
        if effective_files or spans or symbols:
            result.append({"files": effective_files, "spans": spans, "symbols": symbols})
    return result


def convert_records_with_summary(
    records: Iterable[dict[str, object]],
    *,
    expected_agent: str | None = None,
    selected_task_count: int | None = None,
    missing_record_path_count: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records_list = list(records)
    summary = _empty_conversion_summary()
    summary["selected_task_count"] = selected_task_count if selected_task_count is not None else len(records_list)
    summary["record_count"] = len(records_list)
    summary["missing_record_path_count"] = missing_record_path_count

    converted: list[dict[str, object]] = []
    for record in records_list:
        if not record_is_convertible(record, expected_agent=expected_agent):
            summary["nonconvertible_record_count"] = int(summary["nonconvertible_record_count"] or 0) + 1
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
                        "instance_id": str(record.get("instance_id") or ""),
                        "error": "missing_artifact_path",
                        "artifact_paths": artifact_path_errors,
                    }
                )
            continue
        try:
            converted.append(convert_run_record(record))
        except ContextPathValidationError as exc:
            summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
            summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
            errors = summary.setdefault("conversion_errors", [])
            if isinstance(errors, list):
                errors.append(
                    {
                        "instance_id": exc.instance_id,
                        "error": "invalid_predicted_context_path",
                        "invalid_paths": exc.invalid_paths,
                    }
                )
        except ContextIdentityValidationError as exc:
            summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
            summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
            errors = summary.setdefault("conversion_errors", [])
            if isinstance(errors, list):
                errors.append(
                    {
                        "instance_id": exc.record_instance_id or exc.record_original_inst_id,
                        "error": "task_id_identity_mismatch",
                        "reported_task_id": exc.reported_task_id,
                        "record_instance_id": exc.record_instance_id,
                        "record_original_inst_id": exc.record_original_inst_id,
                    }
                )
        except Exception as exc:
            summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
            summary["conversion_error_count"] = int(summary["conversion_error_count"] or 0) + 1
            errors = summary.setdefault("conversion_errors", [])
            if isinstance(errors, list):
                errors.append({"instance_id": str(record.get("instance_id") or ""), "error": str(exc)})

    summary["prediction_count"] = len(converted)
    return converted, _finalize_conversion_summary(summary)


def _provenance_precedence(source: str) -> int:
    order = {
        "agent_report": 3,
        "trace_inference": 2,
    }
    return order.get(source, 0)


def _pick_preferred_source(*sources: str) -> str:
    best = ""
    best_score = -1
    for source in sources:
        score = _provenance_precedence(source)
        if score > best_score:
            best = source
            best_score = score
    return best


def _spans_to_provenance(spans: dict[str, list[dict[str, int]]], source: str) -> SpanProvenanceMap:
    return {
        file_path: [
            {"start": span["start"], "end": span["end"], "source": source}
            for span in file_spans
        ]
        for file_path, file_spans in spans.items()
    }


def _symbols_to_provenance(symbols: SymbolMap, source: str) -> SymbolProvenanceMap:
    return {
        file_path: [{"name": name, "source": source} for name in names]
        for file_path, names in symbols.items()
    }


def _merge_span_provenance(*maps: SpanProvenanceMap) -> SpanProvenanceMap:
    merged: SpanProvenanceMap = {}
    for mapping in maps:
        for file_path, entries in mapping.items():
            bucket = {
                (entry["start"], entry["end"]): dict(entry)
                for entry in merged.get(file_path, [])
            }
            for entry in entries:
                key = (entry["start"], entry["end"])
                if key in bucket:
                    bucket[key]["source"] = _pick_preferred_source(bucket[key]["source"], entry["source"])
                else:
                    bucket[key] = dict(entry)
            merged[file_path] = [bucket[key] for key in sorted(bucket)]
    return merged


def _merge_symbol_provenance(*maps: SymbolProvenanceMap) -> SymbolProvenanceMap:
    merged: SymbolProvenanceMap = {}
    for mapping in maps:
        for file_path, entries in mapping.items():
            bucket = {
                entry["name"]: dict(entry)
                for entry in merged.get(file_path, [])
            }
            for entry in entries:
                key = entry["name"]
                if key in bucket:
                    bucket[key]["source"] = _pick_preferred_source(bucket[key]["source"], entry["source"])
                else:
                    bucket[key] = dict(entry)
            merged[file_path] = [bucket[key] for key in sorted(bucket)]
    return merged


def convert_run_record(record: dict[str, object], parser=None) -> dict[str, object]:
    final_output = record.get("final_output") or {}
    record_instance_id = str(record.get("instance_id") or "").strip()
    record_original_inst_id = str(record.get("original_inst_id") or "").strip()
    reported_task_id = str(final_output.get("task_id") or "").strip() if isinstance(final_output, dict) else ""
    valid_record_ids = {value for value in (record_instance_id, record_original_inst_id) if value}
    if reported_task_id and valid_record_ids and reported_task_id not in valid_record_ids:
        raise ContextIdentityValidationError(
            record_instance_id=record_instance_id,
            record_original_inst_id=record_original_inst_id,
            reported_task_id=reported_task_id,
        )
    task_id = record_instance_id or record_original_inst_id or reported_task_id
    parser = parser or _parser_for_agent(str(record.get("agent") or ""))
    workspace_path = _workspace_path_for_record(record)
    invalid_context_paths: list[str] = []
    raw_response = None
    inferred_traj: TrajectoryData | None = None
    if parser is not None and hasattr(parser, "load_raw_response"):
        raw_response = parser.load_raw_response(record)
        if raw_response is not None:
            inferred_traj = parser.infer_trajectory_data(raw_response, record=record)

    retrieval_steps = _normalize_context_steps(
        final_output.get("retrieval_steps"),
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=True,
    )
    reported_retrieval_steps = list(retrieval_steps)
    retrieved_context_files = _normalize_context_file_list(
        final_output.get("retrieved_context_files") or [],
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=True,
    )
    retrieved_context_spans = _normalize_context_span_map(
        final_output.get("retrieved_context_spans"),
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=True,
    )
    retrieved_context_symbols = _normalize_context_symbol_map(
        final_output.get("retrieved_context_symbols"),
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=True,
    )

    model_patch = str(record.get("model_patch") or "").strip()

    merged_step_spans = merge_span_maps(*(step.get("spans") for step in retrieval_steps))
    merged_step_symbols: SymbolMap = {}
    for step in retrieval_steps:
        for file_path, names in normalize_symbol_map(step.get("symbols")).items():
            merged_step_symbols.setdefault(file_path, []).extend(names)
    merged_step_symbols = {
        file_path: sorted(set(names))
        for file_path, names in merged_step_symbols.items()
        if names
    }

    inferred_steps = _normalize_context_steps(
        inferred_traj.get("pred_steps", []) if inferred_traj else [],
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=False,
    )
    inferred_files = _normalize_context_file_list(
        inferred_traj.get("pred_files", []) if inferred_traj else [],
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=False,
    )
    inferred_spans = _normalize_context_span_map(
        inferred_traj.get("pred_spans", {}) if inferred_traj else {},
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=False,
    )
    inferred_symbols = _normalize_context_symbol_map(
        inferred_traj.get("pred_symbols", {}) if inferred_traj else {},
        workspace_path=workspace_path,
        invalid_paths=invalid_context_paths,
        strict=False,
    )
    inferred_meta = inferred_traj.get("trace_inference_meta", {}) if inferred_traj else {}

    if invalid_context_paths:
        raise ContextPathValidationError(instance_id=str(task_id), invalid_paths=invalid_context_paths)

    pred_steps = merge_retrieval_steps(inferred_steps, retrieval_steps)
    pred_steps = [
        {
            "files": sorted(
                effective_file_list(
                    files=step.get("files", []),
                    spans=step.get("spans", {}),
                    symbols=step.get("symbols", {}),
                )
            ),
            "spans": step.get("spans", {}),
            "symbols": step.get("symbols", {}),
        }
        for step in pred_steps
    ]
    pred_files = sorted(
        set(
            inferred_files
            or []
        )
        | set(retrieved_context_files)
        | {file for step in pred_steps for file in step.get("files", [])}
    )
    pred_spans = merge_span_maps(inferred_spans, retrieved_context_spans, merged_step_spans)
    pred_spans = {
        file_path: [
            span
            for _, span in sorted(
                {
                    (span["start"], span["end"]): span
                    for span in spans
                }.items()
            )
        ]
        for file_path, spans in pred_spans.items()
        if spans
    }
    pred_symbols: SymbolMap = {}
    for mapping in (inferred_symbols, retrieved_context_symbols, merged_step_symbols):
        for file_path, names in mapping.items():
            pred_symbols.setdefault(file_path, []).extend(names)
    pred_symbols = {file_path: sorted(set(names)) for file_path, names in pred_symbols.items() if names}
    pred_files = sorted(effective_file_list(files=pred_files, spans=pred_spans, symbols=pred_symbols))

    inferred_context_files = set(inferred_files) | set(inferred_spans) | set(inferred_symbols)
    agent_report_files = set(retrieved_context_files) | set(retrieved_context_spans) | set(retrieved_context_symbols)
    for step in reported_retrieval_steps:
        agent_report_files.update(
            effective_file_list(
                files=step.get("files", []),
                spans=step.get("spans", {}),
                symbols=step.get("symbols", {}),
            )
        )

    pred_files_provenance: dict[str, str] = {}
    all_provenance_files = sorted(set(pred_files))
    for file_path in all_provenance_files:
        sources: list[str] = []
        if file_path in inferred_context_files:
            sources.append("trace_inference")
        if file_path in agent_report_files:
            sources.append("agent_report")
        pred_files_provenance[file_path] = _pick_preferred_source(*sources) if sources else "trace_inference"

    pred_spans_provenance = _merge_span_provenance(
        _spans_to_provenance(inferred_spans, "trace_inference"),
        _spans_to_provenance(retrieved_context_spans, "agent_report"),
        _spans_to_provenance(merged_step_spans, "agent_report"),
    )
    pred_symbols_provenance = _merge_symbol_provenance(
        _symbols_to_provenance(inferred_symbols, "trace_inference"),
        _symbols_to_provenance(retrieved_context_symbols, "agent_report"),
        _symbols_to_provenance(merged_step_symbols, "agent_report"),
    )

    has_reported_file_context = bool(agent_report_files)

    pred_files_source = _merge_source_lists(
        ["trace_inference"] if inferred_context_files else [],
        ["agent_report"] if has_reported_file_context else [],
    )
    pred_spans_source = _merge_source_lists(
        ["trace_inference"] if inferred_spans else [],
        ["agent_report"] if retrieved_context_spans or merged_step_spans else [],
    )
    pred_symbols_source = _merge_source_lists(
        ["trace_inference"] if inferred_symbols else [],
        ["agent_report"] if retrieved_context_symbols or merged_step_symbols else [],
    )

    traj_data: TrajectoryData = {
        "pred_steps": pred_steps,
        "pred_files": pred_files,
        "pred_spans": pred_spans,
        "pred_symbols": pred_symbols,
        "pred_files_provenance": pred_files_provenance,
        "pred_spans_provenance": pred_spans_provenance,
        "pred_symbols_provenance": pred_symbols_provenance,
        "pred_files_source": pred_files_source,
        "pred_spans_source": pred_spans_source,
        "pred_symbols_source": pred_symbols_source,
    }
    if inferred_meta:
        traj_data["trace_inference_meta"] = inferred_meta

    return {
        "instance_id": task_id,
        "original_inst_id": record.get("original_inst_id") or None,
        "repo_url": record.get("repo_url") or None,
        "commit": record.get("commit") or None,
        "model_patch": model_patch,
        "traj_data": traj_data,
    }


def convert_records(records: Iterable[dict[str, object]], expected_agent: str | None = None) -> list[dict[str, object]]:
    converted, _ = convert_records_with_summary(records, expected_agent=expected_agent)
    return converted


def load_predictions_with_summary_from_path(
    path: str | Path,
    expected_agent: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {source}")

    if source.is_dir():
        task_results = _find_task_results_path(source)
        if task_results is not None:
            rows = [row for row in read_jsonl(task_results) if isinstance(row, dict)]
            records: list[dict[str, object]] = []
            missing_record_path_count = 0
            for row in read_jsonl(task_results):
                if not isinstance(row, dict):
                    continue
                record_path_value = row.get("record_path")
                record_path = resolve_record_path(
                    record_path_value,
                    task_results_path=task_results,
                    source_dir=source,
                )
                if record_path is None:
                    missing_record_path_count += 1
                    continue
                records.append(
                    record_with_resolved_artifact_paths(
                        read_json(record_path),
                        record_path=record_path,
                        require_existing_artifacts=True,
                    )
                )
            predictions, summary = convert_records_with_summary(
                records,
                expected_agent=expected_agent,
                selected_task_count=len(rows),
                missing_record_path_count=missing_record_path_count,
            )
            return predictions, summary
        aggregate = source / "records.jsonl"
        if aggregate.exists():
            return convert_records_with_summary(
                records_with_resolved_aggregate_artifact_paths(read_jsonl(aggregate), aggregate_path=aggregate),
                expected_agent=expected_agent,
            )
        records: list[dict[str, object]] = []
        suffixes = ("*.codex-record.json", "*.claude-record.json")
        for pattern in suffixes:
            for record_path in sorted(source.rglob(pattern)):
                records.append(record_with_resolved_artifact_paths(read_json(record_path), record_path=record_path))
        return convert_records_with_summary(records, expected_agent=expected_agent)

    if source.suffix == ".jsonl":
        return convert_records_with_summary(
            records_with_resolved_aggregate_artifact_paths(read_jsonl(source), aggregate_path=source),
            expected_agent=expected_agent,
        )

    loaded = read_json(source)
    if isinstance(loaded, dict):
        loaded_rows = [record_with_resolved_artifact_paths(loaded, record_path=source)]
    elif isinstance(loaded, list):
        loaded_rows = records_with_resolved_aggregate_artifact_paths(loaded, aggregate_path=source)
    else:
        raise ValueError(f"Unsupported record payload in {source}")
    return convert_records_with_summary(loaded_rows, expected_agent=expected_agent)


def load_predictions_from_path(path: str | Path, expected_agent: str | None = None) -> list[dict[str, object]]:
    predictions, _ = load_predictions_with_summary_from_path(path, expected_agent=expected_agent)
    return predictions
