
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation from the repo-local host evaluator.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report_dir", default=".")
    return parser.parse_args()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned or "instance"


def _load_prediction_rows(predictions_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"SWE-bench predictions must be JSON objects: {predictions_path}")
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id:
            raise RuntimeError(f"SWE-bench prediction row is missing instance_id: {predictions_path}")
        rows.append(payload)
    return rows


def _write_prediction_rows(predictions_path: Path, rows: list[dict[str, object]]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


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


def _candidate_is_fresh(candidate: Path, previous_report_mtimes: dict[Path, int] | None) -> bool:
    if previous_report_mtimes is None:
        return True
    try:
        resolved = candidate.resolve()
        current_mtime = candidate.stat().st_mtime_ns
    except OSError:
        return False
    return previous_report_mtimes.get(resolved) != current_mtime


def _snapshot_report_mtimes(report_root: Path) -> dict[Path, int]:
    search_roots = [report_root]
    if report_root.parent != report_root:
        search_roots.append(report_root.parent)
    if report_root.parent.parent != report_root.parent:
        search_roots.append(report_root.parent.parent)

    report_mtimes: dict[Path, int] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.glob("codex*.json"):
            try:
                report_mtimes[candidate.resolve()] = candidate.stat().st_mtime_ns
            except OSError:
                continue
    if report_root.exists():
        for candidate in report_root.rglob("*.json"):
            try:
                report_mtimes[candidate.resolve()] = candidate.stat().st_mtime_ns
            except OSError:
                continue
    return report_mtimes


def _load_instance_report(
    report_root: Path,
    *,
    previous_report_mtimes: dict[Path, int] | None = None,
) -> dict[str, object]:
    search_roots = [report_root]
    if report_root.parent != report_root:
        search_roots.append(report_root.parent)
    if report_root.parent.parent != report_root.parent:
        search_roots.append(report_root.parent.parent)

    codex_reports: list[Path] = []
    for root in search_roots:
        codex_reports.extend(root.glob("codex*.json"))
    codex_reports = sorted(
        {path.resolve(): path for path in codex_reports}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate in codex_reports:
        if not _candidate_is_fresh(candidate, previous_report_mtimes):
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        report = {
            "resolved_ids": [str(item).strip() for item in (payload.get("resolved_ids") or []) if str(item).strip()],
            "unresolved_ids": [str(item).strip() for item in (payload.get("unresolved_ids") or []) if str(item).strip()],
            "error_ids": [str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()],
            "completed_ids": [str(item).strip() for item in (payload.get("completed_ids") or []) if str(item).strip()],
            "submitted_ids": [str(item).strip() for item in (payload.get("submitted_ids") or []) if str(item).strip()],
            "total_instances": int(payload.get("total_instances") or 0),
            "completed_instances": int(payload.get("completed_instances") or 0),
            "error_instances": int(payload.get("error_instances") or 0),
            "report_path": str(candidate),
        }
        if report["submitted_ids"] or report["resolved_ids"] or report["unresolved_ids"] or report["error_ids"]:
            return report

    candidates = sorted(report_root.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.name == "resolution-error.json":
            continue
        if not _candidate_is_fresh(candidate, previous_report_mtimes):
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        found = _find_resolution_report_payload(payload)
        if found is not None:
            report = dict(found)
            report["report_path"] = str(candidate)
            return report
    raise RuntimeError(f"Unable to locate a SWE-bench resolution report under {report_root}")


def _report_matches_instance(report: dict[str, object], instance_id: str) -> bool:
    wanted = str(instance_id).strip()
    if not wanted:
        return False
    submitted_ids = [str(item).strip() for item in (report.get("submitted_ids") or []) if str(item).strip()]
    resolved_ids = [str(item).strip() for item in (report.get("resolved_ids") or []) if str(item).strip()]
    unresolved_ids = [str(item).strip() for item in (report.get("unresolved_ids") or []) if str(item).strip()]
    error_ids = [str(item).strip() for item in (report.get("error_ids") or []) if str(item).strip()]
    return wanted in set(submitted_ids) | set(resolved_ids) | set(unresolved_ids) | set(error_ids)


def _load_instance_report_for_id(
    report_root: Path,
    instance_id: str,
    *,
    previous_report_mtimes: dict[Path, int] | None = None,
) -> dict[str, object]:
    search_roots = [report_root]
    if report_root.parent != report_root:
        search_roots.append(report_root.parent)
    if report_root.parent.parent != report_root.parent:
        search_roots.append(report_root.parent.parent)

    for root in search_roots:
        codex_reports = sorted(root.glob("codex*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in codex_reports:
            if not _candidate_is_fresh(candidate, previous_report_mtimes):
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            report = {
                "resolved_ids": [str(item).strip() for item in (payload.get("resolved_ids") or []) if str(item).strip()],
                "unresolved_ids": [str(item).strip() for item in (payload.get("unresolved_ids") or []) if str(item).strip()],
                "error_ids": [str(item).strip() for item in (payload.get("error_ids") or []) if str(item).strip()],
                "completed_ids": [str(item).strip() for item in (payload.get("completed_ids") or []) if str(item).strip()],
                "submitted_ids": [str(item).strip() for item in (payload.get("submitted_ids") or []) if str(item).strip()],
                "total_instances": int(payload.get("total_instances") or 0),
                "completed_instances": int(payload.get("completed_instances") or 0),
                "error_instances": int(payload.get("error_instances") or 0),
                "report_path": str(candidate),
            }
            if _report_matches_instance(report, instance_id):
                return report
    return _load_instance_report(report_root, previous_report_mtimes=previous_report_mtimes)


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        target.append(value)


def _write_aggregate_report(
    report_dir: Path,
    *,
    resolved_ids: list[str],
    unresolved_ids: list[str],
    error_ids: list[str],
) -> Path:
    payload = {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "resolved_count": len(resolved_ids),
        "total_instances": len(resolved_ids) + len(unresolved_ids) + len(error_ids),
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()

    from swebench.harness.run_evaluation import main as run_evaluation_main

    predictions_path = Path(args.predictions_path).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = _load_prediction_rows(predictions_path)
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []

    total = len(prediction_rows)
    for index, row in enumerate(prediction_rows, start=1):
        instance_id = str(row.get("instance_id") or "").strip()
        instance_slug = _safe_component(instance_id)
        instance_dir = report_dir / "instances" / instance_slug
        instance_dir.mkdir(parents=True, exist_ok=True)
        instance_predictions_path = instance_dir / "predictions.jsonl"
        _write_prediction_rows(instance_predictions_path, [row])
        instance_run_id = f"{args.run_id}--{instance_slug}"
        previous_report_mtimes = _snapshot_report_mtimes(instance_dir)
        print(f"[swebench-wrapper] evaluating instance {index}/{total}: {instance_id}", flush=True)
        run_evaluation_main(
            dataset_name=str(args.dataset_name),
            split="test",
            instance_ids=[instance_id],
            predictions_path=str(instance_predictions_path),
            max_workers=1,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=instance_run_id,
            timeout=int(args.timeout),
            namespace=None,
            rewrite_reports=False,
            modal=False,
            report_dir=str(instance_dir),
        )
        report = _load_instance_report_for_id(
            instance_dir,
            instance_id,
            previous_report_mtimes=previous_report_mtimes,
        )
        _extend_unique(
            resolved_ids,
            [value for value in (str(item).strip() for item in (report.get("resolved_ids") or [])) if value],
        )
        _extend_unique(
            unresolved_ids,
            [value for value in (str(item).strip() for item in (report.get("unresolved_ids") or [])) if value],
        )
        _extend_unique(
            error_ids,
            [value for value in (str(item).strip() for item in (report.get("error_ids") or [])) if value],
        )
        _write_aggregate_report(
            report_dir,
            resolved_ids=resolved_ids,
            unresolved_ids=unresolved_ids,
            error_ids=error_ids,
        )

    _write_aggregate_report(
        report_dir,
        resolved_ids=resolved_ids,
        unresolved_ids=unresolved_ids,
        error_ids=error_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
