
"""Task-loading helpers for coding-agent integrations."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from .constants import BENCH_LABEL_PREFIXES, DEFAULT_PROMPT_FIELDS
from .types import LoadedTask


BENCH_ALIASES = {
    "verified": "Verified",
    "swe-bench-verified": "Verified",
    "swebench-verified": "Verified",
    "pro": "Pro",
    "swe-bench-pro": "Pro",
    "swebench-pro": "Pro",
    "poly": "Poly",
    "swe-polybench": "Poly",
    "polybench": "Poly",
    "multi": "Multi",
    "multi-swe-bench": "Multi",
    "multiswebench": "Multi",
}
SUPPORTED_BENCHES = frozenset({"Verified", "Pro", "Poly", "Multi"})


def _normalize_bench_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in SUPPORTED_BENCHES:
        return text
    normalized = text.lower().replace("_", "-").replace(" ", "-")
    return BENCH_ALIASES.get(normalized, "")


def detect_bench_from_instance_id(instance_id: str, original_inst_id: str = "") -> str:
    """Infer benchmark family from ContextBench instance identifiers."""
    value = (instance_id or original_inst_id or "").strip()
    if not value:
        return "Verified"
    for prefix, bench in BENCH_LABEL_PREFIXES:
        if value.startswith(prefix):
            return bench
    if value.startswith("instance_") and len(value) > 50:
        return "Pro"
    if "polybench" in value.lower():
        return "Poly"
    if "multi" in value.lower():
        return "Multi"
    return "Verified"


def parse_bench_filter(value: str | None) -> list[str] | None:
    if not value:
        return None
    benches = []
    for raw in value.split(","):
        if not raw.strip():
            continue
        benches.append(_normalize_bench_label(raw) or raw.strip())
    return benches or None


def parse_instance_filter(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _first_defined(row: dict[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _load_subset_order(subset_csv: Path | None) -> list[str] | None:
    if not subset_csv:
        return None
    if not subset_csv.exists():
        raise FileNotFoundError(f"Configured subset CSV not found: {subset_csv}")
    ordered_ids: list[str] = []
    with open(subset_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in ("instance_id", "original_inst_id"):
                value = (row.get(key) or "").strip()
                if value and value not in ordered_ids:
                    ordered_ids.append(value)
    return ordered_ids or None


def _normalize_task_row(row: dict[str, object]) -> LoadedTask:
    instance_id = str(row.get("instance_id") or row.get("inst_id") or "").strip()
    original_inst_id = str(row.get("original_inst_id") or "").strip()
    prompt = _first_defined(row, DEFAULT_PROMPT_FIELDS)
    explicit_bench = _normalize_bench_label(row.get("bench") or row.get("source"))
    raw_explicit_bench = str(row.get("bench") or row.get("source") or "").strip()
    if raw_explicit_bench and not explicit_bench:
        raise ValueError(f"Unsupported benchmark label {raw_explicit_bench!r} for task {instance_id or original_inst_id}")
    inferred_bench = detect_bench_from_instance_id(instance_id, original_inst_id)
    bench = explicit_bench or inferred_bench
    if explicit_bench and inferred_bench and inferred_bench != "Verified" and explicit_bench != inferred_bench:
        raise ValueError(
            "Benchmark label conflicts with instance_id prefix for "
            f"{instance_id or original_inst_id}: explicit={explicit_bench}, inferred={inferred_bench}"
        )
    commit = str(row.get("base_commit") or row.get("commit") or "").strip()
    return {
        "bench": bench,
        "instance_id": instance_id,
        "original_inst_id": original_inst_id,
        "repo": str(row.get("repo") or "").strip(),
        "repo_url": str(row.get("repo_url") or "").strip(),
        "commit": commit,
        "prompt": str(prompt or "").strip(),
        "language": str(row.get("language") or "").strip(),
        "raw": row,
    }


def _load_rows_from_json(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON task data in {path}")


def _load_rows_from_parquet(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to load ContextBench parquet task files. Install dependencies from requirements.txt."
        ) from exc

    dataset = ds.dataset(str(path), format="parquet")
    available_columns = set(dataset.schema.names)
    desired_columns = [
        "instance_id",
        "inst_id",
        "original_inst_id",
        "repo",
        "repo_url",
        "base_commit",
        "commit",
        "bench",
        "source",
        "language",
        "problem_statement",
        "prompt",
        "question",
        "task",
        "instruction",
    ]
    columns = [column for column in desired_columns if column in available_columns]
    return dataset.to_table(columns=columns).to_pylist()


def count_task_rows(task_path: Path) -> int:
    """Return the unfiltered number of task rows in a supported task data file."""
    if not task_path.exists():
        raise FileNotFoundError(f"Task data not found: {task_path}")

    if task_path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pyarrow is required to inspect ContextBench parquet task files. Install dependencies from requirements.txt."
            ) from exc
        return int(pq.ParquetFile(str(task_path)).metadata.num_rows)

    if task_path.suffix == ".jsonl":
        with open(task_path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    with open(task_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return 1
    raise ValueError(f"Unsupported JSON task data in {task_path}")


def load_tasks(
    task_path: Path,
    *,
    subset_csv: Path | None = None,
    bench_filter: Sequence[str] | None = None,
    instance_filter: Sequence[str] | None = None,
    limit: int = 0,
) -> list[LoadedTask]:
    """Load tasks with prompts from parquet/json/jsonl and optional subset ordering."""
    if not task_path.exists():
        raise FileNotFoundError(f"Task data not found: {task_path}")

    if task_path.suffix == ".parquet":
        rows = _load_rows_from_parquet(task_path)
    else:
        rows = _load_rows_from_json(task_path)

    normalized = [_normalize_task_row(row) for row in rows]

    ordered_subset = _load_subset_order(subset_csv)
    if ordered_subset:
        by_id: dict[str, LoadedTask] = {}
        for task in normalized:
            if task["instance_id"]:
                by_id[task["instance_id"]] = task
            if task["original_inst_id"]:
                by_id[task["original_inst_id"]] = task
        ordered_tasks = []
        seen = set()
        missing_subset_ids: list[str] = []
        for task_id in ordered_subset:
            task = by_id.get(task_id)
            if not task:
                missing_subset_ids.append(task_id)
                continue
            key = task["instance_id"] or task["original_inst_id"]
            if key in seen:
                continue
            ordered_tasks.append(task)
            seen.add(key)
        if missing_subset_ids:
            preview = ", ".join(missing_subset_ids[:10])
            suffix = f" ... and {len(missing_subset_ids) - 10} more" if len(missing_subset_ids) > 10 else ""
            raise ValueError(f"Configured subset contains IDs not found in task data: {preview}{suffix}")
        normalized = ordered_tasks

    if bench_filter:
        normalized = [task for task in normalized if task["bench"] in set(bench_filter)]
    if instance_filter:
        allowed = set(instance_filter)
        normalized = [
            task
            for task in normalized
            if task["instance_id"] in allowed or task["original_inst_id"] in allowed
        ]
    if limit > 0:
        normalized = normalized[:limit]
    return normalized
