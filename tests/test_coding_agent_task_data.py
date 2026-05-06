
from __future__ import annotations

import json

import pytest

from contextbench.coding_agents.task_data import (
    detect_bench_from_instance_id,
    load_tasks,
    parse_bench_filter,
    parse_instance_filter,
)


@pytest.mark.parametrize(
    ("instance_id", "expected_bench"),
    [
        ("psf__requests-1142", "Verified"),
        ("SWE-Bench-Pro__numpy__numpy-1", "Pro"),
        ("SWE-PolyBench__repo-1", "Poly"),
        ("example__multi-repo-12", "Multi"),
        ("", "Verified"),
    ],
)
def test_detect_bench_from_instance_id_variants(instance_id: str, expected_bench: str) -> None:
    assert detect_bench_from_instance_id(instance_id) == expected_bench


def test_parse_filter_helpers_normalize_values() -> None:
    assert parse_bench_filter("verified, poly,Custom") == ["Verified", "Poly", "Custom"]
    assert parse_bench_filter("") is None
    assert parse_instance_filter(" a , ,b ") == ["a", "b"]
    assert parse_instance_filter(None) is None


def test_load_tasks_from_jsonl_uses_prompt_aliases_and_normalizes_rows(tmp_path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "psf__requests-1142",
                        "source": "Verified",
                        "repo_url": "https://github.com/psf/requests.git",
                        "base_commit": "abc123",
                        "problem_statement": "Fix request handling.",
                        "language": "python",
                    }
                ),
                json.dumps(
                    {
                        "inst_id": "SWE-PolyBench__repo-2",
                        "original_inst_id": "SWE-PolyBench__repo-2",
                        "source": "Poly",
                        "repo": "example/repo",
                        "commit": "def456",
                        "instruction": "Handle the poly task.",
                        "language": "python",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_tasks(task_path)

    assert [task["instance_id"] for task in loaded] == ["psf__requests-1142", "SWE-PolyBench__repo-2"]
    assert loaded[0]["prompt"] == "Fix request handling."
    assert loaded[0]["commit"] == "abc123"
    assert loaded[1]["prompt"] == "Handle the poly task."
    assert loaded[1]["bench"] == "Poly"


def test_load_tasks_honors_subset_order_filters_and_limit(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    subset_csv = tmp_path / "subset.csv"
    task_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "task-1",
                    "original_inst_id": "task-1",
                    "problem_statement": "one",
                },
                {
                    "instance_id": "task-2",
                    "original_inst_id": "task-2",
                    "problem_statement": "two",
                },
                {
                    "instance_id": "SWE-PolyBench__task-3",
                    "original_inst_id": "SWE-PolyBench__task-3",
                    "problem_statement": "three",
                },
            ]
        ),
        encoding="utf-8",
    )
    subset_csv.write_text(
        "instance_id,original_inst_id\n"
        "SWE-PolyBench__task-3,SWE-PolyBench__task-3\n"
        "task-1,task-1\n",
        encoding="utf-8",
    )

    loaded = load_tasks(
        task_path,
        subset_csv=subset_csv,
        bench_filter=["Poly", "Verified"],
        instance_filter=["task-1", "SWE-PolyBench__task-3"],
        limit=1,
    )

    assert [task["instance_id"] for task in loaded] == ["SWE-PolyBench__task-3"]


def test_load_tasks_raises_for_missing_subset_csv(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps([{"instance_id": "task-1", "problem_statement": "one"}]), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Configured subset CSV not found"):
        load_tasks(task_path, subset_csv=tmp_path / "missing-subset.csv")


def test_load_tasks_raises_for_subset_ids_missing_from_task_data(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    subset_csv = tmp_path / "subset.csv"
    task_path.write_text(json.dumps([{"instance_id": "task-1", "problem_statement": "one"}]), encoding="utf-8")
    subset_csv.write_text("instance_id,original_inst_id\nmissing-task,missing-task\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Configured subset contains IDs not found in task data: missing-task"):
        load_tasks(task_path, subset_csv=subset_csv)


def test_load_tasks_prefers_explicit_bench_metadata(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "legacy-owner__legacy-repo-1",
                    "source": "Poly",
                    "problem_statement": "one",
                }
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_tasks(task_path)

    assert loaded[0]["bench"] == "Poly"


def test_load_tasks_raises_for_conflicting_explicit_bench_metadata(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "SWE-PolyBench__repo-1",
                    "source": "Verified",
                    "problem_statement": "one",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Benchmark label conflicts"):
        load_tasks(task_path)


def test_load_tasks_raises_for_unknown_explicit_bench_metadata(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text(
        json.dumps([{"instance_id": "task-1", "source": "UnknownBench", "problem_statement": "one"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported benchmark label"):
        load_tasks(task_path)


def test_load_tasks_raises_for_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "missing.json")
