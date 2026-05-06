
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
import contextbench.run_suites_setup as run_suites_setup
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CODEX_RUNTIME_IMAGE,
)
from contextbench.run_suites_core.postprocess import (
    ResolutionCommandError,
    describe_resolution_backend_support,
    evaluate_resolution_for_suite,
    export_resolution_predictions,
    run_resolution_evaluation,
)


from .helpers import _fake_run_coding_agent_task, _make_fake_agent_record, _write_task_inputs

def test_write_pro_raw_sample_csv_normalizes_required_columns(tmp_path: Path) -> None:
    raw_sample_jsonl = tmp_path / "sweap_eval_full_v2.jsonl"
    raw_sample_jsonl.write_text(
        json.dumps(
            {
                "instance_id": "instance_repo__repo-1",
                "before_repo_set_cmd": "git checkout abc",
                "selected_test_files_to_run": ["tests/test_a.py"],
                "base_commit": "abc",
                "repo": "owner/repo",
                "FAIL_TO_PASS": ["tests/test_a.py::test_bug"],
                "PASS_TO_PASS": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "raw-sample.csv"
    postprocess._write_pro_raw_sample_csv(
        raw_sample_jsonl=raw_sample_jsonl,
        instance_ids=["instance_repo__repo-1"],
        out_path=out_path,
    )

    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "instance_repo__repo-1"
    assert rows[0]["fail_to_pass"] == "['tests/test_a.py::test_bug']"
    assert rows[0]["pass_to_pass"] == "[]"
    assert rows[0]["selected_test_files_to_run"] == "['tests/test_a.py']"
def test_write_poly_dataset_csv_filters_selected_instances(tmp_path: Path, monkeypatch) -> None:
    dataset_rows = [
        {
            "instance_id": "poly-a",
            "patch": "gold-a",
            "test_patch": "test-a",
            "repo": "owner/repo-a",
            "base_commit": "abc",
            "language": "Python",
            "Dockerfile": "FROM python:3.11",
            "F2P": ["tests/test_a.py::test_bug"],
            "P2P": [],
            "test_command": "pytest tests/test_a.py",
            "modified_nodes": ["a.py::f"],
        },
        {
            "instance_id": "poly-b",
            "patch": "gold-b",
            "test_patch": "test-b",
            "repo": "owner/repo-b",
            "base_commit": "def",
            "language": "Java",
            "Dockerfile": "FROM eclipse-temurin:17",
            "F2P": ["TestB"],
            "P2P": [],
            "test_command": "./gradlew test",
            "modified_nodes": ["B.java::f"],
        },
    ]
    monkeypatch.setattr(
        postprocess,
        "_load_dataset_rows",
        lambda dataset_name, split: (dataset_rows, list(dataset_rows[0].keys())),
    )

    out_path = tmp_path / "poly-subset.csv"
    postprocess._write_poly_dataset_csv(
        dataset_name="AmazonScience/SWE-PolyBench",
        instance_ids=["poly-a"],
        out_path=out_path,
    )

    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "poly-a"
    assert rows[0]["language"] == "Python"
    assert rows[0]["F2P"] == "['tests/test_a.py::test_bug']"
    assert rows[0]["P2P"] == "[]"
    assert rows[0]["modified_nodes"] == '["a.py::f"]'
def test_multi_dataset_export_reads_target_huggingface_jsonl_shard(tmp_path: Path, monkeypatch) -> None:
    hf_mod = type(sys)("huggingface_hub")
    calls: list[dict[str, object]] = []
    shard = tmp_path / "iamkun__dayjs_dataset.jsonl"
    shard.write_text(
        '{"org":"iamkun","repo":"dayjs","number":734,"instance_id":"iamkun__dayjs-734","fix_patch":"gold"}\n',
        encoding="utf-8",
    )

    def fake_list_repo_files(dataset_name, repo_type):
        calls.append({"fn": "list_repo_files", "dataset_name": dataset_name, "repo_type": repo_type})
        return ["js/iamkun__dayjs_dataset.jsonl"]

    def fake_hf_hub_download(repo_id, repo_type, filename):
        calls.append({"fn": "hf_hub_download", "repo_id": repo_id, "repo_type": repo_type, "filename": filename})
        return str(shard)

    hf_mod.list_repo_files = fake_list_repo_files
    hf_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)

    out_path = tmp_path / "dataset.jsonl"
    postprocess._write_multi_dataset_jsonl(
        dataset_name="bytedance-research/Multi-SWE-Bench",
        instance_ids=["iamkun__dayjs-734"],
        out_path=out_path,
    )

    assert calls == [
        {"fn": "list_repo_files", "dataset_name": "bytedance-research/Multi-SWE-Bench", "repo_type": "dataset"},
        {
            "fn": "hf_hub_download",
            "repo_id": "bytedance-research/Multi-SWE-Bench",
            "repo_type": "dataset",
            "filename": "js/iamkun__dayjs_dataset.jsonl",
        },
    ]
    assert json.loads(out_path.read_text(encoding="utf-8"))["instance_id"] == "iamkun__dayjs-734"
def test_multi_dataset_export_does_not_fallback_to_language_aggregate(tmp_path: Path, monkeypatch) -> None:
    hf_mod = type(sys)("huggingface_hub")

    def fake_list_repo_files(dataset_name, repo_type):
        del dataset_name, repo_type
        return ["python/multi_swe_bench_python.jsonl"]

    def fake_hf_hub_download(repo_id, repo_type, filename):
        raise AssertionError(f"unexpected aggregate download: {repo_id} {repo_type} {filename}")

    hf_mod.list_repo_files = fake_list_repo_files
    hf_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)

    with pytest.raises(RuntimeError, match="has no JSONL shard"):
        postprocess._write_multi_dataset_jsonl(
            dataset_name="bytedance-research/Multi-SWE-Bench",
            instance_ids=["iamkun__dayjs-734"],
            out_path=tmp_path / "dataset.jsonl",
        )
