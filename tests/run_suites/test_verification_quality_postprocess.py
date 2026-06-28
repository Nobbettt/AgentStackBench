# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.run_suites_core.postprocess import convert_records_to_jsonl


def test_convert_records_writes_verification_quality_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "variant" / "agent_runs" / "codex"
    task_dir = source_dir / "Verified" / "task-1"
    task_dir.mkdir(parents=True)
    record_path = task_dir / "task-1.codex-record.json"
    record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "bench": "Verified",
                "instance_id": "task-1",
                "original_inst_id": "task-1",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "final_output": {
                    "status": "completed",
                    "final_answer": "Done",
                    "retrieved_context_files": ["src/app.py"],
                    "retrieved_context_spans": [{"file": "src/app.py", "start": 1, "end": 3}],
                    "retrieved_context_symbols": [],
                    "notes": "",
                },
                "command_executions": [
                    {
                        "command": "pytest tests/test_app.py -q",
                        "payload": {"exit_code": 0},
                    }
                ],
                "model_patch": (
                    "diff --git a/tests/test_app.py b/tests/test_app.py\n"
                    "--- a/tests/test_app.py\n"
                    "+++ b/tests/test_app.py\n"
                    "@@\n"
                    "+def test_app():\n"
                    "+    assert True\n"
                ),
            }
        ),
        encoding="utf-8",
    )
    task_results_path = source_dir / "task-results.jsonl"
    task_results_path.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "original_inst_id": "task-1",
                "bench": "Verified",
                "status": "completed",
                "ok": True,
                "timeout": False,
                "record_path": str(record_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = source_dir.parent.parent / "pred.jsonl"

    summary = convert_records_to_jsonl(
        source_dir=source_dir,
        expected_agent="codex",
        out_path=out_path,
    )

    quality_path = Path(str(summary["verification_quality_path"]))
    quality_summary_path = Path(str(summary["verification_quality_summary_path"]))
    csv_path = Path(str(summary["verification_quality_csv_path"]))
    quality_rows = [json.loads(line) for line in quality_path.read_text(encoding="utf-8").splitlines()]
    quality_summary = json.loads(quality_summary_path.read_text(encoding="utf-8"))

    assert quality_rows[0]["verification_quality"]["strongest_verification"] == "repo_tests"
    assert quality_rows[0]["regression_test"]["regression_tests_run"] is True
    assert quality_summary["successful_runtime_verification_count"] == 1
    assert csv_path.exists()
