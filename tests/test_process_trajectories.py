
from __future__ import annotations

import argparse
import json

from contextbench import process_trajectories


def test_cmd_convert_returns_nonzero_and_writes_summary_for_partial_coding_agent_run(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    variant_dir = tmp_path / "baseline"
    source_dir = variant_dir / "agent_runs" / "codex"
    ok_dir = source_dir / "Verified" / "task-ok"
    fail_dir = source_dir / "Verified" / "task-fail"
    ok_dir.mkdir(parents=True)
    fail_dir.mkdir(parents=True)
    ok_record_path = ok_dir / "task-ok.codex-record.json"
    fail_record_path = fail_dir / "task-fail.codex-record.json"
    ok_record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-ok",
                final_output=make_final_output(task_id="task-ok", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    fail_record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-fail",
                "status": "failed",
                "final_output": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "task-ok", "record_path": str(ok_record_path)}),
                json.dumps({"instance_id": "task-fail", "record_path": str(fail_record_path)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pred.jsonl"

    rc = process_trajectories.cmd_convert(
        argparse.Namespace(
            input=[str(source_dir)],
            out=str(out_path),
            summary_out=None,
            agent="codex",
            recursive=False,
            fail_on_partial=False,
        )
    )

    assert rc == 1
    assert out_path.exists()
    summary_path = tmp_path / "pred.summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent"] == "codex"
    assert summary["input_count"] == 1
    assert summary["processed_input_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["selected_task_count"] == 2
    assert summary["missing_prediction_count"] == 1
    assert summary["is_partial"] is True


def test_cmd_convert_fail_on_partial_returns_nonzero_for_coding_agent_run(
    tmp_path,
    make_final_output,
    make_record,
) -> None:
    variant_dir = tmp_path / "baseline"
    source_dir = variant_dir / "agent_runs" / "codex"
    ok_dir = source_dir / "Verified" / "task-ok"
    fail_dir = source_dir / "Verified" / "task-fail"
    ok_dir.mkdir(parents=True)
    fail_dir.mkdir(parents=True)
    ok_record_path = ok_dir / "task-ok.codex-record.json"
    fail_record_path = fail_dir / "task-fail.codex-record.json"
    ok_record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-ok",
                final_output=make_final_output(task_id="task-ok", retrieved_context_files=["a.py"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    fail_record_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "instance_id": "task-fail",
                "status": "failed",
                "final_output": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "task-results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "task-ok", "record_path": str(ok_record_path)}),
                json.dumps({"instance_id": "task-fail", "record_path": str(fail_record_path)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pred.jsonl"

    rc = process_trajectories.cmd_convert(
        argparse.Namespace(
            input=[str(source_dir)],
            out=str(out_path),
            summary_out=None,
            agent="codex",
            recursive=False,
            fail_on_partial=True,
        )
    )

    assert rc == 1
    summary = json.loads((tmp_path / "pred.summary.json").read_text(encoding="utf-8"))
    assert summary["is_partial"] is True
