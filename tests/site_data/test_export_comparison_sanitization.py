
from __future__ import annotations

import json
from pathlib import Path

from contextbench.artifact_sanitization import find_private_path_matches
from scripts.export_comparison_data import build_comparison_export

from .helpers import _write


def test_export_sanitizes_final_answer_trace_and_patch_paths(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    variant_dir = suite_dir / "variants" / "baseline"
    task_dir = variant_dir / "agent_runs" / "codex" / "Verified" / "django__django-1"
    workspace = Path(
        "/Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/"
        "github.com__django__django/django"
    )
    raw_response_path = task_dir / "raw-response.json"
    record_path = task_dir / "django__django-1.codex-record.json"

    _write(
        suite_dir / "experiment.json",
        json.dumps({"experiment_name": "demo-suite", "description": "demo", "agent": "codex"}),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps([{"variant": "baseline", "total_tasks": 1, "completed_tasks": 1}]),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "started_at": "2026-04-29T00:00:00Z",
                "completed_at": "2026-04-29T00:01:00Z",
                "task_set": {"count": 1, "bench_counts": {"Verified": 1}},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(variant_dir / "effective-config.json"),
                        "task_results_path": str(variant_dir / "task-results.jsonl"),
                        "output_dir": str(variant_dir),
                    }
                ],
            }
        ),
    )
    _write(
        variant_dir / "effective-config.json",
        json.dumps({"effective_config": {"name": "baseline", "model": "gpt-5.4", "reasoning_effort": "high"}}),
    )
    _write(
        variant_dir / "task-results.jsonl",
        json.dumps({"instance_id": "django__django-1", "bench": "Verified", "status": "completed", "record_path": str(record_path)})
        + "\n",
    )
    _write(variant_dir / "eval.jsonl", json.dumps({"instance_id": "django__django-1"}) + "\n")
    _write(
        variant_dir / "resolution-summary.json",
        json.dumps({"status": "completed", "pass_at_1": 1.0, "resolved_count": 1, "resolved_ids": ["django__django-1"]}),
    )
    _write(
        raw_response_path,
        json.dumps(
            {
                "agent": "codex",
                "response_format": "jsonl-events",
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "status": "completed",
                            "command": f"sed -n 1,5p {workspace / 'django/db/models.py'}",
                            "aggregated_output": f"Traceback from {workspace / 'django/db/models.py'}",
                            "exit_code": 0,
                        },
                    },
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "I used /Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/github.com__django__django/django/db/models.py",
                        },
                    },
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "file_change",
                            "path": str(workspace / "django/db/models.py"),
                            "description": "Changed /Users/nobbe/.codex/auth.json by mistake in prose only",
                        },
                    },
                ],
            }
        ),
    )
    _write(
        record_path,
        json.dumps(
            {
                "agent": "codex",
                "bench": "Verified",
                "instance_id": "django__django-1",
                "original_inst_id": "django__django-1",
                "repo_url": "https://github.com/django/django.git",
                "commit": "abc123",
                "language": "python",
                "workspace_path": str(workspace),
                "task_dir": str(task_dir),
                "status": "completed",
                "ok": True,
                "timeout": False,
                "duration_ms": 10,
                "token_usage": {"total_tokens": 100},
                "tool_calls": [],
                "raw_response_path": str(raw_response_path),
                "model_patch": f"diff --git a/x b/x\n+path {workspace / 'django/db/models.py'}\n",
                "final_output": {
                    "status": "completed",
                    "final_answer": f"fixed using {workspace / 'django/db/models.py'}",
                    "notes": "local notes mention /Users/nobbe/Repos/ContextBench/results/run_suites/demo",
                    "retrieved_context_files": ["django/db/models.py"],
                },
            }
        ),
    )

    payload, detail_payloads = build_comparison_export(suite_dir)
    serialized = json.dumps({"payload": payload, "details": detail_payloads}, ensure_ascii=False)

    assert find_private_path_matches(serialized) == []
    assert "<worktree>/django/db/models.py" in serialized
    assert "<home>/.codex/auth.json" in serialized
