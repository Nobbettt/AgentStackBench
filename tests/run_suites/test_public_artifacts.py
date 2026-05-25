# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner

from .helpers import _write_task_inputs


def test_write_public_artifacts_does_not_sanitize_operational_tree_in_place(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "public-artifacts",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "runtime_backend": "host",
            },
            "variants": [{"name": "baseline", "runtime_backend": "host"}],
            "postprocess": {"convert": False, "evaluate": False, "runtime_backend": "host"},
        }
    )
    runner = RunSuiteRunner(config)
    runner.experiment_dir.mkdir(parents=True)
    operational_path = runner.experiment_dir / "record.json"
    operational_payload = {
        "workspace_path": "/Users/example/.cache/worktrees/contextbench_worktrees/github.com__example__repo/abc123__default",
        "stderr": "failed under /Users/example/private/project",
    }
    operational_path.write_text(json.dumps(operational_payload), encoding="utf-8")

    runner._write_public_artifacts()

    assert json.loads(operational_path.read_text(encoding="utf-8")) == operational_payload
    public_payload = json.loads((runner.public_artifacts_dir / "record.json").read_text(encoding="utf-8"))
    assert public_payload != operational_payload
    assert "<worktree>" in public_payload["workspace_path"]
    assert "<home>" in public_payload["stderr"]
