
from __future__ import annotations

import sys
from pathlib import Path

from contextbench.artifact_sanitization import find_private_path_matches
from contextbench.run_suites_core import postprocess


def test_resolution_command_log_sanitizes_private_paths(tmp_path: Path) -> None:
    leaked_path = (
        postprocess._REPO_ROOT
        / ".cache"
        / "worktrees"
        / "contextbench_worktrees"
        / "github.com__django__django"
        / "django"
        / "db"
        / "models.py"
    )
    log_path = tmp_path / "resolution-command.log"

    returncode, tail = postprocess._run_resolution_command(
        command=[sys.executable, "-c", f"print({str(leaked_path)!r})"],
        cwd=tmp_path,
        log_path=log_path,
        log_prefix="[test]",
        heartbeat_interval_seconds=0,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert returncode == 0
    assert find_private_path_matches(tail) == []
    assert find_private_path_matches(log_text) == []
    assert "<worktree>/github.com__django__django/django/db/models.py" in log_text
