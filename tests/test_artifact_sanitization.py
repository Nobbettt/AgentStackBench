
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import (
    SanitizationContext,
    assert_no_private_paths,
    find_private_path_matches,
    scan_paths_for_private_artifacts,
    sanitize_artifact_tree,
    sanitize_artifact_tree_in_place,
    sanitize_json_value,
    sanitize_text,
)


def test_sanitize_text_rewrites_worktree_and_home_paths() -> None:
    repo_root = Path("/Users/nobbe/Repos/ContextBench")
    workspace = repo_root / ".cache/worktrees/contextbench_worktrees/github.com__django__django/django"
    context = SanitizationContext(repo_root=repo_root, workspace_path=workspace)

    text = (
        "opened /Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/"
        "github.com__django__django/django/db/models.py and /Users/nobbe/.codex/auth.json "
        "plus /private/var/folders/abc/T/contextbench_worktrees/github.com__django__django/django/forms.py"
    )

    sanitized = sanitize_text(text, context=context)

    assert "<worktree>/db/models.py" in sanitized
    assert "<worktree>/github.com__django__django/django/forms.py" in sanitized
    assert "<home>/.codex/auth.json" in sanitized
    assert find_private_path_matches(sanitized) == []


def test_sanitize_json_value_recurses_and_asserts() -> None:
    context = SanitizationContext(repo_root=Path("/Users/nobbe/Repos/ContextBench"))

    sanitized = sanitize_json_value(
        {
            "final_answer": "see /Users/nobbe/Repos/ContextBench/results/run_suites/demo/log.txt",
            "trace": [{"output": "temp file /var/folders/abc/contextbench.txt"}],
        },
        context=context,
    )

    assert "<contextbench-root>/results/run_suites/demo/log.txt" in json.dumps(sanitized)
    assert "<tmp>" in json.dumps(sanitized)
    assert_no_private_paths(sanitized)
    with pytest.raises(ValueError):
        assert_no_private_paths({"leak": "/Users/nobbe/secret.txt"})


def test_sanitize_text_rewrites_env_path_segments_after_colon() -> None:
    raw = (
        "PYTHONPATH=src:/var/folders/35/j3z165c92v57rwbh03pn5pk80000gn/T/"
        "fake_dist_00depdja python3 -m unittest"
    )

    sanitized = sanitize_text(raw)

    assert "/var/folders" not in sanitized
    assert "PYTHONPATH=src:<tmp>" in sanitized
    assert find_private_path_matches(sanitized) == []


def test_sanitize_artifact_tree_writes_publishable_copy(tmp_path: Path) -> None:
    source = tmp_path / "results" / "run_suites" / "demo"
    source.mkdir(parents=True)
    (source / "record.json").write_text(
        json.dumps(
            {
                "workspace_path": "/Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/github.com__d/repo",
                "final_output": {"final_answer": "used /Users/nobbe/.codex/auth.json"},
            }
        ),
        encoding="utf-8",
    )
    (source / "resolution-command.log").write_text(
        "traceback in /Users/nobbe/Repos/ContextBench/contextbench/run_suites.py\n",
        encoding="utf-8",
    )

    output = tmp_path / "public" / "demo"
    sanitize_artifact_tree(
        source_dir=source,
        output_dir=output,
        repo_root=Path("/Users/nobbe/Repos/ContextBench"),
    )

    assert scan_paths_for_private_artifacts([output]) == {}
    assert "<worktree>/github.com__d/repo" in (output / "record.json").read_text(encoding="utf-8")


def test_sanitize_artifact_tree_in_place_rewrites_operational_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "results" / "run_suites" / "demo"
    source.mkdir(parents=True)
    (source / "record.json").write_text(
        json.dumps(
            {
                "workspace_path": "/Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/github.com__d/repo",
                "final_answer": "used /Users/nobbe/.codex/auth.json",
            }
        ),
        encoding="utf-8",
    )
    (source / "codex-events.jsonl").write_text(
        json.dumps({"message": "opened /Users/nobbe/Repos/ContextBench/.cache/worktrees/contextbench_worktrees/github.com__d/repo/a.py"})
        + "\n",
        encoding="utf-8",
    )

    sanitize_artifact_tree_in_place(
        source_dir=source,
        repo_root=Path("/Users/nobbe/Repos/ContextBench"),
    )

    assert scan_paths_for_private_artifacts([source]) == {}
    assert "<worktree>/github.com__d/repo" in (source / "record.json").read_text(encoding="utf-8")
