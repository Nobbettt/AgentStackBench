
from __future__ import annotations

import os
import subprocess

from contextbench.core.repo import _worktree_dir, checkout


def test_worktree_dir_uses_sibling_paths_for_default_and_keyed_worktrees(tmp_path) -> None:
    root = str(tmp_path / "worktrees")

    default_dir = _worktree_dir(root, "abc123", None)
    keyed_dir = _worktree_dir(root, "abc123", "suite__task__variant")

    assert default_dir.endswith(os.path.join("worktrees", "abc123__default"))
    assert keyed_dir.endswith(os.path.join("worktrees", "abc123__suite__task__variant"))
    assert keyed_dir != default_dir
    assert not keyed_dir.startswith(default_dir + os.sep)


def test_checkout_retries_after_stale_worktree_directory(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    base_dir = cache_dir / "github.com__example__repo"
    (base_dir / ".git").mkdir(parents=True)

    monkeypatch.setenv("CONTEXTBENCH_TMP_ROOT", str(tmp_path / "tmp"))

    attempts = {"worktree_add": 0}

    def fake_git(args, cwd=None, show_progress=False, timeout=600):
        del cwd, show_progress, timeout
        if args[:2] == ["fetch", "--depth"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:2] == ["worktree", "prune"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:3] == ["worktree", "add", "--detach"]:
            attempts["worktree_add"] += 1
            target = args[-2]
            os.makedirs(target, exist_ok=True)
            code = 1 if attempts["worktree_add"] == 1 else 0
            return subprocess.CompletedProcess(["git", *args], code, "", "")
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    def fake_verify_commit(work_dir: str, expected: str) -> bool:
        del expected
        return attempts["worktree_add"] >= 2 and os.path.isdir(work_dir)

    monkeypatch.setattr("contextbench.core.repo._git", fake_git)
    monkeypatch.setattr("contextbench.core.repo._verify_commit", fake_verify_commit)

    worktree = checkout(
        "https://github.com/example/repo.git",
        "abc123",
        str(cache_dir),
        verbose=False,
        workspace_key="suite__task__variant",
    )

    assert worktree is not None
    assert worktree.endswith(os.path.join("abc123__suite__task__variant"))
    assert attempts["worktree_add"] == 2


def test_checkout_clears_stale_locked_worktree_registration(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    base_dir = cache_dir / "github.com__example__repo"
    (base_dir / ".git").mkdir(parents=True)

    monkeypatch.setenv("CONTEXTBENCH_TMP_ROOT", str(tmp_path / "tmp"))

    calls: list[list[str]] = []
    attempts = {"worktree_add": 0}

    def fake_git(args, cwd=None, show_progress=False, timeout=600):
        del cwd, show_progress, timeout
        calls.append(list(args))
        if args[:2] == ["fetch", "--depth"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:2] == ["worktree", "prune"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:2] == ["worktree", "unlock"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:3] == ["worktree", "remove", "--force"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:3] == ["worktree", "add", "--detach"]:
            attempts["worktree_add"] += 1
            target = args[-2]
            if attempts["worktree_add"] == 1:
                return subprocess.CompletedProcess(["git", *args], 1, "", "missing but locked worktree")
            os.makedirs(target, exist_ok=True)
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    def fake_verify_commit(work_dir: str, expected: str) -> bool:
        del expected
        return attempts["worktree_add"] >= 2 and os.path.isdir(work_dir)

    monkeypatch.setattr("contextbench.core.repo._git", fake_git)
    monkeypatch.setattr("contextbench.core.repo._verify_commit", fake_verify_commit)

    worktree = checkout(
        "https://github.com/example/repo.git",
        "abc123",
        str(cache_dir),
        verbose=False,
        workspace_key="suite__task__variant",
    )

    assert worktree is not None
    assert attempts["worktree_add"] == 2
    assert ["worktree", "unlock", worktree] in calls
    assert ["worktree", "remove", "--force", worktree] in calls
