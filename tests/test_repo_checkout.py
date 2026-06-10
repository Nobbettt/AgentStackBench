
from __future__ import annotations

import os
import subprocess

from contextbench.core.repo import _worktree_dir, checkout


def _git(args: list[str], *, cwd) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


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


def test_checkout_cleans_reused_worktree_before_returning(tmp_path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(["init"], cwd=repo_dir)
    _git(["config", "user.email", "test@example.com"], cwd=repo_dir)
    _git(["config", "user.name", "Test User"], cwd=repo_dir)
    (repo_dir / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(["add", "tracked.txt"], cwd=repo_dir)
    _git(["commit", "-m", "initial"], cwd=repo_dir)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()

    monkeypatch.setenv("CONTEXTBENCH_TMP_ROOT", str(tmp_path / "tmp"))
    first_worktree = checkout(str(repo_dir), commit, str(tmp_path / "cache"), verbose=False, workspace_key="eval")
    assert first_worktree is not None
    worktree_path = os.fspath(first_worktree)
    stale_path = os.path.join(worktree_path, "stale.py")
    tracked_path = os.path.join(worktree_path, "tracked.txt")
    with open(stale_path, "w", encoding="utf-8") as handle:
        handle.write("stale\n")
    with open(tracked_path, "w", encoding="utf-8") as handle:
        handle.write("dirty\n")

    second_worktree = checkout(str(repo_dir), commit, str(tmp_path / "cache"), verbose=False, workspace_key="eval")

    assert second_worktree == first_worktree
    assert not os.path.exists(stale_path)
    with open(tracked_path, encoding="utf-8") as handle:
        assert handle.read() == "clean\n"


def test_checkout_cleans_reused_worktree_under_repo_lock(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    base_dir = cache_dir / "github.com__example__repo"
    (base_dir / ".git").mkdir(parents=True)

    monkeypatch.setenv("CONTEXTBENCH_TMP_ROOT", str(tmp_path / "tmp"))
    worktree_dir = tmp_path / "tmp" / "contextbench_worktrees" / "github.com__example__repo" / "abc123__eval"
    worktree_dir.mkdir(parents=True)
    lock_state = {"locked": False, "cleaned_under_lock": False}

    class FakeLock:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self):
            lock_state["locked"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            lock_state["locked"] = False
            return False

    def fake_git(args, cwd=None, show_progress=False, timeout=600):
        del cwd, show_progress, timeout
        if args[:2] == ["fetch", "--depth"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        if args[:2] == ["worktree", "prune"]:
            return subprocess.CompletedProcess(["git", *args], 0, "", "")
        raise AssertionError(f"unexpected git call: {args}")

    def fake_clean(existing_worktree: str, commit: str, *, verbose: bool) -> bool:
        del commit, verbose
        assert existing_worktree == os.fspath(worktree_dir)
        lock_state["cleaned_under_lock"] = lock_state["locked"]
        return True

    monkeypatch.setattr("contextbench.core.repo._file_lock", FakeLock)
    monkeypatch.setattr("contextbench.core.repo._git", fake_git)
    monkeypatch.setattr("contextbench.core.repo._verify_commit", lambda work_dir, expected: work_dir == os.fspath(worktree_dir) and expected == "abc123")
    monkeypatch.setattr("contextbench.core.repo._clean_existing_worktree", fake_clean)

    worktree = checkout("https://github.com/example/repo.git", "abc123", str(cache_dir), verbose=False, workspace_key="eval")

    assert worktree == os.fspath(worktree_dir)
    assert lock_state["cleaned_under_lock"] is True
