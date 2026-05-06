# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: add run-scoped worktree keys, explicit worktree cleanup helpers, and collision-safe worktree paths.

"""Git repository checkout management."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, List, Iterable

def checkout(
    repo_url: str,
    commit: str,
    cache_dir: str,
    verbose: bool = True,
    sparse_paths: Optional[List[str]] = None,
    workspace_key: str | None = None,
    tmp_root: str | None = None,
) -> Optional[str]:
    """
    Checkout repo at specific commit.

    Concurrency-safe strategy:

    - Keep one shared "base" clone per repo under cache_dir (never switch its HEAD).
    - Create a dedicated detached worktree per commit under /tmp (or a configurable tmp root).
      This prevents multiple commits from fighting over the same working directory.
    """
    if not repo_url or not commit or not cache_dir:
        return None

    sparse_list = _normalize_sparse_paths(sparse_paths)
    
    repo_key = _normalize_url(repo_url)

    os.makedirs(cache_dir, exist_ok=True)

    # If repo_url is an existing local git clone, use it as the base directly.
    if os.path.isdir(repo_url) and os.path.isdir(os.path.join(repo_url, ".git")):
        base_dir = repo_url
    else:
        base_dir = os.path.join(cache_dir, repo_key)

    tmp_root = tmp_root or os.environ.get("CONTEXTBENCH_TMP_ROOT") or tempfile.gettempdir()
    worktree_root = os.path.join(tmp_root, "contextbench_worktrees", repo_key)
    worktree_dir = _worktree_dir(worktree_root, commit, workspace_key)

    # Fast path: worktree already exists at the right commit
    if os.path.isdir(worktree_dir) and _verify_commit(worktree_dir, commit):
        if sparse_list:
            _ensure_sparse_checkout(worktree_dir, commit, sparse_list, verbose=verbose)
        return worktree_dir

    lock_path = os.path.join(cache_dir, f"{repo_key}.lock")
    with _file_lock(lock_path):
        # Ensure base repo exists (only when not using an existing local clone)
        if not os.path.isdir(os.path.join(base_dir, ".git")):
            if verbose:
                print(f"  Cloning base repo {repo_url}", file=sys.stderr)
            for attempt in range(1, 4):
                if os.path.isdir(base_dir):
                    try:
                        subprocess.run(["rm", "-rf", base_dir], check=False)
                    except Exception:
                        pass
                result = _git(
                    ["clone", "--filter=blob:none", "--no-checkout", "--progress", repo_url, base_dir],
                    show_progress=verbose,
                    timeout=1800,
                )
                if result.returncode == 0 and os.path.isdir(os.path.join(base_dir, ".git")):
                    break
                if verbose:
                    print(f"  Base clone failed (attempt {attempt}/3)", file=sys.stderr)
            else:
                return None

        # Fetch the desired commit into the base repo (does not change HEAD)
        _git(
            ["fetch", "--depth", "1", "--filter=blob:none", "origin", commit],
            cwd=base_dir,
            show_progress=verbose,
            timeout=1800,
        )

        # Clean up stale worktree registrations (best-effort)
        _git(["worktree", "prune"], cwd=base_dir, timeout=600)

        # If another process created it while we waited for the lock, reuse it.
        if os.path.isdir(worktree_dir) and _verify_commit(worktree_dir, commit):
            if sparse_list:
                _ensure_sparse_checkout(worktree_dir, commit, sparse_list, verbose=verbose)
            return worktree_dir

        os.makedirs(os.path.dirname(worktree_dir), exist_ok=True)

        # Create a detached worktree for the specific commit.
        wt_args = ["worktree", "add", "--detach"]
        if sparse_list:
            # Avoid checking out the full tree (inode explosion on huge repos).
            wt_args.append("--no-checkout")
        wt_args += [worktree_dir, commit]
        wt = _git(wt_args, cwd=base_dir, show_progress=verbose, timeout=1800)
        if wt.returncode != 0:
            # Legacy layouts and interrupted runs can leave behind a plain
            # directory at the target path. If it is not a valid worktree,
            # remove it once and retry.
            if os.path.isdir(worktree_dir) and not _verify_commit(worktree_dir, commit):
                shutil.rmtree(worktree_dir, ignore_errors=True)
                wt = _git(wt_args, cwd=base_dir, show_progress=verbose, timeout=1800)
            if wt.returncode != 0 and _clear_stale_worktree_registration(
                base_dir,
                worktree_dir,
                verbose=verbose,
            ):
                wt = _git(wt_args, cwd=base_dir, show_progress=verbose, timeout=1800)
            # If it failed because the directory/worktree exists, try to reuse.
            if os.path.isdir(worktree_dir) and _verify_commit(worktree_dir, commit):
                if sparse_list:
                    _ensure_sparse_checkout(worktree_dir, commit, sparse_list, verbose=verbose)
                return worktree_dir
            return None

        if sparse_list:
            _ensure_sparse_checkout(worktree_dir, commit, sparse_list, verbose=verbose)

    return worktree_dir if _verify_commit(worktree_dir, commit) else None


def remove_worktree(
    repo_url: str,
    cache_dir: str,
    worktree_dir: str,
    *,
    verbose: bool = False,
) -> None:
    """Remove a detached worktree created by checkout()."""
    if not repo_url or not cache_dir or not worktree_dir:
        return
    if not os.path.isdir(worktree_dir):
        return

    repo_key = _normalize_url(repo_url)
    if os.path.isdir(repo_url) and os.path.isdir(os.path.join(repo_url, ".git")):
        base_dir = repo_url
    else:
        base_dir = os.path.join(cache_dir, repo_key)
    if not os.path.isdir(os.path.join(base_dir, ".git")):
        shutil.rmtree(worktree_dir, ignore_errors=True)
        return

    lock_path = os.path.join(cache_dir, f"{repo_key}.lock")
    with _file_lock(lock_path):
        _git(["worktree", "remove", "--force", worktree_dir], cwd=base_dir, show_progress=verbose, timeout=600)
        shutil.rmtree(worktree_dir, ignore_errors=True)
        _git(["worktree", "prune"], cwd=base_dir, timeout=600)


def _clear_stale_worktree_registration(base_dir: str, worktree_dir: str, *, verbose: bool) -> bool:
    """Best-effort recovery for stale git worktree registrations.

    Detached worktrees can be left registered in the shared base clone even when
    the worktree path itself has disappeared, for example if a container-hosted
    evaluation process is interrupted. In that state, a later `git worktree add`
    fails with messages such as "missing but locked worktree".

    We recover deterministically by trying to unlock, remove, and prune the
    registration under the repo lock before retrying the add once.
    """
    if not base_dir or not worktree_dir:
        return False
    recovered = False
    for args in (
        ["worktree", "unlock", worktree_dir],
        ["worktree", "remove", "--force", worktree_dir],
        ["worktree", "prune"],
    ):
        result = _git(args, cwd=base_dir, show_progress=verbose, timeout=600)
        if result.returncode == 0:
            recovered = True
    if os.path.isdir(worktree_dir):
        shutil.rmtree(worktree_dir, ignore_errors=True)
        recovered = True
    return recovered

def _normalize_url(url: str) -> str:
    """Convert git URL to directory-safe name."""
    s = re.sub(r"^https?://", "", url.strip())
    s = re.sub(r"^git@", "", s).replace(":", "/").rstrip("/")
    s = s.replace("/", "__").replace(".git", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or "repo"


def _normalize_workspace_key(workspace_key: str | None) -> str:
    if workspace_key is None:
        return ""
    value = str(workspace_key).strip()
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or ""


def _worktree_dir(worktree_root: str, commit: str, workspace_key: str | None) -> str:
    """Build a collision-safe detached worktree path.

    The unkeyed evaluator checkout and keyed benchmark worktrees must never
    share a parent path that is itself also used as a worktree. Using sibling
    paths avoids collisions like:
      <repo>/<commit>
      <repo>/<commit>/<workspace_key>
    where the parent commit directory ceases to be a valid git worktree.
    """
    workspace_component = _normalize_workspace_key(workspace_key)
    suffix = workspace_component or "default"
    return os.path.join(worktree_root, f"{commit}__{suffix}")

class _file_lock:
    def __init__(self, path: str):
        self.path = path
        self.f = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.f = open(self.path, "a+", encoding="utf-8")
        try:
            import fcntl  # Linux
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        except Exception:
            # If flock is unavailable, proceed without a lock.
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.f:
                try:
                    import fcntl
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                self.f.close()
        finally:
            self.f = None
        return False

def _git(args, cwd=None, show_progress=False, timeout: int = 600):
    """Run git command."""
    env = os.environ.copy()
    # Prevent git-lfs from downloading large blobs during checkout.
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    if show_progress:
        # Show real-time output for clone/fetch operations
        return subprocess.run(
            ["git"] + args, cwd=cwd, check=False, timeout=timeout, env=env
        )
    else:
        return subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )

def _verify_commit(work_dir: str, expected: str) -> bool:
    """Check if working directory is at expected commit."""
    result = _git(["rev-parse", "HEAD"], cwd=work_dir)
    return result.returncode == 0 and result.stdout.strip() == expected


def _normalize_sparse_paths(paths: Optional[Iterable[str]]) -> List[str]:
    """Normalize sparse paths to repo-relative POSIX-style paths."""
    if not paths:
        return []
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        s = str(p).strip().strip("'\"").replace("\\", "/")
        if not s:
            continue
        while s.startswith("./"):
            s = s[2:]
        s = s.lstrip("/")
        if not s or s.startswith(".."):
            continue
        out.append(s)
    # Deduplicate while keeping stable order.
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _ensure_sparse_checkout(worktree_dir: str, commit: str, sparse_paths: List[str], verbose: bool = True) -> None:
    """Configure sparse checkout for this worktree and checkout commit."""
    if not sparse_paths:
        return
    # Initialize sparse-checkout for this worktree (non-cone to allow file paths).
    _git(["sparse-checkout", "init", "--no-cone"], cwd=worktree_dir, show_progress=verbose, timeout=600)
    # Set sparse paths (prunes everything else from the worktree).
    _git(["sparse-checkout", "set", "--no-cone", "--"] + list(sparse_paths), cwd=worktree_dir, show_progress=verbose, timeout=600)
    # Ensure the intended commit is checked out (with sparse rules applied).
    _git(["checkout", "--detach", commit], cwd=worktree_dir, show_progress=verbose, timeout=1800)
