
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.runtime import (
    build_claude_command,
    build_codex_command,
    git_workspace_diff,
    git_untracked_files,
    prepare_claude_runtime_files,
    prepare_codex_runtime_env,
    run_command,
    run_coding_agent_task,
    validate_claude_auth,
    validate_claude_isolation,
)
from contextbench.coding_agents.runtime_backends import (
    RuntimeBackendConfig,
    DockerTaskRuntime,
    normalize_runtime_backend_config,
)
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"


def test_git_workspace_diff_includes_untracked_files_and_restores_index(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "new.txt").write_text("created\n", encoding="utf-8")
    (repo / "new file with space.txt").write_text("spaced\n", encoding="utf-8")

    diff = git_workspace_diff(repo)

    assert "diff --git a/tracked.txt b/tracked.txt" in diff
    assert "diff --git a/new.txt b/new.txt" in diff
    assert "diff --git a/new file with space.txt b/new file with space.txt" in diff
    assert "+created" in diff
    assert "+spaced" in diff
    status = subprocess.run(["git", "status", "--short"], cwd=repo, check=True, capture_output=True, text=True)
    assert set(status.stdout.splitlines()) == {" M tracked.txt", "?? new.txt", '?? "new file with space.txt"'}


def test_run_coding_agent_task_codex_setup_prompt_runs_before_scored_prompt(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-setup",
        "original_inst_id": "task-setup",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"final_output_paths": {}, "calls": []}
    reset_calls: list[Path] = []

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))

    def fake_reset_workspace(path: Path) -> None:
        reset_calls.append(path)
        setup_marker = path / "setup-ran.txt"
        if setup_marker.exists():
            setup_marker.unlink()

    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", fake_reset_workspace)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home"), "EXPERIMENT": "1"},
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])

    def fake_build_codex_command(**kwargs):
        phase = "setup" if kwargs["schema_path"] is None else "main"
        captured["final_output_paths"][phase] = kwargs["final_output_path"]
        return ["codex", "exec", phase], f"{phase}-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        phase = command[-1]
        assert phase in {"setup", "main"}
        captured["calls"].append(
            {
                "phase": phase,
                "cwd": cwd,
                "stdin_text": stdin_text,
                "timeout": timeout,
                "env": dict(env or {}),
            }
        )
        stdout_path.write_text(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11 if phase == "setup" else 4, "output_tokens": 3 if phase == "setup" else 2},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_paths"][phase]
        assert isinstance(final_output_path, Path)
        if phase == "setup":
            (cwd / "setup-ran.txt").write_text("yes", encoding="utf-8")
            final_output_path.write_text("setup complete", encoding="utf-8")
        else:
            assert (cwd / "setup-ran.txt").exists()
            final_output_path.write_text(
                json.dumps(
                    make_final_output(
                        task_id="task-setup",
                        touched_files=["a.py"],
                        retrieved_context_files=["a.py"],
                    )
                ),
                encoding="utf-8",
            )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        setup={"setup_prompt": "Bootstrap tools", "setup_prompt_timeout": 12},
        runtime_backend="host",
    )

    task_dir = output_dir / "task-setup"

    assert [call["phase"] for call in captured["calls"]] == ["setup", "main"]
    assert captured["calls"][0]["cwd"] == workspace_path
    assert captured["calls"][1]["cwd"] == workspace_path
    assert captured["calls"][0]["timeout"] == 12
    assert captured["calls"][1]["timeout"] == 30
    assert record["status"] == "completed"
    assert record["token_usage"]["input_tokens"] == 4
    assert record["setup_run"]["status"] == "completed"
    assert record["setup_run"]["token_usage"]["input_tokens"] == 11
    assert record["raw_response_path"] != record["setup_run"]["raw_response_path"]
    assert Path(record["setup_run"]["prompt_path"]).name == "setup-prompt.txt"
    assert Path(record["setup_run"]["stderr_path"]).name == "setup-stderr.log"
    assert Path(record["setup_run"]["raw_response_path"]).exists()
    assert (task_dir / "setup-last-message.txt").read_text(encoding="utf-8") == "setup complete"
    assert Path(record["raw_response_path"]).exists()
    assert reset_calls == [workspace_path]

def test_run_coding_agent_task_runtime_setup_command_short_circuits_scored_prompt(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-setup-fail",
        "original_inst_id": "task-runtime-setup-fail",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_workspace_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.run_command",
        lambda *args, **kwargs: pytest.fail("scored prompt should not run after runtime setup failure"),
    )
    times = iter([100.0, 105.0])
    monkeypatch.setattr("contextbench.coding_agents.runtime_backends.time.time", lambda: next(times))

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=["printf 'missing deps' >&2; exit 7"],
    )

    task_dir = output_dir / "task-runtime-setup-fail"

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["duration_ms"] == 5000
    assert record["exit_code"] == 7
    assert record["raw_response_path"] is None
    assert not (task_dir / "raw-response.json").exists()
    assert (task_dir / "runtime-setup-1.stdout.log").exists()
    assert (task_dir / "runtime-setup-1.stderr.log").read_text(encoding="utf-8") == "missing deps"
    assert record["notes"] == "Runtime setup command failed before setup prompts or scored work."
    assert record["runtime_failure"] == {
        "phase": "runtime-setup",
        "command": "printf 'missing deps' >&2; exit 7",
        "stdout_path": str(task_dir / "runtime-setup-1.stdout.log"),
        "stderr_path": str(task_dir / "runtime-setup-1.stderr.log"),
    }
    assert record["setup_contamination"] == {"tracked_diff": False, "untracked_files": []}


def test_run_coding_agent_task_runtime_commands_receive_contextbench_shell_env(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-shell-env",
        "original_inst_id": "task-runtime-shell-env",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, Path] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )

    def fake_build_codex_command(**kwargs):
        captured["final_output_path"] = kwargs["final_output_path"]
        return ["codex", "exec", "-"], "codex-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        captured["final_output_path"].write_text(
            json.dumps(
                make_final_output(
                    task_id="task-runtime-shell-env",
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=[
            'test "$CONTEXTBENCH_WORKSPACE_PATH" = "$PWD" && printf setup > "$CONTEXTBENCH_TASK_DIR/setup-env.txt"',
        ],
        runtime_validation_commands=[
            'test "$CONTEXTBENCH_WORKSPACE_PATH" = "$PWD" && printf validation > "$CONTEXTBENCH_TASK_DIR/validation-env.txt"',
        ],
    )

    task_dir = output_dir / "task-runtime-shell-env"
    assert record["status"] == "completed"
    assert (task_dir / "setup-env.txt").read_text(encoding="utf-8") == "setup"
    assert (task_dir / "validation-env.txt").read_text(encoding="utf-8") == "validation"


def test_run_coding_agent_task_uses_separate_runtime_command_timeouts(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-command-timeouts",
        "original_inst_id": "task-runtime-command-timeouts",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, Path] = {}
    command_timeouts: list[tuple[list[str], int, str]] = []

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.run_runtime_setup_commands",
        lambda runtime, *, commands, workspace_path, task_dir, timeout, env, artifact_prefix="runtime-setup": (
            command_timeouts.append((list(commands), timeout, artifact_prefix)) or None
        ),
    )

    def fake_build_codex_command(**kwargs):
        captured["final_output_path"] = kwargs["final_output_path"]
        return ["codex", "exec", "-"], "codex-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, cwd, stdin_text, timeout, env
        captured["final_output_path"].write_text(
            json.dumps(
                make_final_output(
                    task_id="task-runtime-command-timeouts",
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=["setup-command"],
        runtime_validation_commands=["validation-command"],
        runtime_setup_timeout=120,
        runtime_validation_timeout=45,
    )

    assert record["status"] == "completed"
    assert command_timeouts == [
        (["setup-command"], 120, "runtime-setup"),
        (["validation-command"], 45, "runtime-validation"),
    ]


def test_run_coding_agent_task_runtime_setup_cache_restores_workspace_artifacts(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_paths = [tmp_path / "workspace-1", tmp_path / "workspace-2"]
    for workspace_path in workspace_paths:
        workspace_path.mkdir()
    output_dirs = [tmp_path / "results-1", tmp_path / "results-2"]
    cache_dir = tmp_path / "cache" / "repos"
    setup_cache_dir = tmp_path / "cache" / "runtime-setup"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-setup-cache",
        "original_inst_id": "task-runtime-setup-cache",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    checkouts = iter(workspace_paths)
    captured: dict[str, Path] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(next(checkouts)))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )

    def fake_build_codex_command(**kwargs):
        captured["final_output_path"] = kwargs["final_output_path"]
        return ["codex", "exec", "-"], "codex-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        del command, stdin_text, timeout, env
        assert (cwd / ".context" / "cache.txt").read_text(encoding="utf-8") == "cached"
        captured["final_output_path"].write_text(
            json.dumps(
                make_final_output(
                    task_id="task-runtime-setup-cache",
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)
    setup_command = (
        'if test -f .context/cache.txt; then printf hit > "$CONTEXTBENCH_TASK_DIR/cache-state.txt"; '
        'else mkdir -p .context && printf cached > .context/cache.txt && '
        'printf miss > "$CONTEXTBENCH_TASK_DIR/cache-state.txt"; fi'
    )

    first_record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dirs[0],
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=[setup_command],
        runtime_setup_cache=True,
        runtime_setup_cache_dir=setup_cache_dir,
        diff_exclude_paths=[".context/**"],
    )
    second_record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dirs[1],
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=[setup_command],
        runtime_setup_cache=True,
        runtime_setup_cache_dir=setup_cache_dir,
        diff_exclude_paths=[".context/**"],
    )

    first_task_dir = output_dirs[0] / "task-runtime-setup-cache"
    second_task_dir = output_dirs[1] / "task-runtime-setup-cache"
    cache_entries = list(setup_cache_dir.iterdir())

    assert first_record["status"] == "completed"
    assert second_record["status"] == "completed"
    assert first_task_dir.joinpath("cache-state.txt").read_text(encoding="utf-8") == "miss"
    assert second_task_dir.joinpath("cache-state.txt").read_text(encoding="utf-8") == "hit"
    assert first_record["runtime_setup_cache"]["hit"] is False
    assert second_record["runtime_setup_cache"]["hit"] is True
    assert first_record["runtime_setup_cache"]["saved"] is True
    assert second_record["runtime_setup_cache"]["saved"] is True
    assert len(cache_entries) == 1
    assert cache_entries[0].joinpath("workspace/.context/cache.txt").read_text(encoding="utf-8") == "cached"


def test_run_coding_agent_task_runtime_validation_failure_records_untracked_files(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-validation-fail",
        "original_inst_id": "task-runtime-validation-fail",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: ["validation-leftover.txt"])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.run_command",
        lambda *args, **kwargs: pytest.fail("scored prompt should not run after runtime validation failure"),
    )

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_validation_commands=["touch validation-leftover.txt; exit 9"],
    )

    assert record["status"] == "failed"
    assert record["exit_code"] == 9
    assert record["runtime_failure"] == {
        "phase": "runtime-validation",
        "command": "touch validation-leftover.txt; exit 9",
        "stdout_path": str(output_dir / "task-runtime-validation-fail" / "runtime-validation-1.stdout.log"),
        "stderr_path": str(output_dir / "task-runtime-validation-fail" / "runtime-validation-1.stderr.log"),
    }
    assert record["setup_contamination"] == {
        "tracked_diff": False,
        "untracked_files": ["validation-leftover.txt"],
    }

def test_run_coding_agent_task_fails_when_runtime_setup_creates_untracked_files(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-setup",
        "original_inst_id": "task-runtime-setup",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    reset_calls: list[Path] = []

    def fake_reset_workspace(path: Path) -> None:
        reset_calls.append(path)

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", fake_reset_workspace)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: ["setup-ran.txt"])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.__setitem__("final_output_path", kwargs["final_output_path"]) or ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        if command[:3] == ["/bin/sh", "-c", "touch setup-ran.txt"]:
            (cwd / "setup-ran.txt").write_text("yes", encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}
        pytest.fail("scored prompt should not run after setup creates untracked files")

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=["touch setup-ran.txt"],
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["setup_contamination"] == {
        "tracked_diff": False,
        "untracked_files": ["setup-ran.txt"],
    }
    assert record["model_patch"] == ""
    assert record["diff_path"] is None
    assert (output_dir / "task-runtime-setup" / "runtime-setup-1.stdout.log").exists()
    assert reset_calls == [workspace_path]


def test_git_untracked_files_fails_loudly_when_git_ls_files_fails(tmp_path, monkeypatch) -> None:
    class Result:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    with pytest.raises(RuntimeError, match="git ls-files failed while checking untracked files"):
        git_untracked_files(tmp_path)


def test_run_coding_agent_task_fails_when_runtime_setup_changes_tracked_files(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-runtime-setup-contaminated",
        "original_inst_id": "task-runtime-setup-contaminated",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path, **kwargs: "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path, **kwargs: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        if command[:3] != ["/bin/sh", "-c", "touch tracked-file"]:
            pytest.fail("scored prompt should not run after setup contamination")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        runtime_backend="host",
        runtime_setup_commands=["touch tracked-file"],
    )

    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["model_patch"].startswith("diff --git")
    assert record["raw_response_path"] is None
