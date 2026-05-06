
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
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path: [])

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
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.run_command",
        lambda *args, **kwargs: pytest.fail("scored prompt should not run after runtime setup failure"),
    )

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
    assert record["exit_code"] == 7
    assert record["raw_response_path"] is None
    assert not (task_dir / "raw-response.json").exists()
    assert (task_dir / "runtime-setup-1.stdout.log").exists()
    assert (task_dir / "runtime-setup-1.stderr.log").read_text(encoding="utf-8") == "missing deps"

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
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path: "")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path: ["setup-ran.txt"])
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
        if command[:3] == ["/bin/sh", "-lc", "touch setup-ran.txt"]:
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


def test_git_untracked_files_fails_loudly_when_git_status_fails(tmp_path, monkeypatch) -> None:
    class Result:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    with pytest.raises(RuntimeError, match="git status failed while checking setup contamination"):
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
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_tracked_diff", lambda path: "diff --git a/a.py b/a.py\n")
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_untracked_files", lambda path: [])
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        if command[:3] != ["/bin/sh", "-lc", "touch tracked-file"]:
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
