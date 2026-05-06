
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.artifact_sanitization import find_private_path_matches
from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.runtime import (
    build_claude_command,
    build_codex_command,
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
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.codex.adapter import CodexAdapter


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    start = next(
        (index for index in range(len(values) - len(expected) + 1) if values[index : index + len(expected)] == expected),
        None,
    )
    assert start is not None, f"{expected!r} not found in {values!r}"


def test_run_coding_agent_task_requires_explicit_repo_url(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    checkout_calls: list[object] = []
    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.checkout",
        lambda *args, **kwargs: checkout_calls.append((args, kwargs)) or str(tmp_path / "workspace"),
    )

    with pytest.raises(RuntimeError, match="missing required repo_url"):
        run_coding_agent_task(
            task={
                "bench": "Verified",
                "instance_id": "example__repo-1",
                "original_inst_id": "example__repo-1",
                "commit": "abc123",
                "prompt": "Fix the bug.",
                "language": "python",
            },
            agent="codex",
            output_dir=Path("results"),
            cache_dir=Path("cache"),
            schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
            runtime_backend="host",
        )

    assert checkout_calls == []


def test_run_coding_agent_task_requires_explicit_runtime_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="runtime_backend is required"):
        run_coding_agent_task(
            task={
                "bench": "Verified",
                "instance_id": "example__repo-1",
                "original_inst_id": "example__repo-1",
                "repo_url": "https://github.com/example/repo.git",
                "commit": "abc123",
                "prompt": "Fix the bug.",
                "language": "python",
            },
            agent="codex",
            output_dir=Path("results"),
            cache_dir=Path("cache"),
            schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
        )


def test_run_coding_agent_task_rejects_claude_docker_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="Claude Docker runtime is not supported"):
        run_coding_agent_task(
            task={
                "bench": "Verified",
                "instance_id": "example__repo-1",
                "original_inst_id": "example__repo-1",
                "repo_url": "https://github.com/example/repo.git",
                "commit": "abc123",
                "prompt": "Fix the bug.",
                "language": "python",
            },
            agent="claude",
            output_dir=Path("results"),
            cache_dir=Path("cache"),
            schema_path=CODEX_OUTPUT_SCHEMA_PATH.resolve(),
            timeout=30,
            runtime_backend="docker",
        )


def test_run_coding_agent_task_codex_writes_record_and_diff(tmp_path, monkeypatch, make_final_output) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = Path("results")
    cache_dir = Path("cache")
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-1",
        "original_inst_id": "task-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    reset_calls: list[Path] = []

    def fake_reset_workspace(path: Path) -> None:
        reset_calls.append(path)
        setup_marker = path / "setup-ran.txt"
        if setup_marker.exists():
            setup_marker.unlink()

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", fake_reset_workspace)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"])
            and captured.setdefault("reasoning_effort", kwargs["reasoning_effort"])
            and captured.setdefault("writable_dirs", kwargs["writable_dirs"])
            and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "diff --git a/a.py b/a.py\n")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["stdin_text"] = stdin_text
        captured["env"] = env
        stdout_path.write_text(
            json.dumps({"type": "message", "message": f"opened {workspace_path / 'a.py'}"}) + "\n"
            + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-1",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                    final_answer=f"checked {workspace_path / 'a.py'}",
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
        reasoning_effort="high",
        env_overrides={"EXPERIMENT": "1"},
        prompt_preamble="Variant instructions",
        runtime_backend="host",
    )

    task_dir = (tmp_path / "results" / "task-1").resolve()
    record_path = task_dir / "task-1.codex-record.json"
    public_record_path = task_dir / "task-1.codex-record.public.json"

    assert record["agent"] == "codex"
    assert record["final_output"]["task_id"] == "task-1"
    assert record["tool_calls"] == []
    assert record["model_patch"].startswith("diff --git")
    assert Path(record["raw_response_path"]).exists()
    assert Path(record["diff_path"]).exists()
    assert record_path.exists()
    assert public_record_path.exists()
    assert find_private_path_matches((task_dir / "codex-events.jsonl").read_text(encoding="utf-8")) == []
    assert find_private_path_matches((task_dir / "final-output.json").read_text(encoding="utf-8")) == []
    assert find_private_path_matches((task_dir / "raw-response.json").read_text(encoding="utf-8")) == []
    assert find_private_path_matches(public_record_path.read_text(encoding="utf-8")) == []
    assert "<task-artifacts>" in public_record_path.read_text(encoding="utf-8")
    prompt_text = (task_dir / "prompt.txt").read_text(encoding="utf-8")
    assert prompt_text.startswith("Variant instructions")
    assert "Consider the following PR description:" in prompt_text
    assert "Work inside the checked-out repository workspace for this task." in prompt_text
    assert "Do not add extra bookkeeping fields beyond the required schema." in prompt_text
    assert captured["reasoning_effort"] == "high"
    assert isinstance(captured["final_output_path"], Path)
    assert captured["final_output_path"].is_absolute()
    assert [str(path) for path in captured["writable_dirs"]] == [str(codex_runtime_root(task_dir).resolve())]
    assert Path(record["task_dir"]).is_absolute()
    assert captured["cwd"] == workspace_path
    assert captured["env"] == {"HOME": str(task_dir), "EXPERIMENT": "1"}

def test_run_coding_agent_task_codex_retries_transient_failure(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-retry",
        "original_inst_id": "task-retry",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"attempt": 0, "final_output_path": None}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.__setitem__("final_output_path", kwargs["final_output_path"]) or ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr("contextbench.agents.codex.runtime.time.sleep", lambda seconds: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["attempt"] = int(captured["attempt"]) + 1
        attempt = int(captured["attempt"])
        if attempt == 1:
            stdout_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "thread.started", "thread_id": "t-1"}),
                        json.dumps({"type": "turn.started"}),
                        json.dumps({"type": "error", "message": "Reconnecting... 2/5 (We're currently experiencing high demand, which may cause temporary errors.)"}),
                        json.dumps({"type": "turn.failed", "error": {"message": "unexpected status 401 Unauthorized: Missing bearer or basic authentication in header"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text(
                "failed to connect to websocket: HTTP error: 500 Internal Server Error\n",
                encoding="utf-8",
            )
            return {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-retry",
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
        runtime_backend="host",
    )

    task_dir = (output_dir / "task-retry").resolve()

    assert captured["attempt"] == 2
    assert record["status"] == "completed"
    assert record["ok"] is True
    assert record["token_usage"]["input_tokens"] == 5
    assert (task_dir / "stderr.attempt1.log").exists()
    assert (task_dir / "raw-response.attempt1.json").exists()
    assert (task_dir / "codex-events.attempt1.jsonl").exists()
    assert "high demand" in (task_dir / "raw-response.attempt1.json").read_text(encoding="utf-8")

def test_run_coding_agent_task_passes_workspace_key_to_checkout(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-workspace-key",
        "original_inst_id": "task-workspace-key",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    def fake_checkout(*args, **kwargs):
        captured["workspace_key"] = kwargs.get("workspace_key")
        return str(workspace_path)

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", fake_checkout)
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"]) and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-workspace-key",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        workspace_key="suite__task__variant",
        runtime_backend="host",
    )

    assert captured["workspace_key"] == "suite__task__variant"
