
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

def test_build_prompt_includes_benchmark_guidance_and_omits_identity_echo() -> None:
    codex_prompt = build_prompt(
        {
            "bench": "Verified",
            "repo": "psf/requests",
            "instance_id": "task-1",
            "prompt": "Fix it.",
        },
        "codex",
    )
    claude_prompt = build_prompt(
        {
            "bench": "Verified",
            "repo": "psf/requests",
            "instance_id": "task-1",
            "prompt": "Fix it.",
        },
        "claude",
    )

    for prompt in (codex_prompt, claude_prompt):
        assert "Consider the following PR description:" in prompt
        assert "Work inside the checked-out repository workspace for this task." in prompt
        assert 'set the final schema status to "completed"' in prompt
        assert 'Reserve "partial" only for genuinely unfinished implementation' in prompt
        assert "Fix it." in prompt
    assert "Task ID:" not in codex_prompt
    assert "Task ID:" not in claude_prompt
    assert "Agent:" not in codex_prompt
    assert "Agent:" not in claude_prompt

def test_build_prompt_dispatch_accepts_agent_aliases() -> None:
    prompt = build_prompt(
        {"bench": "Verified", "instance_id": "task-1", "prompt": "Fix it."},
        "claude-code",
    )

    assert "Consider the following PR description:" in prompt

def test_run_module_supports_codex_dry_run_with_task_data(tmp_path) -> None:
    task_data = tmp_path / "tasks.json"
    task_csv = tmp_path / "tasks.csv"
    task_data.write_text(
        json.dumps(
            [
                {
                    "instance_id": "psf__requests-1142",
                    "original_inst_id": "psf__requests-1142",
                    "repo_url": "https://github.com/psf/requests.git",
                    "base_commit": "abc123",
                    "problem_statement": "Fix a bug in requests.",
                    "language": "python",
                }
            ]
        ),
        encoding="utf-8",
    )
    task_csv.write_text(
        "bench,instance_id,original_inst_id,language,status,patch_files,patch_blocks,patch_span,gold_context_length,num_agents,repo,commit\n"
        "Verified,psf__requests-1142,psf__requests-1142,python,pass,1,1,1,10,1,,abc123\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "contextbench.run",
            "--agent",
            "codex",
            "--task-data",
            str(task_data),
            "--task-csv",
            str(task_csv),
            "--dry-run",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Loaded 1 tasks" in result.stderr
