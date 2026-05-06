# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: cover safe repo-root path inference, trace guards, and effective file normalization for coding-agent parsers.

from __future__ import annotations

import json

import jsonschema
import pytest

from contextbench.agents.claude import ClaudeAgentParser
from contextbench.agents.codex import CodexAgentParser
from contextbench.coding_agents import (
    build_claude_raw_response,
    build_codex_raw_response,
    convert_run_record,
    extract_structured_output_from_value,
)
from contextbench.coding_agents.constants import CLAUDE_OUTPUT_SCHEMA_PATH, CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.trace_inference import (
    infer_file_list_from_text,
    infer_grep_spans_from_text,
    infer_retrieval_step_from_command,
    trajectory_from_steps,
)
from contextbench.parsers.trajectory import parse_trajectory

def test_infer_grep_spans_from_text_caps_match_volume(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lines = "\n".join(f"src/file_{i}.py:{i}: hit" for i in range(700))

    spans = infer_grep_spans_from_text(lines, workspace)

    assert sum(len(v) for v in spans.values()) == 700

def test_infer_grep_spans_from_text_supports_repo_root_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = "README.md:12: hit\nDockerfile:5: FROM python\n"

    spans = infer_grep_spans_from_text(text, workspace)

    assert spans == {
        "Dockerfile": [{"start": 5, "end": 5}],
        "README.md": [{"start": 12, "end": 12}],
    }

def test_infer_file_list_from_text_caps_match_volume(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lines = "\n".join(f"src/file_{i}.py" for i in range(700))

    files = infer_file_list_from_text(lines, workspace)

    assert len(files) == 700

def test_infer_file_list_from_text_supports_repo_root_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = "README.md\npyproject.toml\nDockerfile\n"

    files = infer_file_list_from_text(text, workspace)

    assert files == ["Dockerfile", "README.md", "pyproject.toml"]

def test_parse_trajectory_derives_effective_files_from_spans_and_symbols() -> None:
    steps, final_step = parse_trajectory(
        {
            "traj_data": {
                "pred_steps": [
                    {
                        "files": [],
                        "spans": {"src/a.py": [{"start": 1, "end": 3}]},
                        "symbols": {"src/b.py": ["func"]},
                    }
                ],
                "pred_files": [],
                "pred_spans": {"src/c.py": [{"start": 4, "end": 8}]},
                "pred_symbols": {"src/d.py": ["Class.method"]},
            }
        }
    )

    assert steps[0].files == ["src/a.py", "src/b.py"]
    assert final_step is not None
    assert final_step.files == ["src/c.py", "src/d.py"]

def test_infer_file_list_from_text_ignores_environment_variable_lines(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = "PATH=/usr/local/bin:/usr/bin:/bin:/opt/tooling.with.dots\nLLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config\n"

    files = infer_file_list_from_text(text, workspace)

    assert files == []

def test_infer_grep_spans_from_text_ignores_environment_variable_lines(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = "PATH=/usr/local/bin:/usr/bin:/bin:/opt/tooling.with.dots\nLLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config\n"

    spans = infer_grep_spans_from_text(text, workspace)

    assert spans == {}

def test_infer_retrieval_step_from_long_quoted_command_ignores_toolchain_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = (
        '/bin/zsh -lc "make -n LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config '
        "UNAME_M=aarch64 CFLAGS='-Wno-unused-but-set-variable' "
        "CXXFLAGS='-Wno-deprecated-declarations' libponyc.tests | rg 'matchtype\\\\.cc|matchtype\\\\.o'\""
    )
    output = "/opt/homebrew/Cellar/llvm/bin/llvm-ar rcs build/release/libponyc.a build/release/obj/libponyc/type/matchtype.o\n"
    meta = {}

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace, meta=meta)

    assert step is None
    assert meta == {}

def test_trajectory_from_steps_prefers_grounded_files_over_search_only_files() -> None:
    traj = trajectory_from_steps(
        [
            {"files": ["a.py", "b.py", "c.py"], "spans": {}, "symbols": {}},
            {"files": ["core.py"], "spans": {"core.py": [{"start": 10, "end": 20}]}, "symbols": {}},
        ]
    )

    assert traj is not None
    assert traj["pred_files"] == ["core.py"]


def test_trajectory_from_steps_does_not_synthesize_empty_trajectory() -> None:
    assert trajectory_from_steps([]) is None
