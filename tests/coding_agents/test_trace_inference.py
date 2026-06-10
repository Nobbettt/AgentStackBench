# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-08 from upstream ContextBench.
# Summary of changes: cover trace guards, read-span inference, and effective file normalization.

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


def test_infer_retrieval_step_ignores_rg_file_lists_without_line_hits(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/zsh -lc 'rg --files | rg \"pkg|tests\"'"
    output = "pkg/mod.py\ntests/test_mod.py\npkg/extra.py\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_broad_rg_line_hits(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/zsh -lc 'rg -n \"target\" src tests -g \"*.py\"'"
    output = "src/a.py:10: target\ntests/test_a.py:20: target\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_keeps_single_file_search_file_only(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/zsh -lc 'rg -n \"target\" src/a.py'"
    output = "src/a.py:10: target\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {"files": ["src/a.py"], "spans": {}, "symbols": {}}


def test_infer_retrieval_step_ignores_find_file_lists(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/zsh -lc 'find . -name \"*.py\"'"
    output = "./pkg/mod.py\n./tests/test_mod.py\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_truncated_find_file_lists(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/zsh -lc \"find . -name '*.py' | sort | sed -n '1,80p'\""
    output = "./pkg/mod.py\n./tests/test_mod.py\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_from_plain_sed_range(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc \"sed -n '10,20p' src/a.py\""
    output = "def first():\n    pass\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 10, "end": 20}]},
        "symbols": {},
    }


def test_infer_retrieval_step_from_nl_pipe_sed_range(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc \"nl -ba src/a.py | sed -n '70,230p'\""
    output = "    70\tdef first():\n   230\t    return value\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 70, "end": 230}]},
        "symbols": {},
    }


def test_infer_retrieval_step_from_multiline_nl_pipe_sed_ranges(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = (
        '/bin/bash -lc "nl -ba src/a.py | sed -n '
        "'70,230p'\n"
        "nl -ba tests/test_a.py | sed -n '45,70p'\""
    )
    output = "    70\tdef first():\n    45\tdef test_first():\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py", "tests/test_a.py"],
        "spans": {
            "src/a.py": [{"start": 70, "end": 230}],
            "tests/test_a.py": [{"start": 45, "end": 70}],
        },
        "symbols": {},
    }


def test_infer_retrieval_step_from_direct_head_range(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'head -n 50 src/a.py'"
    output = "first line\nsecond line\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 1, "end": 50}]},
        "symbols": {},
    }


def test_infer_retrieval_step_does_not_treat_head_after_file_list_as_read_span(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'rg --files src tests | head -n 50'"
    output = "src/a.py\ntests/test_a.py\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_keeps_plain_cat_file_only(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'cat pyproject.toml'"
    output = "[project]\nname = 'demo'\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {"files": ["pyproject.toml"], "spans": {}, "symbols": {}}


def test_infer_retrieval_step_ignores_empty_cat_output(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'cat pyproject.toml'"

    step = infer_retrieval_step_from_command(command, output_text="", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_redirected_cat_probe(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'cat .babelrc 2>/dev/null || echo no-babelrc'"

    step = infer_retrieval_step_from_command(command, output_text="no-babelrc\n", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_negative_head_count(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'head -n -5 src/a.py'"

    step = infer_retrieval_step_from_command(command, output_text="content\n", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_keeps_compact_head_count(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'head -5 src/a.py'"

    step = infer_retrieval_step_from_command(command, output_text="line\n", workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 1, "end": 5}]},
        "symbols": {},
    }


def test_infer_retrieval_step_uses_numbered_output_for_nl_without_sed(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'nl -ba src/a.py'"
    output = "     1\tdef first():\n    42\t    return value\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 1, "end": 42}]},
        "symbols": {},
    }


def test_infer_retrieval_step_scores_followup_sed_range_without_grep_hit_span(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc \"rg -n 'target' src/a.py && sed -n '30,60p' src/a.py\""
    output = "src/a.py:35: target\ncontext text\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step == {
        "files": ["src/a.py"],
        "spans": {"src/a.py": [{"start": 30, "end": 60}]},
        "symbols": {},
    }


def test_infer_retrieval_step_handles_pipe_before_single_file_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc \"printf 'target' | rg target src/a.py\""

    step = infer_retrieval_step_from_command(command, output_text="target\n", workspace_path=workspace)

    assert step == {"files": ["src/a.py"], "spans": {}, "symbols": {}}


def test_infer_retrieval_step_ignores_empty_single_file_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'rg missing src/a.py'"

    step = infer_retrieval_step_from_command(command, output_text="", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_quiet_single_file_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'rg -q target src/a.py'"

    step = infer_retrieval_step_from_command(command, output_text="", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_redirected_single_file_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'rg target src/a.py >/tmp/hits.txt'"

    step = infer_retrieval_step_from_command(command, output_text="target\n", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_file_listing_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc 'rg -l target src/a.py'"

    step = infer_retrieval_step_from_command(command, output_text="src/a.py\n", workspace_path=workspace)

    assert step is None


def test_infer_retrieval_step_ignores_remote_sed_ranges(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "/bin/bash -lc \"curl -L https://example.com/src/a.py | sed -n '1,50p'\""
    output = "remote content\n"

    step = infer_retrieval_step_from_command(command, output_text=output, workspace_path=workspace)

    assert step is None


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
