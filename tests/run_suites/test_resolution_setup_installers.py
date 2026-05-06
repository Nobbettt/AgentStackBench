
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import contextbench.run_suites_core.postprocess as postprocess
import contextbench.run_suites_setup as run_suites_setup
from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
    DEFAULT_CODEX_RUNTIME_IMAGE,
)
from contextbench.run_suites_core.postprocess import (
    ResolutionCommandError,
    describe_resolution_backend_support,
    evaluate_resolution_for_suite,
    export_resolution_predictions,
    run_resolution_evaluation,
)



def test_polybench_setup_installs_pinned_evaluator_into_expected_venv(monkeypatch, tmp_path: Path) -> None:
    expected_python = tmp_path / ".cache" / "polybench-eval-venv" / "bin" / "python"
    commands: list[list[str]] = []

    monkeypatch.setattr(run_suites_setup, "POLY_BENCH_PYTHON", expected_python)
    monkeypatch.setattr(run_suites_setup.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def fake_run(command, check):
        del check
        commands.append(list(command))
        if command[:2] == ["/usr/bin/uv", "venv"]:
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(run_suites_setup.subprocess, "run", fake_run)

    assert run_suites_setup.setup_polybench() == 0
    assert commands[0] == ["/usr/bin/uv", "venv", "--python", "3.11", str(expected_python.parents[1])]
    assert commands[1] == ["/usr/bin/uv", "pip", "install", "--python", str(expected_python), f"pip=={run_suites_setup.POLY_BENCH_PIP_VERSION}"]
    assert commands[2] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        str(expected_python),
        f"git+{run_suites_setup.POLY_BENCH_REPO}@{run_suites_setup.POLY_BENCH_COMMIT}",
        "-c",
        str(run_suites_setup.POLY_BENCH_CONSTRAINTS),
    ]
    assert commands[3] == [str(expected_python), "-m", "poly_bench_evaluation.run_evaluation", "--help"]


def test_multibench_setup_installs_pinned_evaluator_into_expected_venv(monkeypatch, tmp_path: Path) -> None:
    expected_python = tmp_path / ".cache" / "multibench-eval-venv" / "bin" / "python"
    constraints = tmp_path / "multibench.txt"
    constraints.write_text("multi-swe-bench @ git+https://github.com/multi-swe-bench/multi-swe-bench.git@24f493f8a103e72312ded4f6b9c89f081d69cb09\n", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(run_suites_setup, "MULTI_BENCH_PYTHON", expected_python)
    monkeypatch.setattr(run_suites_setup, "MULTI_BENCH_CONSTRAINTS", constraints)
    monkeypatch.setattr(run_suites_setup.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def fake_run(command, check):
        del check
        commands.append(list(command))
        if command[:2] == ["/usr/bin/uv", "venv"]:
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(run_suites_setup.subprocess, "run", fake_run)

    assert run_suites_setup.setup_multibench() == 0
    assert commands[0] == ["/usr/bin/uv", "venv", "--python", "3.11", str(expected_python.parents[1])]
    assert commands[1] == ["/usr/bin/uv", "pip", "install", "--python", str(expected_python), f"pip=={run_suites_setup.MULTI_BENCH_PIP_VERSION}"]
    assert commands[2] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        str(expected_python),
        f"git+{run_suites_setup.MULTI_BENCH_REPO}@{run_suites_setup.MULTI_BENCH_COMMIT}",
        "-c",
        str(constraints),
    ]
    assert commands[3] == [
        str(expected_python),
        "-c",
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('multi_swe_bench') else 1)",
    ]


def test_multibench_setup_repairs_upstream_qiskit_package_case(tmp_path: Path) -> None:
    python = tmp_path / ".cache" / "multibench-eval-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_repos = python.parents[1] / "lib" / "python3.11" / "site-packages" / "multi_swe_bench" / "harness" / "repos" / "python"
    qiskit_dir = python_repos / "Qiskit"
    qiskit_dir.mkdir(parents=True)
    (qiskit_dir / "__init__.py").write_text("from multi_swe_bench.harness.repos.python.qiskit.qiskit import *\n", encoding="utf-8")
    (python_repos / "__init__.py").write_text("from multi_swe_bench.harness.repos.python.qiskit import *\n", encoding="utf-8")

    run_suites_setup._repair_multibench_packaging_case_paths(python)

    assert (python_repos / "qiskit").exists()
    assert (qiskit_dir / "__init__.py").read_text(encoding="utf-8") == (
        "from multi_swe_bench.harness.repos.python.Qiskit.qiskit import *\n"
    )
    assert (python_repos / "__init__.py").read_text(encoding="utf-8") == (
        "from multi_swe_bench.harness.repos.python.Qiskit import *\n"
    )


def test_swebench_setup_installs_pinned_evaluator_into_expected_venv(monkeypatch, tmp_path: Path) -> None:
    expected_python = tmp_path / ".cache" / "swebench-eval-venv" / "bin" / "python"
    constraints = tmp_path / "swebench.txt"
    constraints.write_text("swebench==4.1.0\n", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(run_suites_setup, "SWE_BENCH_PYTHON", expected_python)
    monkeypatch.setattr(run_suites_setup, "SWE_BENCH_CONSTRAINTS", constraints)
    monkeypatch.setattr(run_suites_setup.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    sklearn_repairs: list[Path] = []
    sympy_repairs: list[Path] = []
    monkeypatch.setattr(run_suites_setup, "_repair_swebench_scikit_learn_testbed_pip", lambda python: sklearn_repairs.append(python))
    monkeypatch.setattr(
        run_suites_setup,
        "_repair_swebench_repo_setup_compatibility_patches",
        lambda python: sympy_repairs.append(python),
    )

    def fake_run(command, check):
        del check
        commands.append(list(command))
        if command[:2] == ["/usr/bin/uv", "venv"]:
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(run_suites_setup.subprocess, "run", fake_run)

    assert run_suites_setup.setup_swebench() == 0
    assert commands[0] == ["/usr/bin/uv", "venv", "--python", "3.11", str(expected_python.parents[1])]
    assert commands[1] == ["/usr/bin/uv", "pip", "install", "--python", str(expected_python), f"pip=={run_suites_setup.SWE_BENCH_PIP_VERSION}"]
    assert commands[2] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        str(expected_python),
        run_suites_setup.SWE_BENCH_PACKAGE,
        "-c",
        str(constraints),
    ]
    assert sklearn_repairs == [expected_python]
    assert sympy_repairs == [expected_python]
    assert commands[3] == [str(expected_python), "-m", "swebench.harness.run_evaluation", "--help"]


def test_swebench_setup_pins_scikit_learn_testbed_pip(tmp_path: Path) -> None:
    python = tmp_path / ".cache" / "swebench-eval-venv" / "bin" / "python"
    constants = python.parents[1] / "lib" / "python3.11" / "site-packages" / "swebench" / "harness" / "constants" / "python.py"
    constants.parent.mkdir(parents=True)
    constants.write_text(
        'SPECS_SKLEARN = {"1.6": {"pip_packages": ["cython"]}, "0.22": {}}\n'
        "SPECS_FLASK = {\n}\n",
        encoding="utf-8",
    )

    run_suites_setup._repair_swebench_scikit_learn_testbed_pip(python)
    run_suites_setup._repair_swebench_scikit_learn_testbed_pip(python)

    source = constants.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(source, namespace)

    sklearn_specs = namespace["SPECS_SKLEARN"]
    assert source.count("ContextBench pins scikit-learn testbed pip") == 1
    assert sklearn_specs["1.6"]["pip_packages"][0] == f"pip=={run_suites_setup.SWE_BENCH_SKLEARN_TESTBED_PIP_VERSION}"
    assert sklearn_specs["0.22"]["pip_packages"] == [f"pip=={run_suites_setup.SWE_BENCH_SKLEARN_TESTBED_PIP_VERSION}"]


def test_swebench_setup_applies_explicit_repo_setup_compatibility_patch(tmp_path: Path) -> None:
    python = tmp_path / ".cache" / "swebench-eval-venv" / "bin" / "python"
    test_spec = python.parents[1] / "lib" / "python3.11" / "site-packages" / "swebench" / "harness" / "test_spec" / "python.py"
    test_spec.parent.mkdir(parents=True)
    test_spec.write_text(
        """
REPO_BASE_COMMIT_BRANCH = {
    "sympy/sympy": {
        "cffd4e0f86fefd4802349a9f9b19ed70934ea354": "1.7",
    },
}

def make_repo_script_list_py(specs, repo, repo_directory, base_commit, env_name):
    branch = REPO_BASE_COMMIT_BRANCH.get(repo, {}).get(base_commit, "")
    branch = f"--branch {branch}" if branch else ""
    setup_commands = [
        f"git clone -o origin {branch} --single-branch https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        # Remove the remote and tags so the agent won't see newer commits.
        "git remote remove origin",
        # Remove only tags pointing to commits after target timestamp
        f"TARGET_TIMESTAMP=$(git show -s --format=%ci {base_commit})",
        'git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done',
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive",
        # Verify future logs aren't available
        "AFTER_TIMESTAMP=$(date -d \\"$TARGET_TIMESTAMP + 1 second\\" '+%Y-%m-%d %H:%M:%S')",
        'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)',
        '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    return setup_commands
""",
        encoding="utf-8",
    )

    run_suites_setup._repair_swebench_repo_setup_compatibility_patches(python)
    run_suites_setup._repair_swebench_repo_setup_compatibility_patches(python)

    source = test_spec.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(source, namespace)

    hidden_commit = run_suites_setup.SWE_BENCH_SYMPY_HIDDEN_BASE_COMMIT
    sympy_commands = namespace["make_repo_script_list_py"]({}, "sympy/sympy", "/testbed", hidden_commit, "testbed")
    regular_commands = namespace["make_repo_script_list_py"]({}, "sympy/sympy", "/testbed", "regular", "testbed")

    compatibility_patch = run_suites_setup.SWE_BENCH_REPO_SETUP_COMPATIBILITY_PATCHES[
        ("sympy/sympy", hidden_commit)
    ]

    assert compatibility_patch["evaluator"] == run_suites_setup.SWE_BENCH_PACKAGE
    assert compatibility_patch["fetch_ref"] == hidden_commit
    assert "deleted upstream branch 1.7" in compatibility_patch["reason"]
    assert source.count("ContextBench applies explicit SWE-bench repo setup compatibility patches") == 1
    assert sympy_commands[:7] == [
        "git init /testbed",
        "chmod -R 777 /testbed",
        "cd /testbed",
        "git remote add origin https://github.com/sympy/sympy",
        f"git fetch origin {hidden_commit}",
        "git checkout --detach FETCH_HEAD",
        f"git reset --hard {hidden_commit}",
    ]
    assert regular_commands[0] == "git clone -o origin  --single-branch https://github.com/sympy/sympy /testbed"


def test_swebench_setup_accepts_legacy_sympy_compatibility_marker(tmp_path: Path) -> None:
    python = tmp_path / ".cache" / "swebench-eval-venv" / "bin" / "python"
    test_spec = python.parents[1] / "lib" / "python3.11" / "site-packages" / "swebench" / "harness" / "test_spec" / "python.py"
    test_spec.parent.mkdir(parents=True)
    test_spec.write_text(
        "# Fork note: ContextBench fetches hidden SymPy 1.7 base commit by exact SHA.\n",
        encoding="utf-8",
    )

    run_suites_setup._repair_swebench_repo_setup_compatibility_patches(python)

    source = test_spec.read_text(encoding="utf-8")
    assert "ContextBench applies explicit SWE-bench repo setup compatibility patches" in source
    assert "fetches hidden SymPy 1.7 base commit" not in source


def test_setup_constraints_reject_unpinned_entries(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("requests==2.33.1\nunidiff\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unpinned constraint"):
        run_suites_setup._verify_fully_pinned_constraints(constraints)
