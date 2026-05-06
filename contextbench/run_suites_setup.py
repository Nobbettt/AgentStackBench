
"""Deterministic setup helpers for run-suite postprocess backends."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from contextbench.coding_agents.constants import (
    DEFAULT_CODEX_RUNTIME_IMAGE,
    DEFAULT_POSTPROCESS_RUNTIME_IMAGE,
)
from contextbench.run_suites_core.postprocess import _postprocess_image_supports_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
SWE_BENCH_PYTHON = REPO_ROOT / ".cache" / "swebench-eval-venv" / "bin" / "python"
SWE_BENCH_PACKAGE = "swebench==4.1.0"
SWE_BENCH_CONSTRAINTS = REPO_ROOT / "contextbench" / "run_suites_constraints" / "swebench.txt"
SWE_BENCH_PIP_VERSION = "26.0.1"
SWE_BENCH_SKLEARN_TESTBED_PIP_VERSION = "21.3.1"
SWE_BENCH_SYMPY_HIDDEN_BASE_COMMIT = "cffd4e0f86fefd4802349a9f9b19ed70934ea354"
SWE_BENCH_REPO_SETUP_COMPATIBILITY_PATCHES = {
    ("sympy/sympy", SWE_BENCH_SYMPY_HIDDEN_BASE_COMMIT): {
        "evaluator": SWE_BENCH_PACKAGE,
        "fetch_ref": SWE_BENCH_SYMPY_HIDDEN_BASE_COMMIT,
        "reason": "SWE-bench maps this hidden base commit to deleted upstream branch 1.7.",
    },
}
POLY_BENCH_PYTHON = REPO_ROOT / ".cache" / "polybench-eval-venv" / "bin" / "python"
POLY_BENCH_REPO = "https://github.com/amazon-science/SWE-PolyBench.git"
POLY_BENCH_COMMIT = "1963184d8b6cc7120f195555c642b754a3d30840"
POLY_BENCH_CONSTRAINTS = REPO_ROOT / "contextbench" / "run_suites_constraints" / "polybench.txt"
POLY_BENCH_PIP_VERSION = "26.0.1"
MULTI_BENCH_PYTHON = REPO_ROOT / ".cache" / "multibench-eval-venv" / "bin" / "python"
MULTI_BENCH_REPO = "https://github.com/multi-swe-bench/multi-swe-bench.git"
MULTI_BENCH_COMMIT = "24f493f8a103e72312ded4f6b9c89f081d69cb09"
MULTI_BENCH_CONSTRAINTS = REPO_ROOT / "contextbench" / "run_suites_constraints" / "multibench.txt"
MULTI_BENCH_PIP_VERSION = "26.0.1"
PRO_BENCH_ROOT = REPO_ROOT / ".cache" / "probench-eval"
PRO_BENCH_PYTHON = REPO_ROOT / ".cache" / "probench-eval-venv" / "bin" / "python"
PRO_BENCH_REPO = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
PRO_BENCH_COMMIT = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
PRO_BENCH_CONSTRAINTS = REPO_ROOT / "contextbench" / "run_suites_constraints" / "probench.txt"
PRO_BENCH_PIP_VERSION = "26.0.1"
CODEX_CLI_VERSION = "0.122.0"
CODEX_RUNTIME_IMAGE = DEFAULT_CODEX_RUNTIME_IMAGE
POSTPROCESS_IMAGE = DEFAULT_POSTPROCESS_RUNTIME_IMAGE
POSTPROCESS_DOCKERFILE = REPO_ROOT / "docker" / "postprocess" / "Dockerfile"
CODEX_RUNTIME_DOCKERFILE = REPO_ROOT / "docker" / "codex-runtime.Dockerfile"


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _docker_build(*, image: str, dockerfile: Path, force: bool = False, build_args: dict[str, str] | None = None) -> None:
    command = ["docker", "build"]
    if force:
        command.append("--no-cache")
    for key, value in sorted((build_args or {}).items()):
        command.extend(["--build-arg", f"{key}={value}"])
    command.extend(["-f", str(dockerfile), "-t", image, str(REPO_ROOT)])
    _run(command)


def _verify_fully_pinned_constraints(path: Path) -> None:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line and " @ " not in line:
            raise RuntimeError(f"Unpinned constraint in {path}:{line_number}: {line}")


def _uv_executable() -> str | None:
    return shutil.which("uv")


def _ensure_uv() -> str | None:
    uv = _uv_executable()
    if uv is None:
        print("ERROR: uv is required to set up repo-local evaluator environments. Install uv and retry.", file=sys.stderr)
        return None
    return uv


def _uv_venv(*, uv: str, venv_dir: Path) -> None:
    _run([uv, "venv", "--python", "3.11", str(venv_dir)])


def _uv_pip_install(*, uv: str, python: Path, packages: list[str]) -> None:
    _run([uv, "pip", "install", "--python", str(python), *packages])


def _python_site_packages(python: Path) -> Path:
    return python.parents[1] / "lib" / "python3.11" / "site-packages"


def _repair_swebench_scikit_learn_testbed_pip(python: Path) -> None:
    """Make SWE-bench's implicit legacy pip dependency explicit for scikit-learn."""
    constants_path = _python_site_packages(python) / "swebench" / "harness" / "constants" / "python.py"
    if not constants_path.exists():
        raise RuntimeError(f"SWE-bench Python constants file not found: {constants_path}")

    marker = "# Fork note: ContextBench pins scikit-learn testbed pip for legacy --no-use-pep517 support."
    text = constants_path.read_text(encoding="utf-8")
    if marker in text:
        return
    needle = "SPECS_FLASK = {\n"
    if needle not in text:
        raise RuntimeError(f"Unable to locate scikit-learn specs insertion point in {constants_path}")

    block = f"""
{marker}
for _contextbench_sklearn_spec in SPECS_SKLEARN.values():
    _contextbench_pip_packages = _contextbench_sklearn_spec.setdefault("pip_packages", [])
    if "pip=={SWE_BENCH_SKLEARN_TESTBED_PIP_VERSION}" not in _contextbench_pip_packages:
        _contextbench_pip_packages.insert(0, "pip=={SWE_BENCH_SKLEARN_TESTBED_PIP_VERSION}")
del _contextbench_sklearn_spec
del _contextbench_pip_packages
"""
    constants_path.write_text(text.replace(needle, f"{block}\n{needle}", 1), encoding="utf-8")


def _repair_swebench_repo_setup_compatibility_patches(python: Path) -> None:
    """Apply explicit evaluator compatibility patches that preserve official base commits."""
    test_spec_path = _python_site_packages(python) / "swebench" / "harness" / "test_spec" / "python.py"
    if not test_spec_path.exists():
        raise RuntimeError(f"SWE-bench Python test spec file not found: {test_spec_path}")

    marker = "# Fork note: ContextBench applies explicit SWE-bench repo setup compatibility patches."
    legacy_marker = "# Fork note: ContextBench fetches hidden SymPy 1.7 base commit by exact SHA."
    text = test_spec_path.read_text(encoding="utf-8")
    if marker in text:
        return
    if legacy_marker in text:
        test_spec_path.write_text(text.replace(legacy_marker, marker, 1), encoding="utf-8")
        return

    patch_entries = {
        key: value["fetch_ref"]
        for key, value in SWE_BENCH_REPO_SETUP_COMPATIBILITY_PATCHES.items()
        if value["evaluator"] == SWE_BENCH_PACKAGE
    }
    if not patch_entries:
        return

    needle = """    branch = REPO_BASE_COMMIT_BRANCH.get(repo, {}).get(base_commit, "")
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
"""
    replacement = f"""    {marker}
    contextbench_repo_setup_compatibility_patches = {patch_entries!r}
    contextbench_fetch_ref = contextbench_repo_setup_compatibility_patches.get((repo, base_commit))
    if contextbench_fetch_ref:
        setup_commands = [
            f"git init {{repo_directory}}",
            f"chmod -R 777 {{repo_directory}}",  # So nonroot user can run tests
            f"cd {{repo_directory}}",
            f"git remote add origin https://github.com/{{repo}}",
            f"git fetch origin {{contextbench_fetch_ref}}",
            "git checkout --detach FETCH_HEAD",
            f"git reset --hard {{base_commit}}",
        ]
    else:
        branch = REPO_BASE_COMMIT_BRANCH.get(repo, {{}}).get(base_commit, "")
        branch = f"--branch {{branch}}" if branch else ""
        setup_commands = [
            f"git clone -o origin {{branch}} --single-branch https://github.com/{{repo}} {{repo_directory}}",
            f"chmod -R 777 {{repo_directory}}",  # So nonroot user can run tests
            f"cd {{repo_directory}}",
            f"git reset --hard {{base_commit}}",
        ]
    setup_commands.extend([
        # Remove the remote and tags so the agent won't see newer commits.
        "git remote remove origin",
        # Remove only tags pointing to commits after target timestamp
        f"TARGET_TIMESTAMP=$(git show -s --format=%ci {{base_commit}})",
        'git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done',
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive",
        # Verify future logs aren't available
        "AFTER_TIMESTAMP=$(date -d \\"$TARGET_TIMESTAMP + 1 second\\" '+%Y-%m-%d %H:%M:%S')",
        'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)',
        '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {{env_name}}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ])
"""
    if needle not in text:
        raise RuntimeError(f"Unable to locate SWE-bench repo setup compatibility patch insertion point in {test_spec_path}")
    test_spec_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def _repair_multibench_packaging_case_paths(python: Path) -> None:
    site_packages = python.parents[1] / "lib" / "python3.11" / "site-packages"
    python_repos = site_packages / "multi_swe_bench" / "harness" / "repos" / "python"
    qiskit_dir = python_repos / "Qiskit"
    lowercase_qiskit_dir = python_repos / "qiskit"
    if qiskit_dir.exists() and not lowercase_qiskit_dir.exists():
        try:
            lowercase_qiskit_dir.symlink_to(qiskit_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(qiskit_dir, lowercase_qiskit_dir)
    qiskit_init = qiskit_dir / "__init__.py"
    if qiskit_init.exists():
        qiskit_init.write_text(
            "from multi_swe_bench.harness.repos.python.Qiskit.qiskit import *\n",
            encoding="utf-8",
        )
    python_init = python_repos / "__init__.py"
    if python_init.exists():
        content = python_init.read_text(encoding="utf-8")
        content = content.replace(
            "from multi_swe_bench.harness.repos.python.qiskit import *",
            "from multi_swe_bench.harness.repos.python.Qiskit import *",
        )
        python_init.write_text(content, encoding="utf-8")


def setup_swebench(*, force: bool = False) -> int:
    venv_dir = SWE_BENCH_PYTHON.parents[1]
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)

    uv = _ensure_uv()
    if uv is None:
        return 1

    if not SWE_BENCH_CONSTRAINTS.exists():
        print(f"ERROR: SWE-bench constraints file not found: {SWE_BENCH_CONSTRAINTS}", file=sys.stderr)
        return 1
    _verify_fully_pinned_constraints(SWE_BENCH_CONSTRAINTS)

    if not SWE_BENCH_PYTHON.exists():
        _uv_venv(uv=uv, venv_dir=venv_dir)

    _uv_pip_install(uv=uv, python=SWE_BENCH_PYTHON, packages=[f"pip=={SWE_BENCH_PIP_VERSION}"])
    _uv_pip_install(
        uv=uv,
        python=SWE_BENCH_PYTHON,
        packages=[SWE_BENCH_PACKAGE, "-c", str(SWE_BENCH_CONSTRAINTS)],
    )
    _repair_swebench_scikit_learn_testbed_pip(SWE_BENCH_PYTHON)
    _repair_swebench_repo_setup_compatibility_patches(SWE_BENCH_PYTHON)
    _run([str(SWE_BENCH_PYTHON), "-m", "swebench.harness.run_evaluation", "--help"])

    print(f"SWE-bench evaluator ready: {SWE_BENCH_PYTHON}")
    return 0


def setup_polybench(*, force: bool = False) -> int:
    venv_dir = POLY_BENCH_PYTHON.parents[1]
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)

    uv = _ensure_uv()
    if uv is None:
        return 1

    if not POLY_BENCH_CONSTRAINTS.exists():
        print(f"ERROR: SWE-PolyBench constraints file not found: {POLY_BENCH_CONSTRAINTS}", file=sys.stderr)
        return 1
    _verify_fully_pinned_constraints(POLY_BENCH_CONSTRAINTS)

    if not POLY_BENCH_PYTHON.exists():
        _uv_venv(uv=uv, venv_dir=venv_dir)

    package_spec = f"git+{POLY_BENCH_REPO}@{POLY_BENCH_COMMIT}"
    _uv_pip_install(uv=uv, python=POLY_BENCH_PYTHON, packages=[f"pip=={POLY_BENCH_PIP_VERSION}"])
    _uv_pip_install(uv=uv, python=POLY_BENCH_PYTHON, packages=[package_spec, "-c", str(POLY_BENCH_CONSTRAINTS)])
    _run([str(POLY_BENCH_PYTHON), "-m", "poly_bench_evaluation.run_evaluation", "--help"])

    print(f"SWE-PolyBench evaluator ready: {POLY_BENCH_PYTHON}")
    return 0


def setup_multibench(*, force: bool = False) -> int:
    venv_dir = MULTI_BENCH_PYTHON.parents[1]
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)

    uv = _ensure_uv()
    if uv is None:
        return 1

    if not MULTI_BENCH_CONSTRAINTS.exists():
        print(f"ERROR: Multi-SWE-Bench constraints file not found: {MULTI_BENCH_CONSTRAINTS}", file=sys.stderr)
        return 1
    _verify_fully_pinned_constraints(MULTI_BENCH_CONSTRAINTS)

    if not MULTI_BENCH_PYTHON.exists():
        _uv_venv(uv=uv, venv_dir=venv_dir)

    package_spec = f"git+{MULTI_BENCH_REPO}@{MULTI_BENCH_COMMIT}"
    _uv_pip_install(uv=uv, python=MULTI_BENCH_PYTHON, packages=[f"pip=={MULTI_BENCH_PIP_VERSION}"])
    _uv_pip_install(uv=uv, python=MULTI_BENCH_PYTHON, packages=[package_spec, "-c", str(MULTI_BENCH_CONSTRAINTS)])
    _repair_multibench_packaging_case_paths(MULTI_BENCH_PYTHON)
    _run(
        [
            str(MULTI_BENCH_PYTHON),
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('multi_swe_bench') else 1)",
        ]
    )

    print(f"Multi-SWE-Bench evaluator ready: {MULTI_BENCH_PYTHON}")
    return 0


def setup_probench(*, force: bool = False) -> int:
    venv_dir = PRO_BENCH_PYTHON.parents[1]
    if force:
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        if PRO_BENCH_ROOT.exists():
            shutil.rmtree(PRO_BENCH_ROOT)

    uv = _ensure_uv()
    if uv is None:
        return 1

    if not PRO_BENCH_ROOT.exists():
        _run(["git", "clone", "--no-checkout", PRO_BENCH_REPO, str(PRO_BENCH_ROOT)])
    _run(["git", "-C", str(PRO_BENCH_ROOT), "fetch", "--depth", "1", "origin", PRO_BENCH_COMMIT])
    _run(["git", "-C", str(PRO_BENCH_ROOT), "checkout", "--detach", PRO_BENCH_COMMIT])

    if not PRO_BENCH_PYTHON.exists():
        _uv_venv(uv=uv, venv_dir=venv_dir)

    requirements = PRO_BENCH_ROOT / "requirements.txt"
    if not requirements.exists():
        print(f"ERROR: SWE-bench Pro requirements file not found: {requirements}", file=sys.stderr)
        return 1
    if not PRO_BENCH_CONSTRAINTS.exists():
        print(f"ERROR: SWE-bench Pro constraints file not found: {PRO_BENCH_CONSTRAINTS}", file=sys.stderr)
        return 1
    _verify_fully_pinned_constraints(PRO_BENCH_CONSTRAINTS)
    _uv_pip_install(uv=uv, python=PRO_BENCH_PYTHON, packages=[f"pip=={PRO_BENCH_PIP_VERSION}"])
    _uv_pip_install(
        uv=uv,
        python=PRO_BENCH_PYTHON,
        packages=["-r", str(requirements), "-c", str(PRO_BENCH_CONSTRAINTS)],
    )
    _run([str(PRO_BENCH_PYTHON), str(PRO_BENCH_ROOT / "swe_bench_pro_eval.py"), "--help"])

    print(f"SWE-bench Pro evaluator ready: {PRO_BENCH_ROOT} ({PRO_BENCH_PYTHON})")
    return 0


def setup_resolution_envs(*, force: bool = False) -> int:
    for setup in (setup_swebench, setup_polybench, setup_multibench, setup_probench):
        result = setup(force=force)
        if result != 0:
            return result
    return 0


def setup_postprocess_image(*, force: bool = False) -> int:
    _docker_build(image=POSTPROCESS_IMAGE, dockerfile=POSTPROCESS_DOCKERFILE, force=force)
    supports_evaluation, detail = _postprocess_image_supports_evaluation(POSTPROCESS_IMAGE)
    if not supports_evaluation:
        print(f"ERROR: Postprocess image is missing required evaluation parsers: {detail}", file=sys.stderr)
        return 1
    print(f"Postprocess image ready: {POSTPROCESS_IMAGE}")
    return 0


def setup_codex_runtime_image(*, force: bool = False) -> int:
    _docker_build(
        image=CODEX_RUNTIME_IMAGE,
        dockerfile=CODEX_RUNTIME_DOCKERFILE,
        force=force,
        build_args={"CODEX_CLI_VERSION": CODEX_CLI_VERSION},
    )
    print(f"Codex runtime image ready: {CODEX_RUNTIME_IMAGE}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up deterministic run-suite backend dependencies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    swebench = subparsers.add_parser("swebench", help="Set up the official SWE-bench evaluator")
    swebench.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the repo-local SWE-bench evaluator venv",
    )
    polybench = subparsers.add_parser("polybench", help="Set up the official SWE-PolyBench evaluator")
    polybench.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the repo-local SWE-PolyBench evaluator venv",
    )
    multibench = subparsers.add_parser("multibench", help="Set up the official Multi-SWE-Bench evaluator")
    multibench.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the repo-local Multi-SWE-Bench evaluator venv",
    )
    probench = subparsers.add_parser("probench", help="Set up the SWE-bench Pro evaluator")
    probench.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the repo-local SWE-bench Pro evaluator checkout and venv",
    )
    resolution_envs = subparsers.add_parser("resolution-envs", help="Set up all repo-local resolution evaluator environments")
    resolution_envs.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate repo-local resolution evaluator environments",
    )
    postprocess_image = subparsers.add_parser("postprocess-image", help="Build the Docker image for conversion and retrieval evaluation")
    postprocess_image.add_argument("--force", action="store_true", help="Build without Docker layer cache")
    codex_runtime_image = subparsers.add_parser("codex-runtime-image", help="Build the Docker image for Codex task runs")
    codex_runtime_image.add_argument("--force", action="store_true", help="Build without Docker layer cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "swebench":
        return setup_swebench(force=bool(args.force))
    if args.command == "polybench":
        return setup_polybench(force=bool(args.force))
    if args.command == "multibench":
        return setup_multibench(force=bool(args.force))
    if args.command == "probench":
        return setup_probench(force=bool(args.force))
    if args.command == "resolution-envs":
        return setup_resolution_envs(force=bool(args.force))
    if args.command == "postprocess-image":
        return setup_postprocess_image(force=bool(args.force))
    if args.command == "codex-runtime-image":
        return setup_codex_runtime_image(force=bool(args.force))
    print(f"ERROR: unsupported setup command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
