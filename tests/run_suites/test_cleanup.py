
from __future__ import annotations

import subprocess
from pathlib import Path

from contextbench.run_suites_cleanup import stale_resolution_dirs, stopped_contextbench_containers


def test_stale_resolution_dirs_keeps_latest_per_variant_bench(tmp_path: Path) -> None:
    bench_root = tmp_path / "suite" / "variants" / "baseline" / "resolution-eval" / "verified"
    old = bench_root / "run-old"
    new = bench_root / "run-new"
    old.mkdir(parents=True)
    new.mkdir()
    old.touch()
    new.touch()

    stale = stale_resolution_dirs(tmp_path / "suite", keep_latest=1)

    assert stale == [old]


def test_stopped_contextbench_containers_filters_running_and_foreign(monkeypatch) -> None:
    monkeypatch.setattr(
        "contextbench.run_suites_cleanup.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "contextbench-created Created\n"
                "contextbench-exited Exited (1) 1 hour ago\n"
                "contextbench-running Up 1 hour\n"
                "container_polybench_python_task-a Exited (1) 1 hour ago\n"
                "container_polybench_python_task-b Up 1 hour\n"
                "other Exited (1)\n"
            ),
            stderr="",
        ),
    )

    assert stopped_contextbench_containers() == [
        "contextbench-created",
        "contextbench-exited",
        "container_polybench_python_task-a",
    ]
