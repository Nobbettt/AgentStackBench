
"""Cleanup helpers for stale run-suite resolution artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


STALE_CONTAINER_PREFIXES = ("contextbench-", "container_polybench_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean stale run-suite resolution artifacts")
    parser.add_argument("--suite-dir", type=Path, required=True, help="Run-suite directory under results/run_suites")
    parser.add_argument("--keep-latest", type=int, default=1, help="Number of newest resolution workdirs to keep per variant/bench")
    parser.add_argument("--apply", action="store_true", help="Delete artifacts instead of printing what would be removed")
    parser.add_argument("--docker", action="store_true", help="Also remove stopped Docker containers from run-suite evaluators")
    return parser.parse_args(argv)


def stale_resolution_dirs(suite_dir: Path, *, keep_latest: int) -> list[Path]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for bench_root in sorted(suite_dir.glob("variants/*/resolution-eval/*")):
        if not bench_root.is_dir():
            continue
        for child in bench_root.iterdir():
            if child.is_dir():
                groups[bench_root].append(child)

    stale: list[Path] = []
    keep = max(0, keep_latest)
    for dirs in groups.values():
        ordered = sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)
        stale.extend(ordered[keep:])
    return sorted(stale)


def stopped_contextbench_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}} {{.Status}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "docker ps failed").strip())

    names: list[str] = []
    for line in result.stdout.splitlines():
        name, _, status = line.partition(" ")
        if not name.startswith(STALE_CONTAINER_PREFIXES):
            continue
        if status.lower().startswith(("created", "exited", "dead")):
            names.append(name)
    return names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite_dir = args.suite_dir.resolve()
    if not suite_dir.exists():
        print(f"ERROR: suite directory not found: {suite_dir}", file=sys.stderr)
        return 2
    if args.keep_latest < 0:
        print("ERROR: --keep-latest must be >= 0", file=sys.stderr)
        return 2

    paths = stale_resolution_dirs(suite_dir, keep_latest=args.keep_latest)
    for path in paths:
        print(("remove " if args.apply else "would remove ") + str(path))
        if args.apply:
            shutil.rmtree(path)

    if args.docker:
        containers = stopped_contextbench_containers()
        for name in containers:
            print(("docker rm " if args.apply else "would docker rm ") + name)
        if args.apply and containers:
            result = subprocess.run(["docker", "rm", "--force", *containers], check=False)
            if result.returncode != 0:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
