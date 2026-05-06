#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bench_sources import (
    LANG_DIR_MAP,
    load_task_ids,
    fetch_records,
    resolve_language,
    map_language_dir,
    to_agentless_raw_json,
)


def run_instance(
    cagentless_dir: Path,
    bench_name: str,
    idx: int,
    instance_id: str,
    language: str,
):
    run_script = cagentless_dir / "script" / "run_single_instance.sh"
    if not run_script.exists():
        raise FileNotFoundError(f"Missing script: {run_script}")

    env = os.environ.copy()
    env["BENCH_NAME"] = bench_name
    env["TARGET_ID"] = instance_id
    env["SWEBENCH_LANG"] = language
    env["FOLDER_NAME"] = f"{bench_name}/{idx}_{instance_id}"
    env.setdefault("DATASET", "local_json")
    env.setdefault("SPLIT", "test")
    env.setdefault("DATA_ROOT", str(cagentless_dir / "results" / bench_name / f"{idx}_{instance_id}" / "input_data"))

    subprocess.run(
        ["bash", str(run_script)],
        cwd=str(cagentless_dir),
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bench_name", choices=["Multi", "Poly", "Pro", "Verified"], type=str
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Run only this instance ID (original_inst_id or instance_id from dataset)",
    )
    args = parser.parse_args()

    cagentless_dir = Path(__file__).resolve().parent
    root_dir = cagentless_dir.parent
    if args.instance:
        tasks = [args.instance]
    else:
        tasks = load_task_ids(args.bench_name, args.limit)
    if not tasks:
        print("No tasks found in CSV.", file=sys.stderr)
        sys.exit(1)

    records = fetch_records(args.bench_name, tasks)
    for idx, instance_id in enumerate(tasks):
        record = records.get(instance_id)
        if record is None:
            print(f"Missing record for {instance_id}", file=sys.stderr)
            continue
        language = resolve_language(record, args.bench_name)
        if not language:
            print(f"Unable to infer language for {instance_id}", file=sys.stderr)
            continue

        lang_dir = map_language_dir(language)
        input_root = (
            cagentless_dir
            / "results"
            / args.bench_name
            / f"{idx}_{instance_id}"
            / "input_data"
            / lang_dir
        )
        input_root.mkdir(parents=True, exist_ok=True)
        raw_json = to_agentless_raw_json(record, instance_id)
        input_file = input_root / "one.jsonl"
        input_file.write_text(f"{json.dumps(raw_json)}\n", encoding="utf-8")

        run_instance(
            cagentless_dir=cagentless_dir,
            bench_name=args.bench_name,
            idx=idx,
            instance_id=instance_id,
            language=language,
        )


if __name__ == "__main__":
    main()
