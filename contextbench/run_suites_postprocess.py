# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-02 from upstream ContextBench.
# Summary of changes: add parallel-safe evaluation workspaces and atomic evaluation summaries.

"""CLI wrappers for run-suite postprocess stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .coding_agents.files import ensure_dir, write_json
from .run_suites_core.postprocess import convert_records_to_jsonl, evaluate_prediction_file


def _write_json_atomic(path: Path, value: object) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ContextBench postprocess stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert run records to pred.jsonl")
    convert.add_argument("--source-dir", type=Path, required=True)
    convert.add_argument("--expected-agent", required=True)
    convert.add_argument("--out-path", type=Path, required=True)
    convert.add_argument("--summary-path", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate pred.jsonl against gold data")
    evaluate.add_argument("--gold-path", type=Path, required=True)
    evaluate.add_argument("--pred-path", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--out-path", type=Path, required=True)
    evaluate.add_argument("--summary-path", type=Path, required=True)
    evaluate.add_argument("--selected-task-count", type=int, required=False, default=None)
    evaluate.add_argument("--workspace-key", required=False, default=None)
    evaluate.add_argument("--tmp-root", type=Path, required=False, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "convert":
        summary = convert_records_to_jsonl(
            source_dir=args.source_dir.resolve(),
            expected_agent=str(args.expected_agent),
            out_path=args.out_path.resolve(),
        )
        write_json(args.summary_path.resolve(), summary)
        if summary.get("is_partial") or int(summary.get("input_error_count") or 0) > 0:
            return 1
        return 0

    if args.command == "evaluate":
        summary = evaluate_prediction_file(
            gold_path=args.gold_path.resolve(),
            pred_path=args.pred_path.resolve(),
            cache_dir=args.cache_dir.resolve(),
            out_path=args.out_path.resolve(),
            selected_task_count=args.selected_task_count,
            workspace_key=args.workspace_key,
            tmp_root=args.tmp_root.resolve() if args.tmp_root is not None else None,
        )
        _write_json_atomic(args.summary_path.resolve(), summary)
        if summary.get("is_partial") or summary.get("has_errors") or summary.get("error_counts"):
            return 1
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
