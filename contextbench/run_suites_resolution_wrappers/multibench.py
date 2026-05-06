
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Multi-SWE-Bench evaluation from the repo-local host evaluator.")
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSONL rows must be objects: {path}")
        rows.append(payload)
    return rows


def _instance_id(row: dict[str, object]) -> str:
    org = str(row.get("org") or "").strip()
    repo = str(row.get("repo") or "").strip()
    number = row.get("number")
    if not org or not repo or number in (None, ""):
        raise RuntimeError(f"Multi-SWE-Bench row is missing org/repo/number: {row}")
    return f"{org}__{repo}-{int(number)}"


def _official_instance_id(row: dict[str, object]) -> str:
    org = str(row.get("org") or "").strip()
    repo = str(row.get("repo") or "").strip()
    number = row.get("number")
    if not org or not repo or number in (None, ""):
        raise RuntimeError(f"Multi-SWE-Bench row is missing org/repo/number: {row}")
    return f"{org}/{repo}:pr-{int(number)}"


def _write_error(*, output_dir: Path, instance_id: str, exit_code: int | None, detail: str) -> Path:
    error_path = output_dir / "evaluation-error.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "exit_code": exit_code,
                "detail": detail,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return error_path


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions_path.resolve()
    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    repo_dir = args.repo_dir.resolve()
    log_dir = args.log_dir.resolve()

    prediction_rows = _load_jsonl(predictions_path)
    if len(prediction_rows) != 1:
        raise RuntimeError("The Multi-SWE-Bench wrapper expects exactly one prediction per invocation.")
    instance_id = _instance_id(prediction_rows[0])
    official_instance_id = _official_instance_id(prediction_rows[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir.parent / "multibench-config.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "evaluation",
                "workdir": str(work_dir),
                "patch_files": [str(predictions_path)],
                "dataset_files": [str(dataset_path)],
                "output_dir": str(output_dir),
                "repo_dir": str(repo_dir),
                "log_dir": str(log_dir),
                "specifics": [official_instance_id],
                "skips": [],
                "force_build": False,
                "need_clone": True,
                "global_env": [],
                "clear_env": True,
                "stop_on_error": False,
                "max_workers": max(1, int(args.max_workers)),
                "max_workers_build_image": 1,
                "max_workers_run_instance": 1,
                "fix_patch_run_cmd": "",
                "log_level": "INFO",
                "log_to_console": True,
                "human_mode": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "multi_swe_bench.harness.run_evaluation",
        "--config",
        str(config_path),
    ]
    final_report = output_dir / "final_report.json"
    error_report = output_dir / "evaluation-error.json"
    for stale_path in (final_report, error_report):
        if stale_path.exists():
            stale_path.unlink()

    print(f"[multibench-wrapper] evaluating instance: {instance_id}", flush=True)
    completed = subprocess.run(command, check=False, cwd=str(output_dir.parent))
    if completed.returncode != 0:
        detail = (
            f"Multi-SWE-Bench evaluator exited {completed.returncode} for {instance_id}; "
            "not recording this as unresolved."
        )
        error_path = _write_error(output_dir=output_dir, instance_id=instance_id, exit_code=completed.returncode, detail=detail)
        print(f"[multibench-wrapper] {detail}; proof={error_path}", flush=True)
        return completed.returncode
    if not final_report.exists():
        detail = (
            f"Multi-SWE-Bench evaluator produced no final_report.json for {instance_id}; "
            "not recording this as unresolved."
        )
        error_path = _write_error(output_dir=output_dir, instance_id=instance_id, exit_code=None, detail=detail)
        print(f"[multibench-wrapper] {detail}; proof={error_path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
