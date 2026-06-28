# SPDX-License-Identifier: Apache-2.0

"""Run a tiny Codex v1 vs Codex OTEL v2 adapter comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from contextbench.agents.codex.adapter import CodexAdapter
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.codex_otel_v2.adapter import CodexOtelV2Adapter
from contextbench.agents.codex_otel_v2.runtime import runtime_root as codex_otel_v2_runtime_root


TASK_PROMPT = (
    "Run these exact shell reads before answering: `sed -n '1,120p' README.md` and "
    "`sed -n '1,120p' src/app.py`. Do not edit files. Keep final_answer to one short sentence."
)


def _write_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "src").mkdir(parents=True)
    (workspace / "README.md").write_text(
        "# OTEL smoke\n\nThis tiny repository exists to compare Codex trace capture.\n",
        encoding="utf-8",
    )
    (workspace / "src" / "app.py").write_text(
        "\n".join(
            [
                "def add(left: int, right: int) -> int:",
                "    return left + right",
                "",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _load_json(path: Path | None) -> object:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_summary(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    if raw.get("response_format") == "jsonl-events":
        events = [item for item in raw.get("events", []) if isinstance(item, dict)]
        return {
            "response_format": raw.get("response_format"),
            "event_count": len(events),
            "event_types": sorted({str(item.get("type") or "") for item in events if item.get("type")}),
            "has_final_message": "final_message" in raw,
        }
    otel = raw.get("otel")
    if isinstance(otel, dict):
        logs = [item for item in otel.get("logs", []) if isinstance(item, dict)]
        traces = [item for item in otel.get("traces", []) if isinstance(item, dict)]
        metrics = [item for item in otel.get("metrics", []) if isinstance(item, dict)]
        return {
            "response_format": raw.get("response_format"),
            "otel_request_count": otel.get("request_count"),
            "log_count": len(logs),
            "trace_count": len(traces),
            "metric_count": len(metrics),
            "log_names": sorted({str(item.get("name") or "") for item in logs if item.get("name")}),
            "trace_names": sorted({str(item.get("name") or "") for item in traces if item.get("name")}),
            "metric_names": sorted({str(item.get("name") or "") for item in metrics if item.get("name")}),
            "has_final_message": "final_message" in raw,
        }
    return {"response_format": raw.get("response_format")}


def _run_one(
    *,
    adapter,
    task_dir: Path,
    workspace: Path,
    model: str | None,
    reasoning_effort: str | None,
    timeout: int,
) -> dict[str, object]:
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    started = time.time()
    prepared = adapter.prepare_runtime(
        task_dir=task_dir,
        setup={},
        env_overrides={
            "CONTEXTBENCH_WORKSPACE_PATH": str(workspace),
            "CONTEXTBENCH_TASK_DIR": str(task_dir),
        },
        runtime_backend="host",
        runtime_env={},
    )
    result = adapter.run_main_invocation(
        task_dir=task_dir,
        workspace_path=workspace,
        prompt=adapter.build_prompt({"repo": "otel-smoke", "prompt": TASK_PROMPT}),
        timeout=timeout,
        model=model,
        reasoning_effort=reasoning_effort,
        extra_args=(),
        schema_path=adapter.output_schema_path,
        prepared_runtime=prepared,
    )
    raw = _load_json(result.raw_response_path)
    return {
        "agent": adapter.name,
        "ok": result.command_result["ok"],
        "exit_code": result.command_result["exit_code"],
        "timeout": result.command_result["timeout"],
        "duration_ms": int((time.time() - started) * 1000),
        "structured_output_present": result.structured_output is not None,
        "structured_output": result.structured_output,
        "token_usage": result.token_usage,
        "tool_call_count": len(result.tool_calls),
        "tool_call_names": [item.get("tool_name") for item in result.tool_calls],
        "available_tools": result.available_tools,
        "diagnostic_note": result.diagnostic_note,
        "raw_response_path": str(result.raw_response_path),
        "schema": str(adapter.output_schema_path),
        "raw_summary": _raw_summary(raw),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/private/tmp/contextbench-codex-otel-v2-smoke")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    workspace = output_root / "workspace"
    _write_workspace(workspace)
    results = []
    try:
        results.append(
            _run_one(
                adapter=CodexAdapter(),
                task_dir=output_root / "v1",
                workspace=workspace,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
        )
        results.append(
            _run_one(
                adapter=CodexOtelV2Adapter(),
                task_dir=output_root / "v2",
                workspace=workspace,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
        )
    finally:
        shutil.rmtree(codex_runtime_root(output_root / "v1"), ignore_errors=True)
        shutil.rmtree(codex_otel_v2_runtime_root(output_root / "v2"), ignore_errors=True)

    comparison = {
        "output_root": str(output_root),
        "workspace": str(workspace),
        "results": results,
    }
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
