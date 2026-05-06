
from __future__ import annotations

import json
from pathlib import Path

def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _record(
    task_dir: Path,
    duration_ms: int,
    total_tokens: int,
    tool_calls: int,
    *,
    status: str = "completed",
    ok: bool = True,
    model_patch: str = "",
) -> str:
    record_path = task_dir / f"{task_dir.name}.codex-record.json"
    _write(
        record_path,
        json.dumps(
            {
                "status": status,
                "ok": ok,
                "duration_ms": duration_ms,
                "token_usage": {"total_tokens": total_tokens},
                "tool_calls": [{} for _ in range(tool_calls)],
                "model_patch": model_patch,
            }
        ),
    )
    return str(record_path)
