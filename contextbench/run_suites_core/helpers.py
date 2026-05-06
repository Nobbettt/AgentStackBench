
"""Pure helpers for run suite orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..agents.registry import get_coding_agent_adapter, has_coding_agent_adapter
from ..coding_agents.files import read_json, safe_path_component


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        raise TypeError("Expected a string or list of strings")
    normalized = [item for item in items if item]
    return normalized or None


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def task_key(task: dict[str, object]) -> str:
    return str(task.get("instance_id") or task.get("original_inst_id") or "").strip()


def task_record_path(*, raw_root: Path, agent: str, task: dict[str, object]) -> Path:
    task_id = safe_path_component(task_key(task) or "task")
    bench = str(task.get("bench") or "Verified")
    suffix = get_coding_agent_adapter(agent).record_suffix if has_coding_agent_adapter(agent) else safe_path_component(agent)
    return raw_root / agent / bench / task_id / f"{task_id}.{suffix}-record.json"


def record_is_resume_complete(record_path: Path) -> bool:
    if not record_path.exists():
        return False
    try:
        record = read_json(record_path)
    except Exception:
        return False
    if not isinstance(record, dict):
        return False
    if record.get("timeout"):
        return False
    if "ok" in record and not bool(record.get("ok")):
        return False
    if record.get("status") != "completed":
        return False
    return isinstance(record.get("final_output"), dict)


def flatten_metrics(metrics: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}_{child_key}"] = child_value
        else:
            flat[key] = value
    return flat
