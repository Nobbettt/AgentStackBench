
"""Extract unified trajectories from Codex wrapper record files."""

from __future__ import annotations

import json
from pathlib import Path

from .parser import CodexAgentParser

PARSER = CodexAgentParser()


def extract_trajectory(record_file: str) -> dict[str, object]:
    path = Path(record_file)
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    return PARSER.extract_trajectory(record)
