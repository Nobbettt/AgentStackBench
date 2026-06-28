# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextbench.agents.claude.adapter import ClaudeAdapter
from contextbench.agents.claude.runtime import runtime_root as claude_runtime_root
from contextbench.agents.claude_otel.adapter import ClaudeOtelAdapter
from contextbench.agents.codex.adapter import CodexAdapter
from contextbench.agents.codex.runtime import runtime_root as codex_runtime_root
from contextbench.agents.codex_otel_v2.adapter import CodexOtelV2Adapter
from contextbench.agents.codex_otel_v2.runtime import runtime_root as codex_otel_v2_runtime_root


def test_runtime_root_lifecycle_is_adapter_owned(tmp_path) -> None:
    adapters = [
        (CodexAdapter(), codex_runtime_root),
        (CodexOtelV2Adapter(), codex_otel_v2_runtime_root),
        (ClaudeAdapter(), claude_runtime_root),
        (ClaudeOtelAdapter(), claude_runtime_root),
    ]

    for adapter, runtime_root in adapters:
        task_dir = tmp_path / adapter.name
        root = runtime_root(task_dir)
        root.mkdir(parents=True)
        stale_file = root / "stale-secret.txt"
        stale_file.write_text("remove me", encoding="utf-8")

        mounts = adapter.prepare_runtime_writable_mounts(task_dir=task_dir)

        assert mounts == (root,)
        assert root.is_dir()
        assert not stale_file.exists()

        secret_file = root / "auth.json"
        secret_file.write_text("secret", encoding="utf-8")

        adapter.scrub_runtime_secrets(task_dir=task_dir)

        assert not root.exists()
