# SPDX-License-Identifier: Apache-2.0

"""Shared adapter lifecycle support for isolated agent runtime roots."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..coding_agents.files import ensure_dir


class RuntimeRootLifecycleMixin:
    """Adapter mixin for agents that use one per-task runtime root."""

    def runtime_root(self, task_dir: Path) -> Path:
        raise NotImplementedError

    def prepare_runtime_writable_mounts(self, *, task_dir: Path) -> tuple[Path, ...]:
        root = self.runtime_root(task_dir)
        shutil.rmtree(root, ignore_errors=True)
        ensure_dir(root)
        return (root,)

    def scrub_runtime_secrets(self, *, task_dir: Path) -> None:
        shutil.rmtree(self.runtime_root(task_dir), ignore_errors=True)
