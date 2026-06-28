# SPDX-License-Identifier: Apache-2.0

"""Shared Codex-family Docker tool-bundle support."""

from __future__ import annotations

from pathlib import Path

from ..adapter_base import RuntimePreflightContext, RuntimePreflightFailure
from .runtime import codex_tool_bundle_root


class CodexToolBundleSupportMixin:
    """Adapter mixin for Codex CLIs mounted into non-Codex Docker images."""

    def extra_runtime_readonly_mounts(
        self,
        *,
        runtime_backend: str,
        runtime_env: dict[str, str],
    ) -> tuple[Path, ...]:
        if runtime_backend != "docker":
            return ()
        bundle_root = codex_tool_bundle_root(runtime_env)
        return (bundle_root,) if bundle_root is not None else ()

    def runtime_preflight_failures(
        self,
        *,
        context: RuntimePreflightContext,
    ) -> tuple[RuntimePreflightFailure, ...]:
        if context.runtime_backend != "docker" or context.runtime_image_source != "resolution":
            return ()
        try:
            bundle_root = codex_tool_bundle_root(context.runtime_env)
        except Exception as exc:
            return (RuntimePreflightFailure(variant=context.variant_name, agent=self.name, error=str(exc)),)
        if bundle_root is not None:
            return ()
        return (
            RuntimePreflightFailure(
                variant=context.variant_name,
                agent=self.name,
                error=(
                    "Codex resolution-image runtimes require a repo-local Codex tool bundle. "
                    "Run 'python3 -m contextbench.run_suites_setup codex-tool-bundle'."
                ),
            ),
        )
