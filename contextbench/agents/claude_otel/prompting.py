# SPDX-License-Identifier: Apache-2.0

"""Minimal Claude v2 prompt construction for OTEL-backed evaluation."""

from __future__ import annotations


def build_prompt(task: dict[str, object]) -> str:
    lines = [
        f"You are working on a programming task in repository {(task.get('repo') or task.get('repo_url') or 'unknown-repo')}.",
        "",
        "<pr_description>",
        "Consider the following PR description:",
        task.get("prompt") or "No task prompt was available.",
        "</pr_description>",
        "",
        "<instructions>",
        "Implement the necessary changes in the checked-out repository workspace.",
        "Analyze the relevant code, make the required source changes, and verify with the strongest checks available locally.",
        "If you implement the requested code change and perform the strongest offline-safe verification available locally, set status to \"completed\" even when full repo-native tests or build-dependent verification cannot run in this benchmark environment.",
        "Reserve \"partial\" only for genuinely unfinished implementation or when you could not make the requested code change.",
        "Return a JSON object matching the required schema with exactly these top-level fields: status, final_answer, and notes.",
        "Use final_answer for a concise implementation and verification summary.",
        "Use notes only for caveats, blockers, or verification limitations.",
        "</instructions>",
    ]
    return "\n".join(lines)
