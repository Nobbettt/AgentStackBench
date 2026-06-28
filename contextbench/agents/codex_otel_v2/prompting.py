# SPDX-License-Identifier: Apache-2.0

"""Prompting for the Codex OTEL v2 adapter."""

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
        "You are helping implement the necessary changes to satisfy the PR description in a way that is general and consistent with the codebase.",
        "Work inside the checked-out repository workspace for this task.",
        "Analyze the relevant code, make the required source changes, and verify with the strongest checks available locally.",
        "If you implement the requested code change and perform the strongest offline-safe verification available locally, set the final schema status to \"completed\" even when full repo-native tests or build-dependent verification cannot run in this benchmark environment.",
        "Reserve \"partial\" only for genuinely unfinished implementation or when you could not make the requested code change.",
        "Report verification limitations clearly in final_answer and notes.",
        "Return your final response as a JSON object that matches the required schema.",
        "Do not spend effort reconstructing a full chronological interaction log.",
        "The final response must contain these required top-level fields: status, final_answer, and notes.",
        "Populate every required field as a direct top-level JSON property or StructuredOutput tool input property; never put one required field inside another field.",
        "For Codex OTEL v2, tool-result telemetry records the files and line spans used for evaluation.",
        "Do not include retrieved_context_files, retrieved_context_spans, or retrieved_context_symbols.",
        "Keep notes as plain text for caveats and verification limitations only; do not put JSON, XML, tags, or schema fields inside notes.",
        "Do not add extra bookkeeping fields beyond the required schema.",
        "</instructions>",
    ]
    return "\n".join(lines)


__all__ = ["build_prompt"]
