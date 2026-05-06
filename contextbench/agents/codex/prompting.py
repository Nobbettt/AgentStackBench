
"""Codex-specific benchmark prompt construction."""

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
        "Populate these fields carefully: status, final_answer, notes, retrieved_context_files, retrieved_context_spans, and retrieved_context_symbols.",
        "Use retrieved_context_files and retrieved_context_spans for that final relied-on context.",
        "Use retrieved_context_symbols when you know the important symbols, otherwise leave it empty.",
        "Do not add extra bookkeeping fields beyond the required schema.",
        "</instructions>",
    ]
    return "\n".join(lines)
