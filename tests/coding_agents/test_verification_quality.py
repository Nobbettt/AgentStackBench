# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from contextbench.coding_agents.verification_quality import (
    analyze_record_quality,
    analyze_verification_quality,
    command_executions_from_record,
    detect_added_regression_tests,
)


def test_verification_quality_classifies_successful_repo_test() -> None:
    record = {
        "command_executions": [
            {
                "command": "python -m pytest tests/test_parser.py -q",
                "payload": {"exit_code": 0},
            }
        ],
        "final_output": {"final_answer": "Done", "notes": ""},
    }

    quality = analyze_verification_quality(record)

    assert quality["strongest_verification"] == "repo_tests"
    assert quality["successful_runtime_verification"] is True
    assert quality["syntax_only"] is False


def test_verification_quality_flags_syntax_only_environment_limitation() -> None:
    record = {
        "command_executions": [
            {
                "command": "python -m py_compile sympy/concrete/products.py",
                "payload": {"exit_code": 0},
            }
        ],
        "final_output": {
            "final_answer": "I could not run the tests because mpmath is a missing dependency.",
            "notes": "",
        },
    }

    quality = analyze_verification_quality(record)

    assert quality["strongest_verification"] == "syntax_or_static"
    assert quality["syntax_only"] is True
    assert quality["environment_limited"] is True
    assert quality["environment_limitation_matches"]


def test_regression_test_detector_flags_added_test_not_run() -> None:
    patch = """diff --git a/tests/test_products.py b/tests/test_products.py
--- a/tests/test_products.py
+++ b/tests/test_products.py
@@
+def test_product_rational_exponent():
+    assert Product(n + 1 / 2**k, (k, 0, n - 1)).doit()
"""

    diagnostic = detect_added_regression_tests(
        patch,
        [{"command": "python -m py_compile sympy/concrete/products.py", "payload": {"exit_code": 0}}],
    )

    assert diagnostic["added_regression_test"] is True
    assert diagnostic["regression_tests_run"] is False
    assert diagnostic["added_test_files"] == ["tests/test_products.py"]


def test_regression_test_detector_accepts_covering_test_command() -> None:
    patch = """diff --git a/tests/test_products.py b/tests/test_products.py
--- a/tests/test_products.py
+++ b/tests/test_products.py
@@
+def test_product_rational_exponent():
+    assert True
"""

    diagnostic = detect_added_regression_tests(
        patch,
        [{"command": "pytest tests/test_products.py -q", "payload": {"exit_code": 0}}],
    )

    assert diagnostic["added_regression_test"] is True
    assert diagnostic["regression_tests_run"] is True
    assert diagnostic["covering_commands"] == ["pytest tests/test_products.py -q"]


def test_claude_bash_commands_are_extracted_from_raw_response_path(tmp_path: Path) -> None:
    raw_response_path = tmp_path / "raw-response.json"
    raw_response_path.write_text(
        json.dumps(
            {
                "agent": "claude",
                "response_format": "stream-json",
                "response": [
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Bash",
                                    "input": {"command": "pytest tests/test_widget.py -q"},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "1 passed",
                                }
                            ]
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    record = {"raw_response_path": str(raw_response_path), "model_patch": ""}
    commands = command_executions_from_record(record)
    quality = analyze_record_quality(record)["verification_quality"]

    assert commands[0]["command"] == "pytest tests/test_widget.py -q"
    assert quality["strongest_verification"] == "repo_tests"


def test_codex_commands_are_extracted_from_raw_response_path(tmp_path: Path) -> None:
    raw_response_path = tmp_path / "raw-response.json"
    raw_response_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "response_format": "jsonl-events",
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "python -m pytest tests/test_widget.py -q",
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record = {"raw_response_path": str(raw_response_path), "model_patch": ""}
    quality = analyze_record_quality(record)["verification_quality"]

    assert quality["strongest_verification"] == "repo_tests"
