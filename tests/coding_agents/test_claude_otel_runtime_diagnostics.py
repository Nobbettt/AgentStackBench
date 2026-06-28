# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from contextbench.coding_agents.constants import CLAUDE_OTEL_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.inference_limits import MAX_COMMAND_OUTPUT_CHARS
from contextbench.coding_agents.runtime import run_coding_agent_task
from contextbench.agents.claude_otel import runtime as claude_otel_runtime
from contextbench.agents.otel_common import force_command_failure, one_attempt_retry_metadata

from .claude_otel_helpers import log_record, logs_payload, otel_attr, span, traces_payload


def test_claude_otel_runtime_uses_otel_error_for_diagnostic_without_retry() -> None:
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "logs": [
                {
                    "name": "claude_code.api_error",
                    "attributes": {
                        "error": "status code: 529 overloaded",
                        "model": "claude-test",
                    },
                }
            ],
            "spans": [],
            "api_response_bodies": [],
        },
    }
    command_result = {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

    note = claude_otel_runtime._classify_diagnostic_note_from_otel(
        command_result=command_result,
        raw_response=raw_response,
        structured_output=None,
        schema_path=None,
    )

    assert note == "claude_code.api_error: status code: 529 overloaded"

    retry = one_attempt_retry_metadata(events=[])
    assert retry == {
        "attempts": 1,
        "max_attempts": 1,
        "retried": False,
        "suppressed": False,
        "suppression_reason": None,
        "events": [],
    }


def test_claude_otel_api_body_ref_errors_are_diagnostic(tmp_path) -> None:
    body_ref_root = tmp_path / "api-bodies"
    body_ref_root.mkdir()
    missing = body_ref_root / "missing.json"
    invalid = body_ref_root / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    errors = claude_otel_runtime._api_body_ref_errors(
        [
            {"name": "claude_code.api_response_body"},
            {"body_ref": str(missing)},
            {"body_ref": str(invalid)},
        ],
        body_ref_root=body_ref_root,
    )

    assert errors[0] == {
        "body_ref": None,
        "event": "claude_code.api_response_body",
        "error": "missing_body_ref",
    }
    assert errors[1] == {"body_ref": str(missing), "error": "missing"}
    assert errors[2]["body_ref"] == str(invalid)
    assert str(errors[2]["error"]).startswith("invalid_json:")
    diagnostic = claude_otel_runtime._api_body_ref_diagnostic(errors)
    assert diagnostic is not None
    assert "3 unreadable body_ref" in diagnostic


def test_claude_otel_api_body_ref_rejects_paths_outside_capture_root(tmp_path) -> None:
    body_ref_root = tmp_path / "api-bodies"
    body_ref_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    symlink = body_ref_root / "linked-outside.json"
    try:
        symlink.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is not available: {exc}")

    errors = claude_otel_runtime._api_body_ref_errors(
        [
            {"body_ref": str(outside)},
            {"body_ref": str(symlink)},
        ],
        body_ref_root=body_ref_root,
    )

    assert errors == [
        {"body_ref": str(outside), "error": "outside_body_ref_root"},
        {"body_ref": str(symlink), "error": "outside_body_ref_root"},
    ]


def test_claude_otel_compaction_does_not_read_outside_body_ref(tmp_path) -> None:
    body_ref_root = tmp_path / "api-bodies"
    body_ref_root.mkdir()
    outside = tmp_path / "outside-response.json"
    outside.write_text(
        json.dumps(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read",
                        "name": "Read",
                        "input": {"file_path": "app.py"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    capture_path = tmp_path / "claude-otel-requests.jsonl"
    capture_path.write_text(
        json.dumps(
            {
                "path": "/v1/logs",
                "body": logs_payload(
                    log_record(
                        "claude_code.api_response_body",
                        [
                            otel_attr("request_id", "req_outside"),
                            otel_attr("body_ref", str(outside)),
                        ],
                    )
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    raw_response = claude_otel_runtime.build_otel_raw_response(
        otel_capture_path=capture_path,
        command_result={"ok": True, "exit_code": 0, "signal": None, "timeout": False},
        api_bodies_dir=body_ref_root,
    )

    assert raw_response["otel"]["api_body_ref_errors"] == [
        {"body_ref": str(outside), "error": "outside_body_ref_root"}
    ]
    assert raw_response["otel"]["api_response_bodies"][0]["request_id"] == "req_outside"
    assert raw_response["otel"]["api_response_bodies"][0]["tool_uses"] == []
    assert raw_response["otel"]["summary"]["api_body_ref_error_count"] == 1


