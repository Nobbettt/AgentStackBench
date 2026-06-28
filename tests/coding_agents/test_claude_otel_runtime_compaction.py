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


def test_claude_otel_compaction_dedupes_and_bounds_tool_results(tmp_path) -> None:
    body_ref_root = tmp_path / "api-bodies"
    body_ref_root.mkdir()
    first_body = body_ref_root / "request-1.json"
    second_body = body_ref_root / "request-2.json"
    huge_content = "x" * (MAX_COMMAND_OUTPUT_CHARS + 10)
    first_body.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "first result",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second_body.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "duplicate result",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_2",
                                "content": huge_content,
                            },
                        ],
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
                        "claude_code.api_request_body",
                        [
                            otel_attr("request_id", "req_1"),
                            otel_attr("body_ref", str(first_body)),
                        ],
                    ),
                    log_record(
                        "claude_code.api_request_body",
                        [
                            otel_attr("request_id", "req_2"),
                            otel_attr("body_ref", str(second_body)),
                        ],
                    ),
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

    summaries = raw_response["otel"]["api_request_bodies"]
    assert summaries[0]["tool_results"] == [
        {
            "tool_use_id": "toolu_1",
            "content": "first result",
            "content_chars": len("first result"),
            "original_content_chars": len("first result"),
            "content_truncated": False,
            "is_error": None,
        }
    ]
    assert len(summaries[1]["tool_results"]) == 1
    compacted = summaries[1]["tool_results"][0]
    assert compacted["tool_use_id"] == "toolu_2"
    assert compacted["content"] == huge_content[:MAX_COMMAND_OUTPUT_CHARS]
    assert compacted["content_chars"] == MAX_COMMAND_OUTPUT_CHARS
    assert compacted["original_content_chars"] == MAX_COMMAND_OUTPUT_CHARS + 10
    assert compacted["content_truncated"] is True


def test_claude_otel_missing_capture_is_diagnostic() -> None:
    assert (
        claude_otel_runtime._otel_capture_diagnostic(
            {"summary": {"request_count": 0, "api_request_body_count": 0, "api_response_body_count": 0}}
        )
        == "Claude OTEL capture did not receive any OTLP requests."
    )
    assert (
        claude_otel_runtime._otel_capture_diagnostic(
            {"summary": {"request_count": 1, "api_request_body_count": 0, "api_response_body_count": 0}}
        )
        == "Claude OTEL capture did not include any API request body artifacts."
    )
    assert (
        claude_otel_runtime._otel_capture_diagnostic(
            {"summary": {"request_count": 1, "api_request_body_count": 1, "api_response_body_count": 0}}
        )
        == "Claude OTEL capture did not include any API response body artifacts."
    )
    assert (
        claude_otel_runtime._otel_capture_diagnostic(
            {"summary": {"request_count": 1, "api_request_body_count": 1, "api_response_body_count": 1}}
        )
        is None
    )


def test_claude_otel_rejects_unsafe_persisted_tool_result_source(tmp_path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    unsafe_path = tmp_path / "outside-runtime" / "toolu_1.txt"
    unsafe_path.parent.mkdir()
    unsafe_path.write_text("secret\n", encoding="utf-8")
    raw_response = {
        "agent": "claude-otel",
        "response_format": "otel-http-json",
        "otel": {
            "api_request_bodies": [
                {
                    "source": "claude.otel.api_request_body",
                    "tool_results": [
                        {
                            "tool_use_id": "toolu_1",
                            "content": (
                                "<persisted-output>\n"
                                f"Output too large (20KB). Full output saved to: {unsafe_path}\n"
                                "</persisted-output>"
                            ),
                        }
                    ],
                }
            ]
        },
    }

    persisted = claude_otel_runtime._archive_persisted_tool_results_from_otel(raw_response, task_dir=task_dir)

    assert persisted[0]["status"] == "rejected_unsafe_source"
    assert persisted[0]["artifact_path"] is None
    assert not (task_dir / "claude-persisted-tool-results").exists()


