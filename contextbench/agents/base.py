
"""Shared base classes for agent-specific response parsers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from ..coding_agents.files import read_json
from ..coding_agents.types import CodingAgentRawResponse, StructuredOutput, TokenUsage, ToolCall, TrajectoryData


class BaseCodingAgentParser(ABC):
    """Abstract parser for wrapper-produced coding-agent records."""

    def load_record(self, source: str | Path | dict[str, object]) -> dict[str, object]:
        if isinstance(source, dict):
            return dict(source)
        path = Path(source)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_raw_response(self, record: dict[str, object]) -> CodingAgentRawResponse | None:
        raw_response_path = record.get("raw_response_path")
        if raw_response_path:
            path = Path(raw_response_path)
            if path.exists():
                return read_json(path)
        return record.get("raw_response")

    @abstractmethod
    def extract_structured_output(self, raw_response: CodingAgentRawResponse) -> StructuredOutput | None:
        """Extract the benchmark-facing structured output from a raw response object."""

    @abstractmethod
    def extract_token_usage(self, raw_response: CodingAgentRawResponse) -> TokenUsage | None:
        """Extract precise token-usage metadata from a raw response object."""

    @abstractmethod
    def extract_tool_calls(self, raw_response: CodingAgentRawResponse) -> list[ToolCall]:
        """Extract precise tool-call metadata from a raw response object."""

    def infer_trajectory_data(
        self,
        raw_response: CodingAgentRawResponse,
        *,
        record: dict[str, object],
    ) -> TrajectoryData | None:
        return None

    def normalize_record(self, source: str | Path | dict[str, object]) -> dict[str, object]:
        record = self.load_record(source)
        raw_response = self.load_raw_response(record)
        if raw_response is not None:
            if not isinstance(record.get("final_output"), dict):
                record["final_output"] = self.extract_structured_output(raw_response)
            if not isinstance(record.get("token_usage"), dict):
                record["token_usage"] = self.extract_token_usage(raw_response)
            if "tool_calls" not in record:
                record["tool_calls"] = self.extract_tool_calls(raw_response)
        final_output = record.get("final_output")
        if isinstance(final_output, dict) and not final_output.get("task_id"):
            final_output["task_id"] = (
                record.get("instance_id")
                or record.get("original_inst_id")
                or ""
            )
        return record

    def extract_trajectory(self, source: str | Path | dict[str, object]) -> dict[str, object]:
        from ..coding_agents.conversion import convert_run_record

        record = self.normalize_record(source)
        return convert_run_record(record, parser=self)["traj_data"]
