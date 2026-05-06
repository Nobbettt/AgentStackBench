
"""Public package surface for ContextBench coding-agent integrations."""

__all__ = [
    "build_claude_raw_response",
    "build_codex_raw_response",
    "build_prompt",
    "convert_records",
    "convert_records_with_summary",
    "convert_run_record",
    "detect_bench_from_instance_id",
    "extract_structured_output_from_value",
    "load_predictions_from_path",
    "load_predictions_with_summary_from_path",
    "load_tasks",
    "parse_bench_filter",
    "parse_instance_filter",
    "parse_unified_diff",
    "record_is_convertible",
]


def __getattr__(name: str):
    if name in {
        "convert_records",
        "convert_records_with_summary",
        "convert_run_record",
        "load_predictions_from_path",
        "load_predictions_with_summary_from_path",
        "record_is_convertible",
    }:
        from .conversion import (
            convert_records,
            convert_records_with_summary,
            convert_run_record,
            load_predictions_from_path,
            load_predictions_with_summary_from_path,
            record_is_convertible,
        )

        values = {
            "convert_records": convert_records,
            "convert_records_with_summary": convert_records_with_summary,
            "convert_run_record": convert_run_record,
            "load_predictions_from_path": load_predictions_from_path,
            "load_predictions_with_summary_from_path": load_predictions_with_summary_from_path,
            "record_is_convertible": record_is_convertible,
        }
        return values[name]
    if name == "build_prompt":
        from .prompting import build_prompt

        return build_prompt
    if name == "parse_unified_diff":
        from .records import parse_unified_diff

        return parse_unified_diff
    if name in {"build_claude_raw_response", "build_codex_raw_response", "extract_structured_output_from_value"}:
        from .response_parsing import (
            build_claude_raw_response,
            build_codex_raw_response,
            extract_structured_output_from_value,
        )

        values = {
            "build_claude_raw_response": build_claude_raw_response,
            "build_codex_raw_response": build_codex_raw_response,
            "extract_structured_output_from_value": extract_structured_output_from_value,
        }
        return values[name]
    if name in {"detect_bench_from_instance_id", "load_tasks", "parse_bench_filter", "parse_instance_filter"}:
        from .task_data import detect_bench_from_instance_id, load_tasks, parse_bench_filter, parse_instance_filter

        values = {
            "detect_bench_from_instance_id": detect_bench_from_instance_id,
            "load_tasks": load_tasks,
            "parse_bench_filter": parse_bench_filter,
            "parse_instance_filter": parse_instance_filter,
        }
        return values[name]
    raise AttributeError(name)
