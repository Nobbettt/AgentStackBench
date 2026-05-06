#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-03-16 from upstream ContextBench.
# Summary of changes: support converting coding-agent run records into ContextBench trajectories and emit attempt-aware conversion summaries.

"""
Unified Trajectory Processing Interface

Converts trajectory outputs from various agents into the unified format required
by ContextBench evaluation. Supports custom output locations and a pluggable
parser for non-built-in agents.

The converted JSONL can be used with contextbench.evaluate for evaluation.

Usage:
  # Convert with built-in parser
  python -m contextbench.process_trajectories convert -i /path/to/your/output -o pred.jsonl --agent prometheus

  # Convert Codex or Claude run records produced by contextbench.run
  python -m contextbench.process_trajectories convert -i results/agent_runs/codex -o pred.jsonl --agent codex

  # Convert with custom parser (edit contextbench/parsers/custom_parser.py first)
  python -m contextbench.process_trajectories convert -i /path/to/output -o pred.jsonl --agent custom
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from contextbench.agents.registry import has_coding_agent_adapter, normalize_coding_agent_name
from contextbench.coding_agents.conversion import (
    load_predictions_from_path,
    load_predictions_with_summary_from_path,
)


def _load_pred(path: str) -> List[dict]:
    """Load trajectories using contextbench parser."""
    from contextbench.parsers import load_pred
    return load_pred(path)


def _load_agentless_dir(instance_dir: Path) -> dict:
    """Load trajectory from agentless instance directory via extract_agentless."""
    from contextbench.agents.agentless.extract import extract_agentless
    # instance_id from dir name: 0_scikit-learn__scikit-learn-25232 -> scikit-learn__scikit-learn-25232
    name = instance_dir.name
    instance_id = name.split("_", 1)[1] if name[0].isdigit() and "_" in name else name
    pc = extract_agentless(str(instance_dir), instance_id)
    # Convert PredContext to unified format
    pred_spans_dict: Dict[str, List[dict]] = {}
    for s in pc.pred_spans:
        f = (s or {}).get("file", "")
        if not f:
            continue
        pred_spans_dict.setdefault(f, []).append({"type": "line", "start": s["start_line"], "end": s["end_line"]})
    pred_symbols_dict = {f: list(v) for f, v in pc.pred_symbols.items()}
    traj_data = {
        "pred_steps": [{"files": pc.pred_files, "spans": pred_spans_dict, "symbols": pred_symbols_dict}],
        "pred_files": pc.pred_files,
        "pred_spans": pred_spans_dict,
        "pred_symbols": pred_symbols_dict,
    }
    model_patch = ""
    all_preds = instance_dir / "all_preds.jsonl"
    if all_preds.is_file():
        for line in all_preds.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (rec.get("instance_id") or rec.get("inst_id")) == instance_id:
                    model_patch = rec.get("model_patch", "")
                    break
            except Exception:
                pass
    return {"instance_id": instance_id, "traj_data": traj_data, "model_patch": model_patch}


def _load_path(path: str, agent: Optional[str] = None) -> List[dict]:
    """Load trajectory from path; handles files, OpenHands dirs, and agentless instance dirs."""
    p = Path(path)
    normalized_agent = normalize_coding_agent_name(agent.lower().replace("_", "-")) if agent else None
    if normalized_agent and has_coding_agent_adapter(normalized_agent):
        return load_predictions_from_path(p, expected_agent=normalized_agent)
    if p.is_dir():
        if agent == "agentless":
            try:
                return [_load_agentless_dir(p)]
            except Exception:
                pass
        loaded = _load_traj_file(path)
        return [loaded] if loaded else []
    return _load_pred(path)


def _default_summary_path(out_path: Path) -> Path:
    return out_path.with_suffix(".summary.json")


def _empty_convert_summary(*, agent: str, input_count: int) -> dict[str, object]:
    return {
        "scope": "converted_predictions",
        "agent": agent,
        "input_count": input_count,
        "processed_input_count": 0,
        "input_error_count": 0,
        "selected_task_count": None,
        "record_count": 0,
        "convertible_record_count": 0,
        "prediction_count": 0,
        "missing_record_path_count": 0,
        "nonconvertible_record_count": 0,
        "coverage_of_attempted_tasks": None,
        "missing_prediction_count": None,
        "is_partial": False,
    }


def _merge_convert_summary(base: dict[str, object], current: dict[str, object]) -> None:
    for key in (
        "record_count",
        "convertible_record_count",
        "prediction_count",
        "missing_record_path_count",
        "nonconvertible_record_count",
    ):
        base[key] = int(base.get(key) or 0) + int(current.get(key) or 0)
    selected_task_count = current.get("selected_task_count")
    if isinstance(selected_task_count, int):
        base["selected_task_count"] = int(base.get("selected_task_count") or 0) + selected_task_count


def _finalize_convert_summary(summary: dict[str, object]) -> dict[str, object]:
    task_count = summary.get("selected_task_count")
    if isinstance(task_count, int):
        prediction_count = int(summary.get("prediction_count") or 0)
        summary["coverage_of_attempted_tasks"] = (prediction_count / task_count) if task_count else 0.0
        summary["missing_prediction_count"] = max(task_count - prediction_count, 0)
        summary["is_partial"] = bool(task_count and prediction_count < task_count)
    else:
        summary["coverage_of_attempted_tasks"] = None
        summary["missing_prediction_count"] = None
        summary["is_partial"] = False
    return summary


def _load_traj_file(path: str) -> dict:
    """Load single trajectory file."""
    from contextbench.parsers import load_traj_file
    return load_traj_file(path)


# ---------------------------------------------------------------------------
# Path discovery for convert (user-provided root, agent-specific layouts)
# ---------------------------------------------------------------------------

def _collect_paths_by_agent(root: Path, agent: str, recursive: bool) -> List[Path]:
    """
    Collect trajectory paths under root using agent-specific layout.
    root is treated as the agent output root (e.g. your custom dir with .log files).
    """
    root = root.resolve()
    agent = agent.lower().replace("_", "-")
    out: List[Path] = []

    if agent == "prometheus":
        # Support both: root/*.log (custom dir) and root/prometheus/*.log (traj-like)
        prom_root = root / "prometheus" if (root / "prometheus").is_dir() else root
        for p in prom_root.rglob("*.log") if recursive else prom_root.glob("*.log"):
            if p.is_file():
                out.append(p)
        for bench in ("verified", "pro", "poly", "multi"):
            sub = prom_root / bench
            if sub.is_dir():
                for p in sub.glob("*.log"):
                    if p.is_file():
                        out.append(p)
    elif agent == "swe-agent":
        agent_root = root / "swe-agent" if (root / "swe-agent").is_dir() else root
        for d in agent_root.iterdir():
            if not d.is_dir():
                continue
            # Prefer .checkpoints.jsonl (run_all_traj_eval_fixed.sh uses it)
            for ext in (".checkpoints.jsonl", ".traj", ".context.json", "patch_context.txt"):
                for p in d.glob(f"*{ext}"):
                    if p.name.endswith(ext) and p.is_file():
                        out.append(p)
                        break
                else:
                    continue
                break
    elif agent == "mini-swe-agent":
        agent_root = root / "mini-swe-agent" if (root / "mini-swe-agent").is_dir() else root
        for d in agent_root.iterdir():
            if not d.is_dir():
                continue
            for p in d.glob("*.traj.json"):
                if p.is_file():
                    out.append(p)
                    break
    elif agent == "openhands":
        agent_root = root / "openhands" if (root / "openhands").is_dir() else root
        for d in agent_root.iterdir():
            if not d.is_dir() or d.name in ("Multi", "Pro", "Poly", "verified", "Verified"):
                continue
            if list(d.glob("*.json")):
                out.append(d)
        for p in agent_root.rglob("output.jsonl") if recursive else []:
            if p.is_file():
                out.append(p)
        for p in agent_root.glob("output*.jsonl"):
            if p.is_file():
                out.append(p)
        multi = agent_root / "Multi"
        if multi.is_dir():
            for lang in ("c", "cpp", "go", "java", "javascript", "rust", "typescript"):
                q = multi / f"{lang}.jsonl"
                if q.is_file():
                    out.append(q)
    elif agent == "agentless":
        agent_root = root / "agentless" if (root / "agentless").is_dir() else root
        for d in agent_root.iterdir():
            if not d.is_dir():
                continue
            # Use instance dir (extract_agentless scans edit_location_individual, file_level, etc.)
            if (d / "edit_location_individual").is_dir() or (d / "file_level_combined").is_dir() or (d / "all_preds.jsonl").is_file():
                out.append(d)
    else:
        raise ValueError(
            f"Unknown agent: {agent}. Use one of: prometheus, swe-agent, mini-swe-agent, openhands, agentless, codex, claude, custom."
        )
    return sorted(set(out))


def _resolve_parser(module_func: str) -> Callable[[str], List[dict]]:
    """Resolve module:function to a callable (path) -> List[dict]."""
    if ":" not in module_func:
        raise ValueError("--parser must be module:function, e.g. my_parser:parse_custom")
    mod_name, func_name = module_func.rsplit(":", 1)
    spec = importlib.util.find_spec(mod_name)
    if spec is None or spec.loader is None:
        raise ValueError(f"Module not found: {mod_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise ValueError(f"Function {func_name} not found in {mod_name}")
    return fn


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_load(args: argparse.Namespace) -> int:
    """Load a trajectory file/dir and print unified format as JSON."""
    path = args.path
    if not Path(path).exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        return 2
    try:
        preds = _load_pred(path)
        if not preds:
            print("ERROR: No trajectories loaded", file=sys.stderr)
            return 1
        for p in preds:
            mp = p.get("model_patch") or ""
            if len(mp) > 200:
                mp = mp[:200] + "..."
            out = {
                "instance_id": p.get("instance_id"),
                "model_patch": mp,
                "traj_data": p.get("traj_data", {}),
            }
            print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List trajectory files in a directory."""
    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: Path not found: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(str(root))
        return 0

    suffixes = (
        ".traj.json", "_traj.json", ".checkpoints.jsonl",
        ".context.json", ".traj", ".log",
        "patch_context.txt", "output.jsonl"
    )
    found: List[Path] = []
    pattern = "**/*" if args.recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        name = p.name
        if any(name.endswith(s) or s in name for s in suffixes):
            found.append(p)
    for p in sorted(found):
        rel = p.relative_to(root) if args.recursive else p.name
        print(rel)
    print(f"# Total: {len(found)}", file=sys.stderr)
    return 0


CUSTOM_PARSER_SPEC = "contextbench.parsers.custom_parser:parse_custom"


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert trajectory files to evaluation-ready JSONL."""
    if not args.agent:
        print("ERROR: convert requires --agent. Specify which agent produced the trajectories.", file=sys.stderr)
        print("  --agent: prometheus, openhands, swe-agent, mini-swe-agent, agentless, codex, claude, custom", file=sys.stderr)
        print("  Use --agent custom when format is not built-in (edit contextbench/parsers/custom_parser.py)", file=sys.stderr)
        return 2

    inputs = [Path(p) for p in args.input]
    for inp in inputs:
        if not inp.exists():
            print(f"ERROR: Path not found: {inp}", file=sys.stderr)
            return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if args.summary_out else _default_summary_path(out_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    agent = args.agent.lower().replace("_", "-")
    written = 0
    summary = _empty_convert_summary(agent=agent, input_count=len(inputs))
    with open(out_path, "w", encoding="utf-8") as f:
        if agent == "custom":
            parser_fn = _resolve_parser(CUSTOM_PARSER_SPEC)
            for inp in inputs:
                try:
                    preds = parser_fn(str(inp))
                    summary["processed_input_count"] = int(summary["processed_input_count"] or 0) + 1
                    for p in preds:
                        _ensure_traj_format(p)
                        f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
                        written += 1
                except Exception as e:
                    summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                    print(f"  Warning: {inp}: {e}", file=sys.stderr)
        elif has_coding_agent_adapter(agent):
            for inp in inputs:
                try:
                    preds, input_summary = load_predictions_with_summary_from_path(inp, expected_agent=agent)
                    summary["processed_input_count"] = int(summary["processed_input_count"] or 0) + 1
                    _merge_convert_summary(summary, input_summary)
                    for p in preds:
                        _ensure_traj_format(p)
                        f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
                        written += 1
                except Exception as e:
                    summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                    print(f"  Warning: {inp}: {e}", file=sys.stderr)
        else:
            for inp in inputs:
                paths = _collect_paths_by_agent(inp, agent, args.recursive)
                summary["processed_input_count"] = int(summary["processed_input_count"] or 0) + 1
                for fp in paths:
                    try:
                        preds = _load_path(str(fp), agent=agent)
                        for p in preds:
                            _ensure_traj_format(p)
                            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
                            written += 1
                    except Exception as e:
                        summary["input_error_count"] = int(summary["input_error_count"] or 0) + 1
                        print(f"  Warning: {fp}: {e}", file=sys.stderr)
    if not has_coding_agent_adapter(agent):
        summary["prediction_count"] = written
    summary = _finalize_convert_summary(summary)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote {written} records to {out_path}", file=sys.stderr)
    print(f"Wrote conversion summary to {summary_path}", file=sys.stderr)
    if summary["is_partial"]:
        print(
            "WARNING: Conversion covered a subset of attempted tasks "
            f"({summary.get('prediction_count')}/{summary.get('selected_task_count')}).",
            file=sys.stderr,
        )
        return 1
    if int(summary.get("input_error_count") or 0) > 0:
        print(
            f"ERROR: Conversion failed for {summary.get('input_error_count')} input(s).",
            file=sys.stderr,
        )
        return 1
    return 0


def _ensure_traj_format(p: dict) -> None:
    """Ensure record has instance_id and traj_data for evaluate."""
    if "instance_id" not in p:
        p["instance_id"] = p.get("original_inst_id", "")
    if "traj_data" not in p or not isinstance(p.get("traj_data"), dict):
        p["traj_data"] = p.get("traj_data") or {"pred_steps": [], "pred_files": [], "pred_spans": {}}


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate trajectory format."""
    path = args.path
    if not Path(path).exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        return 2
    try:
        preds = _load_pred(path)
        if not preds:
            print("FAIL: No trajectories loaded", file=sys.stderr)
            return 1
        for i, p in enumerate(preds):
            tid = p.get("instance_id", "?")
            td = p.get("traj_data", {})
            steps = td.get("pred_steps", [])
            files = td.get("pred_files", [])
            spans = td.get("pred_spans", {})
            ok = bool(steps or files or spans)
            status = "OK" if ok else "WARN (empty traj_data)"
            print(f"  [{i+1}] {tid}: {status} (steps={len(steps)}, files={len(files)})")
        print("Validation passed", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge multiple trajectory sources into one JSONL."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for src in args.sources:
            p = Path(src)
            if not p.exists():
                print(f"  Skip (not found): {src}", file=sys.stderr)
                continue
            try:
                if p.is_file():
                    preds = _load_pred(str(p))
                else:
                    preds = []
                    for fp in p.rglob("*"):
                        if fp.is_file() and any(
                            str(fp).endswith(s) for s in
                            (".traj.json", "_traj.json", ".checkpoints.jsonl", ".context.json", ".traj", ".log", "output.jsonl")
                        ):
                            preds.extend(_load_pred(str(fp)))
                for pred in preds:
                    kid = pred.get("instance_id") or pred.get("original_inst_id") or ""
                    if args.dedupe and kid in seen:
                        continue
                    seen.add(kid)
                    f.write(json.dumps(pred, ensure_ascii=False, default=str) + "\n")
                    written += 1
            except Exception as e:
                print(f"  Warning: {src}: {e}", file=sys.stderr)
    print(f"Merged {written} records to {out_path}", file=sys.stderr)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Print trajectory statistics."""
    path = args.path
    if not Path(path).exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        return 2
    try:
        preds = _load_pred(path)
        if not preds:
            print("No trajectories loaded", file=sys.stderr)
            return 0
        total_steps = 0
        total_files = 0
        for p in preds:
            td = p.get("traj_data", {})
            total_steps += len(td.get("pred_steps", []))
            total_files += len(td.get("pred_files", []))
        print(f"Instances: {len(preds)}")
        print(f"Total steps: {total_steps}")
        print(f"Total final files: {total_files}")
        if preds:
            print(f"Avg steps/instance: {total_steps / len(preds):.1f}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified trajectory processing interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sp = ap.add_subparsers(dest="command", required=True)

    # load
    p_load = sp.add_parser("load", help="Load trajectory and print unified JSON")
    p_load.add_argument("path", help="Trajectory file or directory")
    p_load.set_defaults(func=cmd_load)

    # list
    p_list = sp.add_parser("list", help="List trajectory files in directory")
    p_list.add_argument("path", help="Directory to scan")
    p_list.add_argument("-r", "--recursive", action="store_true", help="Recurse subdirectories")
    p_list.set_defaults(func=cmd_list)

    # convert
    p_convert = sp.add_parser(
        "convert",
        help="Convert agent output to ContextBench pred JSONL (use contextbench.evaluate for evaluation)",
    )
    p_convert.add_argument(
        "-i", "--input",
        nargs="+",
        default=["traj"],
        help="Input path(s): your custom dir or files (default: traj)",
    )
    p_convert.add_argument("--out", "-o", required=True, help="Output JSONL path")
    p_convert.add_argument(
        "--summary-out",
        help="Optional summary JSON path. Defaults to <out>.summary.json",
    )
    p_convert.add_argument(
        "--agent", "-a",
        required=True,
        help="Agent that produced trajectories: prometheus, openhands, swe-agent, mini-swe-agent, agentless, codex, claude, custom (edit contextbench/parsers/custom_parser.py)",
    )
    p_convert.add_argument("-r", "--recursive", action="store_true", help="Recurse subdirectories")
    p_convert.add_argument(
        "--fail-on-partial",
        action="store_true",
        help="Return exit code 1 when conversion covers only a subset of attempted tasks.",
    )
    p_convert.set_defaults(func=cmd_convert)

    # validate
    p_validate = sp.add_parser("validate", help="Validate trajectory format")
    p_validate.add_argument("path", help="Trajectory file or directory")
    p_validate.set_defaults(func=cmd_validate)

    # merge
    p_merge = sp.add_parser("merge", help="Merge multiple trajectory sources")
    p_merge.add_argument("sources", nargs="+", help="Paths to trajectory files or directories")
    p_merge.add_argument("--out", "-o", required=True, help="Output JSONL path")
    p_merge.add_argument("--dedupe", action="store_true", help="Deduplicate by instance_id")
    p_merge.set_defaults(func=cmd_merge)

    # stats
    p_stats = sp.add_parser("stats", help="Print trajectory statistics")
    p_stats.add_argument("path", help="Trajectory file or directory")
    p_stats.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
