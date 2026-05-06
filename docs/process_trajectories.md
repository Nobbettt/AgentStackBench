# Unified Trajectory Processing Interface

`contextbench.process_trajectories` provides a single entry point for loading, converting, validating, and merging trajectory outputs from various agents into the format required by ContextBench evaluation. It supports custom output locations and a pluggable custom parser for agents not built-in.

## Supported Trajectory Formats

| Format | Extensions / Layout | Agent |
|--------|---------------------|-------|
| MiniSWE | `.traj.json` | mini-swe-agent |
| SWE-agent | `.checkpoints.jsonl` (preferred), `.traj`, `.context.json`, `patch_context.txt` | swe-agent |
| Agentless | Instance dirs with `edit_location_individual`, `file_level_combined`, or `all_preds.jsonl` | agentless |
| Prometheus | `.log` | prometheus |
| OpenHands | `output.jsonl`, per-instance dirs with `*.json` (llm_completions) | openhands |

## Subcommands

### load

Load a trajectory file or directory and print the unified JSON format.

```bash
python -m contextbench.process_trajectories load path/to/instance.traj.json
```

Output includes `instance_id`, `model_patch` (truncated), and `traj_data` (pred_steps, pred_files, pred_spans).

### list

List trajectory files in a directory.

```bash
python -m contextbench.process_trajectories list traj/
python -m contextbench.process_trajectories list traj/ -r   # recursive
```

### convert

Convert trajectory files or directories into evaluation-ready JSONL (one instance per line). This is the main entry point for preparing agent outputs for evaluation.

**Required: `--agent`** — indicate which agent produced the trajectories. No auto-detection.

**Agent-specific path discovery:**

| Agent | Path Layout |
|-------|-------------|
| prometheus | `root/*.log` or `root/prometheus/{bench}/*.log` |
| swe-agent | `root/swe-agent/{instance_id}/*.checkpoints.jsonl` (or `.traj`, etc.) |
| mini-swe-agent | `root/mini-swe-agent/{instance_id}/*.traj.json` |
| openhands | `root/openhands/{instance_id}/` (dirs with `*.json`), `Multi/{lang}.jsonl`, `output*.jsonl` |
| agentless | `root/agentless/{idx}_{instance_id}/` (instance dirs with `edit_location_individual`, etc.) |
| custom | Edit `contextbench/parsers/custom_parser.py`; input path passed directly to `parse_custom()` |

**Examples:**

```bash
# Convert with built-in agent parser
python -m contextbench.process_trajectories convert -i /path/to/your/output -o pred.jsonl --agent prometheus

# Convert with custom parser (edit contextbench/parsers/custom_parser.py first)
python -m contextbench.process_trajectories convert -i /path/to/output -o pred.jsonl --agent custom
```

**Options:**

- `-i`, `--input`: Input path(s); default `traj`. Can be files or directories.
- `-o`, `--out`: Output JSONL path (required).
- `-a`, `--agent`: **Required**. Agent that produced the trajectories: `prometheus`, `openhands`, `swe-agent`, `mini-swe-agent`, `agentless`, `custom`.
- `-r`, `--recursive`: Recurse subdirectories when scanning.

### validate

Validate that a trajectory file or directory conforms to the expected format.

```bash
python -m contextbench.process_trajectories validate path/to/pred.jsonl
```

### merge

Merge multiple trajectory sources into a single JSONL file.

```bash
python -m contextbench.process_trajectories merge traj/agentless/ traj/mini-swe-agent/ -o merged.jsonl --dedupe
```

`--dedupe` deduplicates by `instance_id`.

### stats

Print trajectory statistics (instance count, steps, files).

```bash
python -m contextbench.process_trajectories stats pred.jsonl
```

**Evaluation:** Use the converted JSONL output with `python -m contextbench.evaluate` for evaluation. See the user guide for the full workflow.

## Custom Parser (For Your Own Agent)

If your agent format is not in the built-in list, edit `contextbench/parsers/custom_parser.py` and implement the `parse_custom(path: str) -> List[dict]` function to parse your trajectory format.

**Steps:**

1. Open `contextbench/parsers/custom_parser.py`.
2. Replace the stub implementation of `parse_custom` with your parsing logic.
3. Ensure each returned dict has:
   - `instance_id` (str): e.g. `"owner__repo-12345"`
   - `traj_data` (dict): with `pred_steps`, `pred_files`, `pred_spans` (see unified format below)
   - `model_patch` (str, optional): for EditLoc metrics

**Example implementation:**

```python
def parse_custom(path: str) -> List[dict]:
    import json
    import os
    results = []
    # If path is a directory, iterate instance subdirs
    if os.path.isdir(path):
        for d in os.listdir(path):
            instance_dir = os.path.join(path, d)
            # Parse your format and build traj_data
            traj_data = {"pred_steps": [...], "pred_files": [...], "pred_spans": {...}}
            results.append({"instance_id": d, "traj_data": traj_data, "model_patch": ""})
    # If path is a file (e.g. JSONL), read and convert each line
    else:
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                # Convert your format to traj_data
                results.append({...})
    return results
```

**Usage after editing:**

```bash
python -m contextbench.process_trajectories convert -i /path/to/your/output -o pred.jsonl --agent custom
```

## Unified Format

The internal format used by ContextBench:

```json
{
  "instance_id": "owner__repo-1234",
  "traj_data": {
    "pred_steps": [{"files": [...], "spans": {...}, "symbols": {...}}, ...],
    "pred_files": [...],
    "pred_spans": {"file_path": [{"type": "line", "start": 1, "end": 10}, ...]}
  },
  "model_patch": "..."
}
```

This format is accepted by `contextbench.evaluate` for evaluation.
