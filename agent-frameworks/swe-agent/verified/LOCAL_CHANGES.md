# Local changes in `agent-frameworks/swe-agent/verified`

This file documents what was changed **under** `Context-Bench/agent-frameworks/swe-agent/verified/` during debugging/running SWE-agent locally.

## Summary

- The original failure was running `run_verified.sh` incorrectly via `python -m ...` and then hitting missing dependency `swerex`.
- This SWE-agent snapshot requires **Python >= 3.11** and **SWE-ReX** (`swe-rex`, import name `swerex`).
- To make `run_verified.sh` runnable, we created a required directory (`trajectories/`) and a small instance list file (`one_instance.txt`) for a single-instance smoke test.

## What was added/created in this directory

- **`trajectories/`** (directory)
  - **Why**: `sweagent/__init__.py` asserts that `TRAJECTORY_DIR` exists at import time (default points to `./trajectories`).
- **`one_instance.txt`** (file)
  - **Why**: `run_verified.sh` expects `VERIFIED_FILE` to contain instance IDs (one per line). This file was created to run a **single** instance (`django__django-7530`) for validation.
- **`run_verified_*.log`** (files)
  - **Why**: Produced by `run_verified.sh` (`tee "$LOG_FILE"`).
- **`sweagent.egg-info/`** (directory)
  - **Why**: Created by installing the package in editable mode (`pip install -e .`) in a Python 3.11 environment.

## Environment / dependency changes (outside this repo)

These are not repository code changes, but were required to run the vendored SWE-agent:

- Created conda env: **`sweagent311`** (Python 3.11)
- Installed dependencies into `sweagent311`:
  - `swe-rex==1.4.0` (provides `import swerex`)
  - `pip install -e .` from this directory (installs `sweagent` itself)

## How to run (recommended)

### 1) Single-instance smoke test

```bash
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/verified
echo "django__django-7530" > one_instance.txt
VERIFIED_FILE="$PWD/one_instance.txt" conda run -n sweagent311 bash ./run_verified.sh 1
```

### 2) Batch run with your own instance list

Create a file with one instance id per line, then:

```bash
VERIFIED_FILE="/abs/path/to/your_instances.txt" conda run -n sweagent311 bash ./run_verified.sh 4
```

## Notes about evaluation output formats

- `Context-Bench` evaluation typically extracts context successfully from SWE-agent **`.checkpoints.jsonl`** outputs.
- A `.traj` file can legitimately yield `no_context_extracted` if it contains no usable context (e.g., early `exit_error` before any file views).

