# Local changes in `agent-frameworks/swe-agent/poly`

This document records the **local fixes/adjustments** made in `Context-Bench/agent-frameworks/swe-agent/poly/` so that the scripts can "run up" (start successfully) in this workspace.

## Summary (what was wrong)

- The SWE-agent snapshot in this folder requires **Python >= 3.11** and **SWE-ReX** (`swe-rex`, import name `swerex`).
- When running `run_pro.sh`, dataset download could fail if your environment uses a **SOCKS proxy** but `socksio` is not installed (httpx raises an ImportError).
- `sweagent/__init__.py` asserts that a trajectories directory exists at import time (default: `./trajectories`).

## What was added/created in this directory

- **`trajectories/`** (directory)
  - **Why**: required by `sweagent` import-time assertion; it can be empty.
- **`one_instance.txt`** (file)
  - **Why**: convenience file for single-instance smoke tests via `VERIFIED_FILE=...` with `run_verified.sh`.
  - Contents used: `django__django-7530`
- **`run_verified_*.log`, `run_pro_*.log`** (files)
  - **Why**: produced by the scripts via `tee`.

## Environment / dependency changes (outside this repo)

These are not repository code changes, but were required to run the vendored SWE-agent:

- Use conda env **`sweagent311`** (Python 3.11)
- Install SWE-ReX + socks support:
  - `pip install -U swe-rex`
  - `pip install -U "httpx[socks]"` (installs `socksio`)
- Install this snapshot editable:
  - `pip install -e .`

## How to run

### 1) Verified subset (single-instance smoke test)

```bash
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/poly
echo "django__django-7530" > one_instance.txt
VERIFIED_FILE="$PWD/one_instance.txt" conda run -n sweagent311 bash ./run_verified.sh 1
```

Notes:
- If an existing `output/<id>/<id>.traj` exists, SWE-agent may **skip** it.

### 2) Pro subset (single-instance smoke test)

```bash
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/poly
conda run -n sweagent311 bash ./run_pro.sh 1 "django__django-7530"
```

If you see:
`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`
install socks support in the env:

```bash
conda run -n sweagent311 python -m pip install -U "httpx[socks]"
```

