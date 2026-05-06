# Local changes in `agent-frameworks/swe-agent/pro`

This document records the **local fixes/adjustments** made in `Context-Bench/agent-frameworks/swe-agent/pro/` so that the scripts can "run up" (start successfully) in this workspace.

## Background: why `pro` didn't run initially

`pro/test_run_pro.sh` and `pro/run_missing_pro.sh` assume an **external checkout** exists at:

- `../SWE-bench_Pro-os` (or `$SWE_BENCH_PRO_DIR`)

They need it to generate / locate:

- `SWE-agent/data/instances.yaml`
- `helper_code/generate_sweagent_instances.py`

In this workspace that directory was missing, so the script failed early with `cd: ../SWE-bench_Pro-os: No such file or directory`.

## Code changes in this directory

### 1) `test_run_pro.sh`: add fallback smoke test when `SWE_BENCH_PRO_DIR` is missing

If `$SWE_BENCH_PRO_DIR` does **not** exist, we now run a minimal local smoke test to verify that:

- the `sweagent` CLI can start
- output directories are writable
- a trajectory + `preds.json` can be produced

Fallback uses:

- `tests/test_data/data_sources/simple_instances.yaml`
- `--agent.model.name instant_empty_submit`
- `--instances.deployment.type=dummy`

This is intentionally **not** a full SWE-bench Pro run; it only proves the runner works.

### 2) `run_missing_pro.sh`: fail fast with a clear error if `SWE_BENCH_PRO_DIR` is missing

We added an explicit directory existence check for `$SWE_BENCH_PRO_DIR` and print a helpful message to set it, e.g.:

```bash
export SWE_BENCH_PRO_DIR=/abs/path/to/SWE-bench_Pro-os
```

## Local filesystem / environment notes (not repo logic)

To make `sweagent` import cleanly, `sweagent/__init__.py` requires a trajectories directory to exist. We ensured:

- `Context-Bench/agent-frameworks/swe-agent/pro/trajectories/` exists (it can be empty)

Also, the Python environment used is:

- conda env `sweagent311` (Python 3.11)
- installed `swe-rex` (import name `swerex`)
- installed `httpx[socks]` (for environments where SOCKS proxy variables are set)
- installed this `pro` snapshot editable: `pip install -e .`

## How to run

### A) Smoke test (works without SWE-bench Pro checkout)

```bash
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/pro
conda run -n sweagent311 bash ./test_run_pro.sh
```

Outputs:

- `trajectories/smoke_test_pro/`
  - `<instance_id>/<instance_id>.traj`
  - `preds.json`

### B) Real Pro run (requires external checkout)

```bash
export SWE_BENCH_PRO_DIR=/abs/path/to/SWE-bench_Pro-os
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/pro
conda run -n sweagent311 bash ./run_missing_pro.sh
```

