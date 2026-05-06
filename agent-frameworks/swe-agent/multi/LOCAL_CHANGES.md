# Local changes in `agent-frameworks/swe-agent/multi`

This document records what was changed / installed to make `Context-Bench/agent-frameworks/swe-agent/multi/` runnable in this workspace.

## Repo code changes under `multi/`

- **No source code was modified** under `multi/` for the smoke test.
- We only installed missing Python dependencies into the local Python environment so the runner/tests can import and start.

## Environment changes (outside the repo)

All changes below were applied to the conda environment:

- **conda env**: `sweagent311` (Python 3.11)

Missing runtime deps encountered and installed:

- **`swebench==1.0.1`**
  - Symptom: `ModuleNotFoundError: No module named 'swebench'`
  - Fix:
    ```bash
    conda run -n sweagent311 python -m pip install -U swebench==1.0.1
    ```
- **`dataclasses-json`**
  - Symptom: `ModuleNotFoundError: No module named 'dataclasses_json'` (imported by `multi_swe_bench` harness)
  - Fix:
    ```bash
    conda run -n sweagent311 python -m pip install -U dataclasses-json
    ```

Optional (if you want to align with the snapshot’s dependency set):

- Install `multi/requirements.txt`:
  ```bash
  cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/multi
  conda run -n sweagent311 python -m pip install -r requirements.txt
  ```

## How we validated "it runs"

We ran a minimal smoke test that only checks the CLI wiring can start (no docker build, no model calls):

```bash
cd /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/swe-agent/multi
conda run -n sweagent311 python -m pytest -q tests/test_run.py::test_run_cli_help
```

Expected output: `1 passed` (warnings are OK).

## Notes

- This `multi/` snapshot declares `requires-python = ">=3.9"` in its `pyproject.toml`, so Python 3.11 is compatible.
- `run_single_test.py` contains hardcoded paths (`/home/lih/...`) and is **not** portable as-is; prefer running `run.py` / `multirun.py` following `README.md`.

