## Overview

This document explains how to run SWE-bench Verified evaluations using the helper scripts in this `verified` directory:

- `run_verified.sh`: unified entry point to launch OpenHands evaluations.
- `evaluate_results.sh`: helper to evaluate and inspect completed runs.

All examples below assume you are in the `verified` directory (or that `OPENHANDS_RUN_DIR` points here).

---

## 1. Prerequisites

- **Python environment**: the OpenHands project and its Python dependencies must be installed (see the main project documentation).
- **Config file**: copy and edit `config.template.toml` into `config.toml`:

```bash
cp config.template.toml config.toml
```

Then open `config.toml` and:

- Set a valid API key for your chosen model provider.
- Adjust any other settings as needed (model choice, logging, etc.).

> `run_verified.sh` will refuse to start if `config.toml` still contains a placeholder API key.

---

## 2. Running SWE-bench Verified with `run_verified.sh`

### 2.1 Basic usage

The main script is:

```bash
./run_verified.sh [options]
```

If you run it with **no arguments**, it will:

- Use the default dataset `princeton-nlp/SWE-bench_Verified`.
- Run **174 verified instances**.
- Use:
  - Agent: `CodeActAgent`
  - Model: `llm.eval_gpt5`
  - Max iterations: `100`
  - Concurrency: `4`

Example:

```bash
./run_verified.sh
```

You will see a configuration summary and will be asked to confirm before the run starts.

### 2.2 Preset-based runs

You can use the `--preset` flag to quickly choose common evaluation sizes:

- `quick`   – 10 instances (fast sanity check)
- `medium`  – 50 instances (medium-scale run)
- `verified` – 174 verified instances (default / recommended)
- `full`    – 500 instances (full SWE-bench Verified)

Examples:

```bash
# 10 instances, quick check
./run_verified.sh --preset quick

# 50 instances, medium scale
./run_verified.sh --preset medium

# 500 instances (full benchmark)
./run_verified.sh --preset full
```

You can also skip the confirmation prompt (useful for automation/CI):

```bash
./run_verified.sh --preset verified --no-confirm
```

### 2.3 Custom instance count and other parameters

You can override the number of instances and other settings:

- `--limit N` – explicitly set the number of instances (overrides `--preset`).
- `--concurrency N` – number of parallel workers (default `4`).
- `--max-iters N` – maximum iterations per instance (default `100`).
- `--agent NAME` – agent name (default `CodeActAgent`).
- `--model NAME` – model identifier (default `llm.eval_gpt5`).
- `--dataset NAME` – dataset name (default `princeton-nlp/SWE-bench_Verified`).

Example:

```bash
./run_verified.sh \
  --limit 30 \
  --concurrency 8 \
  --max-iters 200 \
  --agent CodeActAgent \
  --model llm.eval_gpt5
```

---

## 3. Running a custom list of instances (`--instance-file`)

Often you want to evaluate only a **specific set of instance IDs**, for example:

- A custom subset from `selected_500_instances.csv`.
- The 174 official verified instances (`all_verified_174.txt`).
- Any ad-hoc list you define.

`run_verified.sh` supports this via the `--instance-file` option.

### 3.1 Instance file format

The instance file should be a **plain text file**, with **one `instance_id` per line**, for example:

```text
django__django-12345
scikit-learn__scikit-learn-67890
...
```

You can re-use existing files in this directory, such as:

- `all_verified_174.txt` – the official 174 verified instances.
- A custom file you create from `selected_500_instances.csv`.

### 3.2 How it is used internally

When you pass `--instance-file` for the default dataset `princeton-nlp/SWE-bench_Verified`, the script will:

1. Read all non-empty lines from the instance file and de-duplicate them (preserving order).
2. Write these IDs into `evaluation/benchmarks/swe_bench/config.toml` as:
   - `selected_ids = [ 'id1', 'id2', ... ]`
3. Set `LIMIT` to the number of IDs in the file (this **overrides** `--preset` and `--limit`).
4. Run `evaluation/benchmarks/swe_bench/scripts/run_infer.sh` with that configuration.

This means the evaluation will be **strictly limited** to the IDs in your instance file.

### 3.3 Example: run exactly the 174 verified instances

```bash
./run_verified.sh --instance-file all_verified_174.txt
```

This will:

- Load IDs from `all_verified_174.txt`.
- Write them into `evaluation/benchmarks/swe_bench/config.toml` as `selected_ids`.
- Run evaluation with `LIMIT` equal to the number of IDs in the file.

### 3.4 Example: run a custom subset from `selected_500_instances.csv`

You can create your own subset file (for example, 20 specific instances), then pass it:

```bash
# Example: create a file with some instance_ids from selected_500_instances.csv
cut -d',' -f2 selected_500_instances.csv | head -n 20 > my_instances.txt

# Run only these 20 instances
./run_verified.sh --instance-file my_instances.txt
```

As long as the instance IDs are valid SWE-bench IDs and present in the dataset, they will be evaluated.

---

## 4. Evaluating results with `evaluate_results.sh`

After a run finishes, outputs are written under:

```text
evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/
```

Each evaluation run produces (among other files) an `output.jsonl` file. To evaluate and inspect results:

```bash
./evaluate_results.sh
```

The script will:

1. Search under the default output directory for all `output.jsonl` files.
2. Show an interactive menu where you can pick one file.
3. Print basic file info (size, number of instances).
4. Call:
   - `./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh <selected_output.jsonl>`
5. If present, pretty-print `report.json` and print the path to the corresponding `README.md` report.

This gives you a quick way to:

- Evaluate a specific output run.
- Inspect the high-level metrics and a more detailed report.

---

## 5. Typical workflows

### 5.1 Quick sanity check

```bash
./run_verified.sh --preset quick
./evaluate_results.sh
```

### 5.2 Full 174 verified instances

```bash
./run_verified.sh --instance-file all_verified_174.txt
./evaluate_results.sh
```

### 5.3 Custom subset from `selected_500_instances.csv`

```bash
cut -d',' -f2 selected_500_instances.csv | grep -v '^$' | head -n 50 > my_50_instances.txt
./run_verified.sh --instance-file my_50_instances.txt
./evaluate_results.sh
```

These workflows should cover most common SWE-bench Verified evaluation scenarios using the helper scripts in this directory.

