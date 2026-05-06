# run_batch_eval.sh Usage Guide

Unified entry script for running SWE-PolyBench and/or SWE-bench Pro batch evaluation. The script lives in the `poly-pro/` root and internally invokes `run_polybench_batch.sh` and `run_probench_batch.sh` under `evaluation/benchmarks/swe_bench/`.

## Quick Start

```bash
# Run from poly-pro root
cd agent-frameworks/openhands/poly-pro

# Run both PolyBench and Pro with defaults
./run_batch_eval.sh

# PolyBench only
./run_batch_eval.sh poly

# SWE-bench Pro only
./run_batch_eval.sh pro
```

## Command Line Usage

```bash
./run_batch_eval.sh [OPTIONS] [BENCH] [batch_size] [start_batch] [end_batch] [max_concurrent_batches]
```

### Options (OPTIONS)

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --poly-instance-file FILE` | Instance ID list file for PolyBench | `evaluation/benchmarks/swe_bench/poly_instance_ids.txt` |
| `-I, --pro-instance-file FILE` | Instance ID list file for SWE-bench Pro | `evaluation/benchmarks/swe_bench/pro_instance_ids.txt` |
| `-o, --output-dir DIR` | Base output directory; creates `poly/` and `pro/` subdirs | `result/` (relative to poly-pro root) |
| `--no-cleanup` | Keep batch-generated temporary .toml configs (for debugging) | Configs cleaned up by default |
| `-h, --help` | Show help | - |

### BENCH Mode

| Value | Description |
|-------|-------------|
| `both` | Run PolyBench then Pro sequentially (default) |
| `poly` | Run SWE-PolyBench only |
| `pro` | Run SWE-bench Pro only |

### Positional Arguments (passed to underlying batch scripts)

| Argument | Description | Default |
|----------|-------------|---------|
| `batch_size` | Instances per batch | 10 |
| `start_batch` | Starting batch number (1-based) | 1 |
| `end_batch` | Ending batch number (0 = all batches) | 0 |
| `max_concurrent_batches` | Maximum concurrent batches | 3 |

## Usage Examples

### 1. Default run (both benchmarks, default instance files)

```bash
./run_batch_eval.sh
```

### 2. Specify benchmark type

```bash
# PolyBench only
./run_batch_eval.sh poly

# Pro only
./run_batch_eval.sh pro
```

### 3. Custom instance files

```bash
# Specify both PolyBench and Pro instance files
./run_batch_eval.sh -i ./my_poly_instances.txt -I ./my_pro_instances.txt both

# Custom list for PolyBench only
./run_batch_eval.sh -i ./selected_poly.txt poly
```

### 4. Custom output directory

```bash
./run_batch_eval.sh -o ./eval_results both
# Results go to: ./eval_results/poly/ and ./eval_results/pro/
```

### 5. Batch parameters

```bash
# 15 per batch, batches 1–5, max 4 concurrent
./run_batch_eval.sh poly 15 1 5 4
```

### 6. Combined options

```bash
# Custom instance files, output dir, batch params, keep configs for debugging
./run_batch_eval.sh -i poly.txt -I pro.txt -o ./outputs --no-cleanup both 20 1 3 2
```

### 7. Extract instances from CSV and run

If instances come from a CSV (e.g. `selected_500_instances.csv`), split by the `bench` column:

```bash
# Extract PolyBench instances (bench column = Poly)
awk -F',' 'NR>1 && $1=="Poly" {print $3}' selected_500_instances.csv > poly_instances.txt

# Extract Pro instances (bench column = Pro)
awk -F',' 'NR>1 && $1=="Pro" {print $3}' selected_500_instances.csv > pro_instances.txt

# Run with generated lists
./run_batch_eval.sh -i poly_instances.txt -I pro_instances.txt both
```

## Output Structure

With default or `-o DIR`, the directory layout is:

```
result/                    # or the directory given with -o
├── poly/
│   ├── AmazonScience__SWE-PolyBench-test/
│   │   └── CodeActAgent/.../output.jsonl
│   ├── batch_1.log
│   └── batch_*.status
└── pro/
    ├── ScaleAI__SWE-bench_Pro-test/
    │   └── CodeActAgent/.../output.jsonl
    ├── batch_1.log
    └── batch_*.status
```

## Relationship to Underlying Scripts

- `run_batch_eval.sh` is the **unified entry** in the poly-pro root.
- Execution is done by `run_polybench_batch.sh` and `run_probench_batch.sh`.
- To change MODEL_CONFIG, AGENT, etc., edit those scripts in `evaluation/benchmarks/swe_bench/`.

For details on the underlying scripts, see `evaluation/benchmarks/swe_bench/BATCH_SCRIPTS_USAGE.md`.

## FAQ

**Q: How do I evaluate only a subset of instances from a CSV?**  
A: Extract instance IDs by `bench` and `instance_id` columns into `poly_instances.txt` and `pro_instances.txt`, then pass them with `-i` and `-I`.

**Q: Does it run PolyBench first, then Pro?**  
A: Yes. In `both` mode, PolyBench runs to completion first, then Pro runs.

**Q: How do I change the model or Agent for each benchmark?**  
A: Edit the `MODEL_CONFIG`, `AGENT`, and related variables in `run_polybench_batch.sh` and `run_probench_batch.sh`.

**Q: Does a PolyBench failure affect Pro?**  
A: Pro still runs. The script exit code reflects whether any benchmark failed.
