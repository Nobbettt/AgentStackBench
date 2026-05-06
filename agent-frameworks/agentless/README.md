# Cagentless (Context Bench)

Cagentless is the Context Bench team’s multi-benchmark evaluation system. It is built on top of Agentless and MagentLess, with architectural optimizations and a unified pipeline to run multiple benchmarks through the same interface.

## What We Changed / Optimized

- **Unified multi-bench runner**: `run_bench.py` provides a single CLI entry for all benchmarks.
- **Data adapter layer**: `bench_sources.py` loads CSV/Parquet/JSONL sources into a standard single‑instance JSONL format.
- **Language inference**: infer repository language from file extensions via majority heuristic.
- **Per‑instance isolation**: each instance gets its own `input_data/{lang}/one.jsonl` under `results/{bench}/{idx}_{instance_id}/`.
- **Repo cloning fallback**: automatically clone missing repositories when local cache is not available.

## Data & Setup

### Folder layout (Cagentless and data are siblings)

```
/workspace
├── Cagentless/
│   ├── run_bench.py
│   ├── bench_sources.py
│   ├── script/
│   └── results/
└── data/
    ├── Multi.csv
    ├── Poly.csv
    ├── Pro.csv
    ├── Verified.csv
    ├── Multi/
    ├── Poly/
    ├── Pro/
    └── Verified/
```

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare CSV task lists

Ensure `data/` contains the following CSVs:

- `Multi.csv`
- `Poly.csv`
- `Pro.csv`
- `Verified.csv`

### 3) Clone datasets and rename folders

Clone the datasets:

```bash
git clone https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro
git clone https://huggingface.co/datasets/AmazonScience/SWE-PolyBench_Verified
git clone https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
git clone https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench
```

Rename the folders to match the bench identifiers under `data/`:

```bash
mv SWE-bench_Pro data/Pro
mv SWE-PolyBench_Verified data/Poly
mv SWE-bench_Verified data/Verified
mv Multi-SWE-bench data/Multi
```

## Configuration (API Keys)

Edit `script/api_key.sh` and set:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`
- (optional) embedding base URL if you use a custom endpoint

## Usage

Unified interface:

```bash
python run_bench.py {bench_name} --limit N
```

- `bench_name`: `Multi | Poly | Pro | Verified`
- `--limit N`: optional; run only the first N instances

### Example (smoke test)

```bash
python run_bench.py Multi --limit 1
python run_bench.py Poly --limit 1
python run_bench.py Pro --limit 1
python run_bench.py Verified --limit 1
```

## Outputs

Results are stored in:

```
results/{bench}/{idx}_{instance_id}/
```

Each instance folder includes:

- `input_data/{lang}/one.jsonl`
- localization outputs
- retrieval outputs
- repair samples
- selection outputs

## Notes / Troubleshooting

- Network access is required for repo cloning unless repositories are already cached locally.
- Language inference is based on file extensions; mixed-language repos may require manual override.
