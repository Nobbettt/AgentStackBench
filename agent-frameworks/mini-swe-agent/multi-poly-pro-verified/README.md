# mini-SWE-agent Extensions

> **multi-poly-pro-verified** — Modified files on top of [mini-SWE-agent](https://github.com/princeton-nlp/mini-swe-agent), extracted for review and customization. These extensions are **already synced** into the `mini-swe-agent/` sub-repo. To apply them to your own fork, copy the corresponding files from this directory to the paths in the table below.

---

## Table of Contents

- [File Mapping](#file-mapping)
- [Directory Structure](#directory-structure)
- [Main Components](#main-components)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Notes & Links](#notes--links)

---

## File Mapping

Extension files in this directory map to the following paths **inside the mini-swe-agent repo**:

| This directory | Path inside `mini-swe-agent/` |
|----------------|-------------------------------|
| `agents/context_aware.py` | `src/minisweagent/agents/context_aware.py` |
| `configs/swebench_context_aware.yaml` | `src/minisweagent/config/extra/swebench_context_aware.yaml` |
| `configs/swebench_following_context.yaml` | `src/minisweagent/config/extra/swebench_following_context.yaml` |
| `configs/swebench_multi.yaml` | `src/minisweagent/config/extra/swebench_multi.yaml` |
| `run/swebench_context_aware.py` | `src/minisweagent/run/extra/swebench_context_aware.py` |

**Customization:**

- **Agent behavior** → Overwrite `src/minisweagent/agents/context_aware.py`
- **Configs** → Overwrite YAMLs under `src/minisweagent/config/extra/`, or point `--config` to this repo’s `configs/`
- **Run entrypoint** → After overwriting the run script, invoke:  
  `python -m minisweagent.run.extra.swebench_context_aware`

---

## Directory Structure

```
multi-poly-pro-verified/
├── agents/
│   └── context_aware.py          # Context-Aware Agent
├── configs/
│   ├── swebench_context_aware.yaml
│   ├── swebench_following_context.yaml
│   └── swebench_multi.yaml
├── run/
│   └── swebench_context_aware.py  # Batch run entry
├── mini-swe-agent/                # Upstream repo (full code & docs)
│   ├── src/minisweagent/
│   ├── docs/
│   ├── tests/
│   └── ...
└── README.md
```

---

## Main Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Context-Aware Agent** | `agents/context_aware.py` | Asks the agent to provide code context before submitting a patch for easier analysis and reproduction. Config: `configs/swebench_context_aware.yaml`. |
| **Following-Context Agent** | `configs/swebench_following_context.yaml` | Uses markers such as `<EXPLORE_CONTEXT>` to track code exploration during the run. |
| **Multi-Bench Config** | `configs/swebench_multi.yaml` | Configuration tuned for running on Multi-SWE-bench, used by ContextBench `run_miniswe` for the `Multi` bench. |
| **Batch Run Script** | `run/swebench_context_aware.py` | Supports SWE-bench (lite/verified), SWE-bench Pro, Multi-SWE-bench, PolyBench. Instance selection via `--subset`, `--filter`, `--slice`; parallel runs via `--workers`. |

---

## Quick Start

From this directory (ensure `minisweagent` is importable):

```bash
python run/swebench_context_aware.py main \
  --subset verified \
  --split test \
  --config configs/swebench_context_aware.yaml \
  --model openai/gpt-4o \
  --output ./output \
  --workers 4
```

Or via the installed package:

```bash
python -m minisweagent.run.extra.swebench_context_aware main \
  --subset verified \
  --split test \
  --config configs/swebench_context_aware.yaml \
  --model openai/gpt-4o \
  --output ./output
```

---

## Usage

### Environment and Dependencies

Install mini-SWE-agent and its dependencies (from `mini-swe-agent/` or via `PYTHONPATH`). Configure your model API (e.g. OpenAI/Anthropic) and Docker (for SWE-bench environments).

### Common Options

| Option | Description |
|--------|-------------|
| `--subset` | Dataset subset or path: `lite`, `verified`, `pro`, `multi-swe-bench`, or a local path |
| `--split` | Split name: `dev`, `test` |
| `--filter` | Regex to filter instance IDs |
| `--slice` | Slice spec, e.g. `0:5` for the first 5 instances |
| `-c` / `--config` | Config file (e.g. `configs/swebench_context_aware.yaml`) |
| `-o` / `--output` | Output directory for results |
| `-w` / `--workers` | Number of parallel workers |
| `--redo-existing` | Re-run instances that already have results |

---

## Configuration

- **Timeouts, limits**: Set in YAML configs (`timeout`, `step_limit`, `cost_limit`, etc.).
- **Environment**: Multi-language / env vars (e.g. `PIP_PROGRESS_BAR`, `NPM_CONFIG_LOGLEVEL`, `MAVEN_OPTS`) can be set as needed.

---

## Notes & Links

**Requirements:**

1. **Docker** — SWE-bench runs require Docker and sufficient disk space for image pulls.
2. **API** — Configure the API key for your chosen model (OpenAI, Anthropic, etc.).
3. **Layout** — Core logic follows `mini-swe-agent/`; the top-level `agents/` and `configs/` here are extensions and local overrides.

**References:**

- [mini-SWE-agent](https://github.com/princeton-nlp/mini-swe-agent) (upstream)
- [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified)
- [SWE-PolyBench](https://huggingface.co/datasets/AmazonScience/SWE-PolyBench)

**License:** Same as the original mini-SWE-agent project.
