# AgentStackBench

AgentStackBench is a benchmark harness for comparing coding-agent configurations: model, runtime, tools, mounted resources, prompts, and evaluator setup.

This repository is derived from the official ContextBench project and keeps the upstream evaluator and dataset-processing flow as the source of truth. The fork-specific layer adds thin support for running agents such as Codex and Claude, converting their outputs to ContextBench-compatible trajectories, and publishing distilled comparison results.

## Current Scope

- Run coding agents on ContextBench-derived SWE-style tasks.
- Convert saved agent records into ContextBench trajectory and prediction artifacts.
- Evaluate both context quality and issue-resolution success across SWE-Bench Verified, SWE-Bench Pro, SWE-PolyBench, and Multi-SWE-Bench sources.
- Publish sanitized, frontend-ready result summaries without exposing local paths or raw private run state.

The Python import package is still `contextbench` for compatibility with the upstream evaluator path.

## Quickstart

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the current Codex comparison suite:

```bash
python3 -m contextbench.run_suites --config configs/run_suites/codex-superpowers-bootstrap.json
```

Resume an interrupted run:

```bash
python3 -m contextbench.run_suites --config configs/run_suites/codex-superpowers-bootstrap.json --resume --resume-resolution
```

Run the all-benchmark smoke config:

```bash
python3 -m contextbench.run_suites --config configs/run_suites/codex-superpowers-bootstrap-5-all-benches-smoke.json
```

## Frontend

The static frontend is in `frontend/` and reads distilled public data from `site-data/`.

```bash
npm ci --prefix frontend
npm run build --prefix frontend
```

GitHub Pages is configured to build the frontend and nested Sphinx docs on pushes to `main`.

## Repository Layout

```text
AgentStackBench/
├── contextbench/                # Upstream-compatible Python package and forked run-suite helpers
├── configs/run_suites/          # Reusable benchmark run configs
├── data/                        # Benchmark datasets and selected instance lists
├── docker/                      # Runtime and postprocess Dockerfiles
├── frontend/                    # React frontend for distilled results
├── results/                     # Committed public artifacts only
├── scripts/                     # Export and utility scripts
├── site-data/                   # Frontend-ready comparison payloads
└── tests/                       # Unit and regression tests
```

## Attribution

AgentStackBench is derived from ContextBench:

```bibtex
@misc{li2026contextbenchbenchmarkcontextretrieval,
  title={ContextBench: A Benchmark for Context Retrieval in Coding Agents},
  author={Han Li and Letian Zhu and Bohan Zhang and Rili Feng and Jiaming Wang and Yue Pan and Earl T. Barr and Federica Sarro and Zhaoyang Chu and He Ye},
  year={2026},
  eprint={2602.05892},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.05892}
}
```

The ContextBench dataset is available from Hugging Face at `Contextbench/ContextBench`.

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE).
