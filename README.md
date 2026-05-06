<h1 align="center">AgentStackBench</h1>

<p align="center">
  <strong>Repeatable A/B testing for real coding-agent setups</strong>
</p>

<p align="center">
  Measure whether models, prompts, skills, MCPs, plugins, mounted resources, and runtime choices actually improve coding agents, and what they cost.
</p>

<p align="center">
  <a href="https://github.com/Nobbettt/AgentStackBench"><img alt="Repository" src="https://img.shields.io/badge/repo-AgentStackBench-111827"></a>
  <a href="https://nobbettt.github.io/AgentStackBench/"><img alt="Live frontend" src="https://img.shields.io/badge/live-frontend-16a34a"></a>
  <a href="https://github.com/EuniAI/ContextBench"><img alt="Using" src="https://img.shields.io/badge/Using-ContextBench-2563eb"></a>
</p>

<p align="center">
  <a href="#why-it-exists">Why</a> ·
  <a href="#what-agentstackbench-adds">What it adds</a> ·
  <a href="#using-contextbench">ContextBench</a> ·
  <a href="#metrics-for-evaluation">Metrics</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#frontend">Frontend</a>
</p>

---

AgentStackBench is using the official [ContextBench](https://github.com/EuniAI/ContextBench) methodology. It keeps the upstream evaluator and dataset-processing flow as the source of truth, then adds a fork-specific layer for running agents such as Codex and Claude, converting their outputs to ContextBench-compatible trajectories, and publishing distilled comparison results.

The Python import package is still `contextbench` for compatibility with the upstream evaluator path.

## Why It Exists

Coding-agent setups are getting more complex: instruction files, skills, MCP servers, plugins, bootstrap prompts, isolated runtimes, and model-specific effort settings can all change how an agent behaves. Most claims about these additions are hard to verify. They may improve issue resolution, or they may only increase token usage, tool calls, and noise.

AgentStackBench turns those changes into repeatable experiments. It compares agent configurations side by side and reports both correctness and efficiency: did the agent solve the task, did it retrieve the right context, how many tools did it use, and how many tokens did it spend getting there?

## What AgentStackBench Adds

| Area | Purpose |
| --- | --- |
| Real agent execution | Run Codex and Claude Code on ContextBench-derived tasks through local CLI adapters. |
| Configuration experiments | Compare models, prompts, skills, MCPs, plugins, resources, setup steps, and runtimes. |
| Durable run records | Preserve raw output, final answers, patches, traces, stderr, and workspace diffs. |
| Comparable metadata | Normalize token counts, execution time, tool calls, failures, setup results, and model/runtime settings. |
| ContextBench compatibility | Convert saved runs into ContextBench-compatible trajectories and predictions. |
| Correctness and context metrics | Score pass@1 plus file, symbol, span, line, and edit-location quality. |
| Native resolution routing | Use the native evaluators for Verified, Pro, Poly, and Multi benchmark sources. |
| Frontend reporting | Export sanitized summaries and per-instance traces for public comparison views. |

The Codex and Claude Code adapters execute the locally installed CLIs with the user's existing authentication. That means benchmark runs can use the same subscription or account setup already configured for those tools, instead of requiring a separate per-token API integration inside the harness.

## Using ContextBench

ContextBench provides the methodology AgentStackBench uses for context-aware coding-agent evaluation: record the agent's trajectory, convert that trajectory into comparable retrieval and prediction artifacts, then score both issue resolution and whether the agent found the expert-validated gold context needed to solve the task.

The underlying gold dataset covers **Instances: 1,136** issue-resolution tasks, **Repositories: 66**, and **Programming languages: 8**.

**AgentStackBench** extends that methodology into a practical evaluation harness for the kinds of coding-agent configurations developers actually change day to day. The goal is to answer questions like:

- Does adding a skill, MCP, plugin, or instruction file improve pass@1?
- Does it improve context retrieval, or just make the agent inspect more irrelevant code?
- Does a model or reasoning-effort change justify its token cost?
- Which setup produces the best tradeoff between correctness, context quality, tool use, and cost?

## Metrics For Evaluation

The frontend comparison page is built from normalized run artifacts, so each variant can be compared across the same concrete metrics.

### Execution Metrics

| Metric | Measures |
| --- | --- |
| Completed Run Rate | Share of attempted tasks whose agent run finished with `completed` status. |
| Patch Production Rate | Share of attempted tasks that produced a non-empty model patch. |
| Valid Evaluation Rate | Share of attempted tasks with a valid evaluator output row. |

### Resolution Metrics

| Metric | Measures |
| --- | --- |
| Pass@1 | Task-resolution rate from the benchmark-specific resolution harness. |
| Fix overlap F1 | Balanced overlap between model patch edit locations and gold patch edit locations. |
| Fix overlap recall | Share of gold patch edit locations covered by the model patch. |
| Fix overlap precision | Share of model patch edit locations that overlap gold edit locations. |

### Context Retrieval Metrics

| Metric | Measures |
| --- | --- |
| Context F1 | Combined file, symbol, and span retrieval F1. |
| File F1 | Whether the agent retrieved the gold files. |
| Symbol F1 | Whether the agent retrieved the gold definitions or symbols. |
| Span F1 | Whether the agent retrieved the gold code spans. |
| Avg. Line F1 | Whether the agent retrieved the gold line ranges. |

### Patch Overlap Metrics

| Metric | Measures |
| --- | --- |
| A covered by B | Share of one variant's edit locations also touched by the other variant. |
| B covered by A | Reverse edit-location coverage between the compared variants. |
| Overlap F1 | Balanced edit-location overlap between the two compared variants. |

### Resource Usage Metrics

| Metric | Measures |
| --- | --- |
| Average Steps | Average inferred retrieval steps per run. |
| Average Duration | Average wall-clock runtime per run. |
| Total Tokens | Total token usage across included runs. |
| Tool / MCP Calls | Total recorded tool or MCP telemetry events. |
| Execution Cost | Average per-run inference cost when cost metadata is available. |

### Trajectory Metrics

| Metric | Measures |
| --- | --- |
| Efficiency | How quickly the trajectory accumulates relevant gold context. |
| Redundancy | Repeated or overlapping retrieval work across inferred trajectory steps. |
| Usage Drop | Drop-off in useful context retrieval as the trajectory progresses. |
| Avg. Lines Per Step | Average inspected line volume per inferred retrieval step. |

### Skill And Tool Usage

| Metric | Measures |
| --- | --- |
| Skill Invocations / Run | Average detected skill file invocations per run. |
| Skill Breakdown | Per-skill average invocation counts. |
| Tool Calls / Run | Average recorded tool or MCP telemetry events per run. |
| Tool Breakdown | Per-tool or per-MCP average invocation counts. |

### Task And Instance Metadata

| Metric | Measures |
| --- | --- |
| Benchmark Source | Which benchmark family the task came from. |
| Language | Primary programming language for the task. |
| Original Instance ID | Upstream instance identifier before AgentStackBench export normalization. |
| Variant Configuration | Model, reasoning effort, runtime backend, setup steps, mounted resources, MCPs, plugins, and skills. |
| Artifact Availability | Whether patches, predictions, evaluation rows, resolution rows, raw traces, and detailed instance payloads are present. |

## Quickstart

Install Python dependencies and make sure the agent CLI you want to test is already logged in locally:

```bash
pip install -r requirements.txt
```

Create a run-suite config under `configs/run_suites/`. A config answers four basic questions:

| Field | Meaning |
| --- | --- |
| `experiment_name` | The output folder name under `results/run_suites/`. Use a new name for a new experiment. |
| `agent` | Which adapter to run, usually `codex` or `claude`. |
| `base_run` | Shared task selection and default runtime settings. Start with a small `limit`. |
| `variants` | The setups to compare. Keep one baseline and change one thing in the treatment. |

Use the existing configs in `configs/run_suites/` as starting points. The safest first edit is to copy one, give it a new `experiment_name`, set a small `base_run.limit`, and change only the treatment variant.

Run the suite with your config:

```bash
python3 -m contextbench.run_suites --config configs/run_suites/<your-config>.json
```

Resume an interrupted run:

```bash
python3 -m contextbench.run_suites --config configs/run_suites/<your-config>.json --resume --resume-resolution
```

For more setup options, including mounted files, runtime environment variables, MCP config, and benchmark-specific resolution setup, see [docs/run_suites.md](docs/run_suites.md).

## Frontend

The deployed comparison frontend is available at [nobbettt.github.io/AgentStackBench](https://nobbettt.github.io/AgentStackBench/).

The static frontend lives in `frontend/` and reads distilled public data from `site-data/`.

After a run suite has finished, export frontend-ready comparison data from `results/`:

```bash
python3 scripts/export_comparison_data.py --suite-dir results/run_suites/codex-superpowers-mounted
```

By default this writes `site-data/comparison.json` and per-instance detail payloads under `site-data/instances/`.

```bash
npm ci --prefix frontend
npm run build --prefix frontend
```

GitHub Pages is configured to build the frontend and nested Sphinx docs on pushes to `main`.

## Attribution

AgentStackBench builds on ContextBench:

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

The ContextBench dataset is available from Hugging Face at [`Contextbench/ContextBench`](https://huggingface.co/datasets/Contextbench/ContextBench).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
