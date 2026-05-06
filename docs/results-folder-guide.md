<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interpreting `results/`

This guide explains the artifacts written by the local agent runner under `results/agent_runs/`, especially for the new `codex` and `claude` wrapper integrations.

## Folder layout

For local agent runs, artifacts are written under:

```text
results/
  agent_runs/
    <agent>/
      <bench>/
        <instance_id>/
```

Example:

```text
results/agent_runs/codex/Verified/SWE-Bench-Verified__python__maintenance__bugfix__27320d49/
```

The main levels mean:

- `agent_runs/`: raw task-level run artifacts from `python -m contextbench.run`
- `<agent>/`: one agent family, for example `codex` or `claude`
- `<bench>/`: one benchmark family, for example `Verified`, `Pro`, `Poly`, or `Multi`
- `<instance_id>/`: one concrete benchmark task

## Files inside one task directory

Typical task directory contents:

```text
<instance_id>.codex-record.json or <instance_id>.claude-record.json
setup-prompt.txt         # Optional: unscored setup prompt
setup-raw-response.json  # Optional: wrapped setup response
setup-stderr.log         # Optional: setup stderr
setup-codex-events.jsonl # Optional: Codex setup event stream
setup-last-message.txt   # Optional: Codex setup last message
raw-response.json
codex-events.jsonl        # Codex only
final-output.json         # Codex only
prompt.txt
workspace.diff
stderr.log
codex-runtime/   # Codex only
```

### `prompt.txt`

The exact prompt sent to the agent wrapper for this task.

Use this when you want to answer:

- what issue text the model saw
- what benchmark instructions were given
- whether the prompt itself explains a surprising output

If the run used an unscored setup phase, `setup-prompt.txt` contains that bootstrap prompt and `prompt.txt` still contains the scored benchmark prompt.

### `raw-response.json`

The source-of-truth saved response artifact for the wrapper integration.

This is the most important file if you want to understand what the CLI actually returned.

For Codex it contains:

- `agent`
- `response_format: "jsonl-events"`
- `events`: parsed JSONL event stream
- `final_message`: the schema-constrained final model message

For Claude it contains:

- `agent`
- `response_format: "json"`
- `response`: the full verbose JSON response envelope returned by Claude Code

Use this when you want to inspect:

- token usage
- tool calls
- the actual raw response structure
- future parser regressions after CLI updates

If the run used a setup prompt, `setup-raw-response.json` is the equivalent wrapped artifact for the unscored bootstrap phase.

### `final-output.json`

Codex only.

The extracted benchmark-facing structured output object written by the Codex wrapper.

This is the cleaned summary derived from `raw-response.json` and used to build the evaluation trajectory.

It answers:

- what files the agent says it touched
- what retrieval steps it reported
- what final context it says it relied on
- what final textual answer it returned

This is easier to read than `raw-response.json`, but it is not the authoritative raw artifact.

### `codex-events.jsonl`

Codex only.

This is the direct Codex CLI event stream before normalization.

Claude does not currently keep a separate pre-normalization file alongside `raw-response.json`; its verbose CLI JSON is read and then wrapped into `raw-response.json`.

These direct-output artifacts are useful mainly for debugging wrapper behavior. In normal analysis, prefer `raw-response.json`.

When a setup prompt is enabled for Codex, `setup-codex-events.jsonl` is the corresponding pre-normalization event stream for that bootstrap phase.

### `<instance_id>.<agent>-record.json`

This is the normalized per-task run record that the rest of the ContextBench adapter stack consumes.

Important fields:

- `agent`, `bench`, `instance_id`, `original_inst_id`
- `repo`, `repo_url`, `commit`
- `workspace_path`
- `status`, `ok`, `exit_code`, `timeout`, `duration_ms`
- `final_output`
- `token_usage`
- `tool_calls`
- `raw_response_path`
- `diff_path`
- `model_patch`
- `setup_run` (optional separate metadata for the unscored setup phase)

Use this file when you want the best “single summary object” for one run.

It is the bridge between:

- raw CLI artifacts
- ContextBench trajectory conversion

`setup_run` keeps separate bootstrap metadata such as status, duration, raw response path, token usage, and tool calls. These setup-phase stats are intentionally excluded from the main `token_usage` and `tool_calls` fields.

### `workspace.diff`

The final `git diff` from the checked-out task workspace after the agent run.

Use this when you want to inspect:

- what code actually changed
- whether the agent made any changes at all
- what edit spans may be inferred by the adapter

If this file is absent or empty, the agent did not produce a non-empty diff.

### `stderr.log`

Anything the CLI wrote to stderr.

This may be empty even for valid runs.

Use it mainly for:

- auth problems
- wrapper failures
- CLI startup/runtime errors

If present, `setup-stderr.log` serves the same role for the unscored setup phase.

### `codex-runtime/`

Codex only.

This is the isolated runtime environment created for the run. It includes a temporary:

- `HOME`
- `XDG_CONFIG_HOME`
- `XDG_DATA_HOME`
- `XDG_CACHE_HOME`

Its purpose is to isolate benchmark runs from the normal user environment while still supplying the minimum auth material required to run Codex.

You usually do not need to inspect this unless you are debugging auth/config isolation.

## Which file should I trust first?

Use this order:

1. `raw-response.json`
2. `<instance_id>.<agent>-record.json`
3. `final-output.json`
4. `workspace.diff`
5. `stderr.log`

Interpretation rule:

- `raw-response.json` tells you what the CLI actually returned
- the record file tells you how the harness interpreted that response
- `final-output.json` tells you the extracted benchmark summary
- `workspace.diff` tells you what changed in the repo

## Common questions

### Why do file paths look absolute and long?

The wrappers record paths from the checked-out benchmark workspace, which currently lives in a temp worktree. That is why you will often see absolute paths under `/var/.../contextbench_worktrees/...`.

This is expected.

For run suites, successful fan-out workspaces may be deleted after the task finishes. In that case `workspace_path` still records where the run executed, but the directory may no longer exist on disk.

### Why is `stderr.log` empty?

That is normal. Many successful runs write nothing to stderr.

### Why do `final-output.json` and `workspace.diff` disagree?

They measure different things:

- `final-output.json` is self-reported benchmark summary data from the model
- `workspace.diff` is the actual git diff in the workspace

The adapter may use both, but the diff is the authoritative source for code edits.

### Why is `tool_calls` empty even when the model looked at files?

`tool_calls` currently tracks parsed runtime/tool telemetry from the raw response, not every conceptual retrieval step the model reported. Retrieval steps live under `final_output.retrieval_steps`.

### Why does a run have `ok: true` but still limited verification?

`ok` means the CLI invocation itself succeeded.

It does not guarantee:

- unit tests ran successfully
- dependencies were installed
- runtime imports worked

Those details are usually described in `final_output.final_answer` and sometimes in `stderr.log`.

## How these results feed evaluation

The task directory itself is not the final evaluation output.

The normal flow is:

1. `contextbench.run` writes task artifacts under `results/agent_runs/...`
2. `contextbench.process_trajectories convert` reads the saved record files
3. It emits a unified prediction JSONL
4. `contextbench.evaluate` scores that prediction JSONL against the gold parquet

So this folder is best understood as:

- execution artifacts
- debug artifacts
- adapter inputs

not the final score report itself.
