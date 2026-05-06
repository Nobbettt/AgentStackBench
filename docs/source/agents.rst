.. SPDX-License-Identifier: Apache-2.0
.. Fork note: Modified by Norbert Laszlo on 2026-03-16 from upstream ContextBench.
.. Summary of changes: document fork-specific Codex CLI and Claude Code CLI trajectory extractors.

Agent Trajectory Extractors
============================

ContextBench includes trajectory extractors for different coding agents, exposed via a unified API.

Supported Agents
----------------

.. toctree::
   :maxdepth: 1

   agents/codex
   agents/claude

1. MiniSWE-agent
~~~~~~~~~~~~~~~~

- **Format**: ``.traj.json`` files
- **Location**: ``contextbench/agents/minisweagent/extract.py``

**Features**:

- Extracts file views from bash commands in messages
- Supports ``cat``, ``sed -n``, ``head``, ``grep``, ``nl | sed`` commands
- Parses ``patch_context_data.patch_context`` for final context
- Returns model patch from ``info.submission``

2. SWE-agent
~~~~~~~~~~~~

- **Format**: ``.checkpoints.jsonl`` files
- **Location**: ``contextbench/agents/sweagent/extract.py``

**Features**:

- Extracts from ``str_replace_editor view`` commands with ``--view_range``
- Only includes steps with explicit line ranges
- Parses ``patch_context`` string format (``File:/Lines:``)

3. Codex CLI
~~~~~~~~

- **Format**: ``*.codex-record.json``
- **Location**: ``contextbench/agents/codex/extract.py``

**Features**:

- Extracts from the wrapper-produced Codex CLI record file
- Uses reported retrieval steps when present
- Falls back to touched files and diff-derived spans

4. Claude Code CLI
~~~~~~~~~

- **Format**: ``*.claude-record.json``
- **Location**: ``contextbench/agents/claude/extract.py``

**Features**:

- Extracts from the wrapper-produced Claude Code CLI record file
- Uses reported retrieval steps when present
- Falls back to touched files and diff-derived spans

5. Agentless
~~~~~~~~~~~~

- **Format**: Custom JSON format
- **Location**: ``contextbench/agents/agentless/extract.py``

**Features**:

- Extracts localization and repair steps
- Parses file access from retrieval outputs
- Supports multi-stage reasoning

6. OpenHands
~~~~~~~~~~~~

- **Format**: Trajectory logs
- **Location**: ``contextbench/agents/openhands/extract.py``

**Features**:

- Extracts file operations from action logs
- Supports browsing and editing actions
- Handles multi-file contexts

7. Prometheus
~~~~~~~~~~~~~

- **Format**: Agent-specific format
- **Location**: ``contextbench/agents/prometheus/extract.py``

**Features**:

- Extracts context from reasoning traces
- Supports iterative refinement steps

Unified Interface
-----------------

All agent extractors use a unified interface:

.. code-block:: python

   from contextbench.agents import extract_trajectory

   # Automatically detects format based on file extension
   result = extract_trajectory("path/to/trajectory.traj.json")
   result = extract_trajectory("path/to/trajectory.checkpoints.jsonl")

The extractor returns a unified structure:

.. code-block:: python

   {
       "pred_steps": [
           {"files": [...], "spans": {...}},
           ...
       ],
       "pred_files": [...],
       "pred_spans": {...},
       "pred_patch": "...",  # Optional: model-generated patch
   }

Return Structure
----------------

pred_steps
~~~~~~~~~~

List of per-step context:

.. code-block:: python

   "pred_steps": [
       {
           "files": ["src/utils.py", "src/main.py"],
           "spans": {
               "src/utils.py": [(0, 100), (200, 300)],
               "src/main.py": [(0, 500)]
           }
       },
       ...
   ]

pred_files
~~~~~~~~~~

Cumulative set of all viewed files:

.. code-block:: python

   "pred_files": ["src/utils.py", "src/main.py", "tests/test.py"]

pred_spans
~~~~~~~~~~

Cumulative union of all viewed spans:

.. code-block:: python

   "pred_spans": {
       "src/utils.py": [(0, 100), (200, 300)],
       "src/main.py": [(0, 500)],
       "tests/test.py": [(0, 1000)]
   }

pred_patch
~~~~~~~~~~

Optional: The model-generated patch (if available):

.. code-block:: python

   "pred_patch": "diff --git a/src/utils.py b/src/utils.py\n..."

Adding a New Agent
------------------

To add support for a new agent:

1. Create extractor module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``contextbench/agents/myagent/extract.py``:

.. code-block:: python

   def extract_trajectory(traj_path: str) -> dict:
       """
       Extract trajectory from MyAgent format.
       
       Args:
           traj_path: Path to trajectory file
           
       Returns:
           dict with keys: pred_steps, pred_files, pred_spans
       """
       # Parse trajectory file
       with open(traj_path) as f:
           traj = json.load(f)
       
       # Extract per-step context
       steps = []
       for step in traj["steps"]:
           files, spans = parse_step(step)
           steps.append({"files": files, "spans": spans})
       
       # Compute cumulative context
       all_files = compute_union_files(steps)
       all_spans = compute_union_spans(steps)
       
       return {
           "pred_steps": steps,
           "pred_files": all_files,
           "pred_spans": all_spans,
       }

2. Register in dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``contextbench/agents/__init__.py``:

.. code-block:: python

   from contextbench.agents.myagent.extract import extract_trajectory as extract_myagent

   def extract_trajectory(traj_path: str) -> dict:
       if traj_path.endswith(".myagent.json"):
           return extract_myagent(traj_path)
       # ... other formats

3. Add tests
~~~~~~~~~~~~

Create ``tests/test_myagent_extractor.py``:

.. code-block:: python

   def test_myagent_extraction():
       result = extract_trajectory("test_data/myagent.json")
       assert "pred_files" in result
       assert "pred_spans" in result
       assert len(result["pred_steps"]) > 0

Testing Extractors
------------------

Test an extractor on a single trajectory:

.. code-block:: bash

   python -m contextbench.evaluate \
       --gold data/full.parquet \
       --pred traj_verified-mini/instance/instance.traj.json \
       --out results.jsonl

Check the extracted context:

.. code-block:: python

   from contextbench.agents import extract_trajectory
   
   result = extract_trajectory("path/to/traj.json")
   print(f"Files: {result['pred_files']}")
   print(f"Steps: {len(result['pred_steps'])}")
   print(f"Spans: {result['pred_spans']}")

Common Issues
-------------

Missing line ranges
~~~~~~~~~~~~~~~~~~~

Some trajectories don't include explicit line ranges. In this case:

- Extract full file content as spans
- Or skip steps without line information

Inconsistent file paths
~~~~~~~~~~~~~~~~~~~~~~~~

Normalize paths to match gold annotations:

.. code-block:: python

   import os
   file_path = os.path.normpath(file_path)  # Remove ./, ../ etc.

Duplicate context
~~~~~~~~~~~~~~~~~

When computing cumulative context, use union operations:

.. code-block:: python

   from contextbench.core.intervals import union
   
   all_spans = {}
   for step in steps:
       for file, intervals in step["spans"].items():
           if file not in all_spans:
               all_spans[file] = []
           all_spans[file].extend(intervals)
   
   # Union overlapping intervals
   for file in all_spans:
       all_spans[file] = union(all_spans[file])

Next Steps
----------

- See :doc:`run_agent_on_contextbench` for batch evaluation
- Understand the :doc:`pipeline` for how trajectories are processed
- Explore :doc:`api/agents` for API reference
