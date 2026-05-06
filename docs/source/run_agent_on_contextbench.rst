.. SPDX-License-Identifier: Apache-2.0
.. Fork note: Modified by Norbert Laszlo on 2026-03-16 from upstream ContextBench.
.. Summary of changes: document fork-specific Codex CLI and Claude Code CLI wrapper support.

Run Agents on ContextBench
==========================

ContextBench includes a unified runner for executing agents and collecting trajectories.

Runner entrypoint
-----------------

Use the module entrypoint:

.. code-block:: bash

   python -m contextbench.run --help

Common examples
---------------

.. code-block:: bash

   # Run agentless on Verified
   python -m contextbench.run --agent agentless --bench Verified

   # Run MiniSWE on Pro, first 5 instances
   python -m contextbench.run --agent miniswe --bench Pro --limit 5

   # Run the Codex CLI wrapper on the default selected slice
   python -m contextbench.run --agent codex --limit 5

   # Run the Claude Code CLI wrapper on Poly
   python -m contextbench.run --agent claude --bench Poly --limit 3

Task lists
----------

By default the runner reads:

- ``data/selected_500_instances.csv``

For ``codex`` and ``claude``, the runner also reads a prompt-capable task source via ``--task-data``. The default is ``data/full.parquet``.

See also
--------

- The Markdown guide: ``docs/run_agent_on_contextbench.md``
