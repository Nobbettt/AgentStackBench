Evaluation
==========

This page describes how to run ContextBench evaluations and interpret results.

Run a single evaluation
-----------------------

.. code-block:: bash

   python -m contextbench.evaluate \
       --gold data/full.parquet \
       --pred path/to/trajectory.traj.json \
       --out results.jsonl

Common options
--------------

- ``--gold``: Parquet/JSONL path containing gold contexts.
- ``--pred``: Path to an agent trajectory file.
- ``--cache``: Repository cache directory (optional).
- ``--out``: Output JSONL path.

Next steps
----------

- See :doc:`metrics` for metric definitions.
- See :doc:`run_agent_on_contextbench` for batch runs.
