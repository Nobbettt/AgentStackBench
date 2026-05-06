Quick Start
===========

This guide will help you get started with ContextBench evaluation in minutes.

Basic Usage
-----------

The simplest way to evaluate an agent trajectory:

.. code-block:: bash

   python -m contextbench.evaluate \
       --gold data/full.parquet \
       --pred path/to/trajectory.traj.json \
       --out results.jsonl

This command will:

1. Load the gold annotations from ``data/full.parquet``
2. Parse the trajectory file (auto-detects format)
3. Clone the target repository (if not cached)
4. Extract symbols using tree-sitter
5. Compute all metrics
6. Save results to ``results.jsonl``

Example with Trajectory
------------------------

Evaluate a single trajectory on the verified benchmark:

.. code-block:: bash

   cd /path/to/contextbench/ContextBench

   # Example with MiniSWE-agent trajectory
   python -m contextbench.evaluate \
       --gold data/full.parquet \
       --pred traj_verified/psf__requests-1142/psf__requests-1142.traj.json \
       --out result.jsonl

Understanding the Output
------------------------

The evaluation produces a JSON Lines file with detailed metrics for each instance:

.. code-block:: json

   {
     "instance_id": "psf__requests-1142",
     "num_steps": 6,
     "final": {
       "file": {"coverage": 1.0, "precision": 0.5, "f1": 0.667},
       "symbol": {"coverage": 1.0, "precision": 0.125, "f1": 0.222},
       "span": {"coverage": 1.0, "precision": 0.126, "f1": 0.224}
     },
     "trajectory": {
      "steps": [
        {"step": 1, "coverage": {"file": 0.5, "symbol": 0.3, "span": 0.2}},
        {"step": 2, "coverage": {"file": 0.8, "symbol": 0.6, "span": 0.5}}
      ],
       "auc_coverage": {"file": 1.0, "symbol": 1.0, "span": 1.0},
       "redundancy": {"file": 0.5, "symbol": 0.58, "span": 0.12}
       },
     "editloc": {"recall": 0.782, "precision": 1.0, "f1": 0.878}
   }

Key Metrics
~~~~~~~~~~~

- **Coverage**: What fraction of gold context was retrieved?
- **Precision**: What fraction of retrieved context was relevant?
- **F1**: Harmonic mean of coverage and precision
- **AUC-Coverage**: Area under the coverage curve (trajectory efficiency)
- **Redundancy**: How much context was re-examined?

Environment Variables
---------------------

You can set default paths to avoid repeating arguments:

.. code-block:: bash

   export CONTEXTBENCH_GOLD=/path/to/full.parquet
   export CONTEXTBENCH_CACHE=/path/to/repos_cache
   export CONTEXTBENCH_SELECTED_CSV=/path/to/selected_500_instances.csv

Then run evaluation with fewer arguments:

.. code-block:: bash

   python -m contextbench.evaluate \
       --pred path/to/trajectory.traj.json \
       --out result.jsonl

Next Steps
----------

- Read about :doc:`metrics` to understand the evaluation framework
- Learn how to :doc:`run_agent_on_contextbench` for batch evaluation
- Explore :doc:`agents` to see supported agent frameworks
