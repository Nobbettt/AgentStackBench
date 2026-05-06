Leaderboard
===========

Live Leaderboard
----------------

Visit the **interactive leaderboard** at: https://contextbench.github.io/

The live leaderboard features:

- Real-time rankings of all evaluated models and agents
- Interactive filtering by benchmark variant (Verified, Pro, Poly, Multi)
- Detailed breakdowns by granularity (File, Symbol, Span, EditLoc)
- Trajectory efficiency metrics (AUC-Coverage, Redundancy)
- Cost analysis and performance comparisons

Current Rankings
----------------

Main Board (Verified Split)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Top-performing systems on the ContextBench Verified benchmark (500 instances):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 15 15

   * - Agent
     - File Cov. ↑
     - File Prec. ↑
     - Symbol Cov. ↑
     - Symbol Prec. ↑
     - AUC-Cov ↑
     - Redun. ↓
   * - Prometheus
     - 0.799
     - 0.346
     - 0.716
     - 0.255
     - 0.598
     - 0.422
   * - Agentless
     - 0.656
     - 0.398
     - 0.357
     - 0.393
     - 0.056
     - 0.000
   * - SWE-Agent
     - 0.576
     - 0.496
     - 0.436
     - 0.233
     - 0.563
     - 0.094

Backbone Model Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance of different LLM backbones (using Mini SWE-agent):

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15 15

   * - Backbone
     - Pass@1 ↑
     - Context F1 ↑
     - Efficiency ↑
     - File F1 ↑
     - Symbol F1 ↑
     - Span F1 ↑
   * - Claude Sonnet 4.5
     - 53.0%
     - 0.344
     - 0.658
     - 0.468
     - 0.496
     - 0.468
   * - GPT-5
     - 47.2%
     - 0.312
     - 0.591
     - 0.468
     - 0.496
     - 0.468
   * - Devstral 2
     - 40.2%
     - 0.332
     - 0.616
     - 0.384
     - 0.489
     - 0.456
   * - Gemini 2.5 Pro
     - 36.4%
     - 0.311
     - 0.529
     - 0.460
     - 0.433
     - 0.362

Key Findings
------------

The Bitter Lesson of Coding Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sophisticated agent scaffolding yields only **marginal gains** in context retrieval compared to raw LLM capabilities. The backbone model choice has a much larger impact than agent architecture.

Recall vs. Precision Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLMs consistently favor **recall over precision**, retrieving broad context at the cost of including irrelevant code. This suggests a need for better filtering mechanisms.

Explored vs. Utilized Context Gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Substantial gaps exist between context that agents **explore** (view) and context they actually **utilize** (reference in edits). This indicates inefficient context usage.

Submitting Results
------------------

To submit your agent's results to the leaderboard:

1. **Run evaluation** on all benchmark splits
2. **Generate results** in the standard JSON Lines format
3. **Submit via GitHub** by opening a pull request with your results
4. **Include metadata**: Model name, version, cost, and configuration

See the `submission guidelines <https://github.com/EuniAI/ContextBench/blob/main/SUBMISSION.md>`_ for details.

Evaluation Criteria
-------------------

Submissions are ranked by:

1. **Primary metric**: Pass@1 (task success rate)
2. **Context F1**: Balanced file/symbol/span F1 score
3. **Efficiency**: AUC-Coverage (how quickly relevant context is found)
4. **Cost**: Average inference cost per instance

Benchmark Variants
------------------

Verified
~~~~~~~~

- **Size**: 500 instances
- **Source**: SWE-bench Verified
- **Difficulty**: Moderate
- **Languages**: Primarily Python

Pro
~~~

- **Size**: 2,294 instances
- **Source**: SWE-bench Pro
- **Difficulty**: High
- **Languages**: Python, Java, JavaScript

Poly
~~~~

- **Size**: 640 instances
- **Source**: SWE-PolyBench
- **Difficulty**: Moderate-High
- **Languages**: Python, Java, JavaScript, TypeScript, Go, Rust, C, C++

Multi
~~~~~

- **Size**: 1,000+ instances
- **Source**: Multi-SWE-bench
- **Difficulty**: High
- **Languages**: Multiple

Next Steps
----------

- Visit the **live leaderboard**: https://contextbench.github.io/
- Learn how to :doc:`run_agent_on_contextbench`
- Understand the :doc:`metrics` used for ranking
- See :doc:`citation` to reference ContextBench in your work
