AgentStackBench Documentation
=============================

.. centered:: **A benchmark harness for comparing coding-agent configurations**

.. centered:: Derived from ContextBench, with fork-specific support for agent runtime, prompt, tool, and evaluator experiments.

----

Overview
--------

AgentStackBench is a fork-derived benchmark harness for comparing coding-agent configurations across model, runtime, mounted resources, prompts, tools, and evaluator setup.

The project keeps the upstream ContextBench evaluator and dataset-processing path as the compatibility layer, while adding fork-specific support for running agents such as Codex and Claude, converting their run records to ContextBench-compatible trajectories, and publishing sanitized comparison data for the frontend.

The Python package remains ``contextbench`` so existing evaluator imports, command lines, and upstream integration points continue to work.

Key Features
------------

- **Run suites**: Compare multiple coding-agent setups over the same selected task set
- **Runtime controls**: Isolate agent runs, mount resources, and capture raw task artifacts
- **Context metrics**: Preserve ContextBench-compatible file, symbol, span, line, and edit-location scoring
- **Resolution metrics**: Route SWE-Bench Verified, SWE-bench Pro, SWE-PolyBench, and Multi-SWE-Bench tasks to their native evaluators
- **Frontend exports**: Publish distilled, sanitized comparison payloads without exposing private local paths

Quick Links
-----------

- **AgentStackBench repository**: https://github.com/Nobbettt/AgentStackBench
- **Upstream ContextBench repository**: https://github.com/EuniAI/ContextBench
- **ContextBench paper**: https://arxiv.org/abs/2602.05892
- **ContextBench dataset**: https://huggingface.co/datasets/Contextbench/ContextBench

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   evaluation
   agents
   metrics
   datasets

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage

   run_agent_on_contextbench
   process_trajectories
   environment_variables

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/parsers
   api/extractors
   api/metrics
   api/agents

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   leaderboard
   citation
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
