ContextBench Documentation
==========================

.. image:: ../assets/branding/contextbench-logo.jpg
   :align: center
   :width: 200px
   :alt: ContextBench Logo

.. centered:: **A Comprehensive Benchmark for Evaluating Context Retrieval in Code Agents**

.. centered:: A collaboration between Nanjing University and University College London

----

Overview
--------

LLM-based coding agents have shown strong performance on automated issue resolution benchmarks, yet existing evaluations largely focus on final task success, providing limited insight into how agents retrieve and use code context during problem solving.

We introduce **ContextBench**, a process-oriented evaluation of context retrieval in coding agents. ContextBench consists of **1,136** issue-resolution tasks from 66 repositories across eight programming languages, each augmented with human-annotated gold contexts. We further implement an automated evaluation framework that tracks agent trajectories and measures context recall, precision, and efficiency throughout issue resolution.

Using ContextBench, we evaluate four frontier LLMs and five coding agents. Our results show that sophisticated agent scaffolding yields only marginal gains in context retrieval (**"The Bitter Lesson"** of coding agents), LLMs consistently favor recall over precision, and substantial gaps exist between explored and utilized context.

ContextBench augments existing end-to-end benchmarks with intermediate gold-context metrics that unbox the issue-resolution process. These contexts offer valuable intermediate signals for guiding LLM reasoning in software tasks.

Key Features
------------

- **Multi-granularity metrics**: File, symbol, span, and edit-location analysis
- **Trajectory-aware**: Per-step coverage tracking with AUC and redundancy metrics
- **Multi-language support**: Python, Java, JavaScript, TypeScript, Go, Rust, C, C++
- **Agent-agnostic**: Unified extractors for multiple agent frameworks
- **Tree-sitter powered**: Precise symbol extraction across programming languages
- **Reproducible**: Deterministic evaluation with cached repository snapshots

Quick Links
-----------

- **GitHub Repository**: https://github.com/EuniAI/ContextBench
- **Paper**: https://arxiv.org/abs/2602.05892
- **Dataset**: https://huggingface.co/datasets/Contextbench/ContextBench
- **Live Leaderboard**: https://contextbench.github.io/

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
