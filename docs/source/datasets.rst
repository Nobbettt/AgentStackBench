Datasets
========

ContextBench provides a unified table of issue-resolution tasks augmented with gold contexts.

Hugging Face
------------

The dataset is available on Hugging Face:

- https://huggingface.co/datasets/Contextbench/ContextBench

It contains:

- ``default``: the full ContextBench table (single ``train`` split)
- ``contextbench_verified``: a 500-instance subset

Local files
-----------

This repo also ships Parquet/CSV files under ``data/`` for convenience.

- ``data/full.parquet``
- ``data/contextbench_verified.parquet``
- ``data/selected_500_instances.csv``

