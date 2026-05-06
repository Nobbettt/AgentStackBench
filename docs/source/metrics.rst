Metrics
=======

ContextBench provides comprehensive metrics at multiple granularities to evaluate context retrieval in coding agents.

Granularity Levels
------------------

ContextBench evaluates context retrieval at four granularities:

1. **File-level**: Which files contain relevant context?
2. **Symbol-level**: Which functions/classes are relevant?
3. **Span-level**: What exact code regions are relevant?
4. **EditLoc-level**: Where should edits be made?

Core Metrics
------------

For each granularity, we compute:

Coverage (Recall)
~~~~~~~~~~~~~~~~~

What fraction of the gold context was retrieved?

.. math::

   \text{Coverage} = \frac{|\text{Gold} \cap \text{Pred}|}{|\text{Gold}|}

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**: Did the agent find all relevant context?

Precision
~~~~~~~~~

What fraction of the retrieved context was relevant?

.. math::

   \text{Precision} = \frac{|\text{Gold} \cap \text{Pred}|}{|\text{Pred}|}

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**: Did the agent avoid irrelevant context?

F1 Score
~~~~~~~~

Harmonic mean of coverage and precision:

.. math::

   \text{F1} = 2 \cdot \frac{\text{Coverage} \cdot \text{Precision}}{\text{Coverage} + \text{Precision}}

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**: Balanced measure of retrieval quality

Trajectory Metrics
------------------

In addition to final context, ContextBench tracks **how** agents retrieve context over time:

AUC-Coverage
~~~~~~~~~~~~

Area under the per-step coverage curve:

.. math::

   \text{AUC-Coverage} = \frac{1}{N} \sum_{i=1}^{N} \text{Coverage}_i

where :math:`N` is the number of steps and :math:`\text{Coverage}_i` is coverage at step :math:`i`.

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**: How quickly did the agent find relevant context?

A higher AUC-Coverage means the agent found relevant context earlier in the trajectory.

Redundancy
~~~~~~~~~~

Fraction of context that was re-examined:

.. math::

   \text{Redundancy} = \frac{\sum_{i=2}^{N} |\text{Context}_i \cap \bigcup_{j=1}^{i-1} \text{Context}_j|}{\sum_{i=1}^{N} |\text{Context}_i|}

- **Range**: [0, 1]
- **Lower is better**
- **Interpretation**: How much duplicate examination occurred?

High redundancy indicates the agent repeatedly viewed the same context.

Per-Step Coverage
~~~~~~~~~~~~~~~~~

Coverage at each trajectory step:

.. code-block:: python

   "trajectory": {
     "steps": [
       {"step": 1, "coverage": {"file": 0.5, "symbol": 0.3, "span": 0.2}},
       {"step": 2, "coverage": {"file": 0.8, "symbol": 0.6, "span": 0.5}},
       ...
     ]
   }

This enables analyzing the agent's exploration strategy over time.

Granularity-Specific Details
-----------------------------

File-Level Metrics
~~~~~~~~~~~~~~~~~~

**Set operations**: Treat file paths as sets

.. code-block:: python

   gold_files = {"src/utils.py", "src/main.py"}
   pred_files = {"src/utils.py", "src/config.py", "tests/test.py"}
   
   coverage = len(gold_files & pred_files) / len(gold_files)  # 0.5
   precision = len(gold_files & pred_files) / len(pred_files)  # 0.33

Symbol-Level Metrics
~~~~~~~~~~~~~~~~~~~~

**Symbol extraction**: Use tree-sitter to identify definitions

.. code-block:: python

   # Gold symbols (file, symbol_name)
   gold_symbols = {
       ("src/utils.py", "parse_config"),
       ("src/utils.py", "Config"),
       ("src/main.py", "main")
   }
   
   # Pred symbols: extracted from viewed spans
   pred_symbols = extract_symbols_from_spans(pred_spans)

Only definition nodes (classes, functions, methods) are considered symbols.

Span-Level Metrics
~~~~~~~~~~~~~~~~~~

**Interval operations**: Compute union and intersection of byte ranges

.. code-block:: python

   from contextbench.core.intervals import union, intersection, measure
   
   gold_spans = {"file.py": [(0, 100), (200, 300)]}
   pred_spans = {"file.py": [(50, 150), (250, 350)]}
   
   gold_union = union(gold_spans["file.py"])      # Total gold bytes
   pred_union = union(pred_spans["file.py"])      # Total pred bytes
   overlap = intersection(gold_union, pred_union)  # Overlapping bytes
   
   coverage = measure(overlap) / measure(gold_union)
   precision = measure(overlap) / measure(pred_union)

EditLoc-Level Metrics
~~~~~~~~~~~~~~~~~~~~~

**Line-based comparison**: Compare sets of edited line numbers

.. code-block:: python

   # Gold: lines where the ground-truth patch made changes
   gold_edit_lines = {15, 16, 17, 42, 43}
   
   # Pred: lines the agent identified for editing
   pred_edit_lines = {16, 17, 18, 42, 100}
   
   coverage = len(gold_edit_lines & pred_edit_lines) / len(gold_edit_lines)
   precision = len(gold_edit_lines & pred_edit_lines) / len(pred_edit_lines)

Aggregation Methods
-------------------

Macro Average
~~~~~~~~~~~~~

Mean over instances (each instance counts equally):

.. math::

   \text{Macro-Avg}(\text{metric}) = \frac{1}{N} \sum_{i=1}^{N} \text{metric}_i

**Use case**: Standard benchmark reporting

Micro Average
~~~~~~~~~~~~~

Aggregate sizes first, then compute metric:

.. math::

   \text{Micro-Coverage} = \frac{\sum_{i=1}^{N} |\text{Gold}_i \cap \text{Pred}_i|}{\sum_{i=1}^{N} |\text{Gold}_i|}

**Use case**: Size-weighted performance (larger tasks have more influence)

Interpreting Results
--------------------

High Coverage, Low Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent retrieved most relevant context but also retrieved much irrelevant context.

**Example**: ``coverage=0.9, precision=0.2``

**Interpretation**: Agent uses a broad search strategy (recall-focused)

Low Coverage, High Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent retrieved little context, but most of it was relevant.

**Example**: ``coverage=0.3, precision=0.8``

**Interpretation**: Agent uses a narrow search strategy (precision-focused)

Balanced F1
~~~~~~~~~~~

The agent achieves a good balance between coverage and precision.

**Example**: ``coverage=0.7, precision=0.6, f1=0.65``

**Interpretation**: Agent effectively retrieves relevant context without excessive noise

High Redundancy
~~~~~~~~~~~~~~~

The agent repeatedly examines the same context.

**Example**: ``redundancy=0.6``

**Interpretation**: Inefficient exploration; 60% of viewed context was already seen

Low AUC-Coverage
~~~~~~~~~~~~~~~~

The agent takes many steps to find relevant context.

**Example**: ``auc_coverage=0.3``

**Interpretation**: Relevant context is found late in the trajectory

Next Steps
----------

- See :doc:`evaluation` for running evaluations
- Understand the :doc:`pipeline` for how metrics are computed
- Explore :doc:`api/metrics` for implementation details
