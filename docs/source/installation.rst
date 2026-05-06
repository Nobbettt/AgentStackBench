.. SPDX-License-Identifier: Apache-2.0
.. Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
.. Summary of changes: document Python-version-specific tree-sitter installation.

Installation
============

Requirements
------------

- Python 3.8 or higher
- Git (for cloning repositories)
- Required Python packages (see below)

Install from Source
-------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/EuniAI/ContextBench.git
   cd ContextBench

Install Dependencies
--------------------

**Recommended**: Install pinned runtime dependencies:

.. code-block:: bash

   pip install -r requirements.txt

**Critical**: Tree-sitter is required for symbol extraction. Install the variant
that matches your Python version:

.. code-block:: bash

   # Python 3.10-3.12
   pip install "tree-sitter==0.20.4" tree-sitter-languages

   # Python 3.13+
   pip install "tree-sitter>=0.24,<0.25" tree-sitter-language-pack

Verify Installation
-------------------

Test that ContextBench is correctly installed:

.. code-block:: bash

   python -m contextbench.evaluate --help

You should see the command-line help message for the evaluation module.

Optional: Development Installation
-----------------------------------

If you plan to contribute or modify the code:

.. code-block:: bash

   pip install -e ".[dev]"

This installs ContextBench in editable mode with the development dependencies
used by the local test suite.

Troubleshooting
---------------

**Tree-sitter installation issues**

If you encounter errors with tree-sitter, make sure you have a C compiler available:

- **Linux**: ``sudo apt-get install build-essential``
- **macOS**: ``xcode-select --install``
- **Windows**: Install Visual Studio Build Tools

**Permission errors**

If you encounter permission errors, consider using a virtual environment:

.. code-block:: bash

   python -m venv contextbench_env
   source contextbench_env/bin/activate  # On Windows: contextbench_env\\Scripts\\activate
   pip install -r requirements.txt
