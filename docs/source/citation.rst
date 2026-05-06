Citation
========

If you use ContextBench in your research, please cite our paper:

BibTeX
------

.. code-block:: bibtex

   @misc{li2026contextbenchbenchmarkcontextretrieval,
     title={ContextBench: A Benchmark for Context Retrieval in Coding Agents}, 
     author={Han Li and Letian Zhu and Bohan Zhang and Rili Feng and Jiaming Wang and Yue Pan and Earl T. Barr and Federica Sarro and Zhaoyang Chu and He Ye},
     year={2026},
     eprint={2602.05892},
     archivePrefix={arXiv},
     primaryClass={cs.LG},
     url={https://arxiv.org/abs/2602.05892}
   }

Paper
-----

The full paper is available at: https://arxiv.org/abs/2602.05892

Abstract
--------

LLM-based coding agents have shown strong performance on automated issue resolution benchmarks, yet existing evaluations largely focus on final task success, providing limited insight into how agents retrieve and use code context during problem solving.

We introduce **ContextBench**, a process-oriented evaluation of context retrieval in coding agents. ContextBench consists of **1,136** issue-resolution tasks from 66 repositories across eight programming languages, each augmented with human-annotated gold contexts. We further implement an automated evaluation framework that tracks agent trajectories and measures context recall, precision, and efficiency throughout issue resolution.

Using ContextBench, we evaluate four frontier LLMs and five coding agents. Our results show that sophisticated agent scaffolding yields only marginal gains in context retrieval (**"The Bitter Lesson"** of coding agents), LLMs consistently favor recall over precision, and substantial gaps exist between explored and utilized context.

ContextBench augments existing end-to-end benchmarks with intermediate gold-context metrics that unbox the issue-resolution process. These contexts offer valuable intermediate signals for guiding LLM reasoning in software tasks.

Related Work
------------

If you use specific components, please also consider citing:

Tree-sitter
~~~~~~~~~~~

For symbol extraction:

.. code-block:: bibtex

   @misc{tree-sitter,
     title={Tree-sitter},
     author={Max Brunsfeld},
     year={2018},
     url={https://tree-sitter.github.io/tree-sitter/}
   }

SWE-bench
~~~~~~~~~

For the base benchmark:

.. code-block:: bibtex

   @inproceedings{jimenez2024swebench,
     title={SWE-bench: Can Language Models Resolve Real-world GitHub Issues?},
     author={Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
     booktitle={ICLR},
     year={2024}
   }

Acknowledgements
----------------

ContextBench is a collaborative research project between:

- **Nanjing University** (南京大学)
- **University College London**

We thank the developers of the agent frameworks evaluated in this benchmark:

- Agentless
- SWE-agent
- Mini-SWE-Agent
- OpenHands
- Prometheus

We gratefully acknowledge **Mistral AI** and **Amazon Web Services (AWS)** for providing API support that enabled large-scale experiments and evaluations.

We also thank the open-source community for their contributions to the tools and libraries that make ContextBench possible.

Contact
-------

For questions, suggestions, or collaborations, please:

- **Open an issue**: https://github.com/EuniAI/ContextBench/issues
- **Email**: [contact email]
- **Website**: https://contextbench.github.io/

License
-------

ContextBench is released under the **Apache License 2.0**. See the `LICENSE <https://github.com/EuniAI/ContextBench/blob/main/LICENSE>`_ file for details.
