# Research References: LLM Evaluation and RAG

This document lists foundational research papers, tools, libraries, and resources utilized in designing the AI Response Quality Evaluator Agent.

---

## Academic Papers

1. **LLM-as-a-Judge**
   * *Title*: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)
   * *Link*: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
   * *Key Contribution*: Explored the viability of using strong models (like GPT-4) to evaluate chat assistants, identifying position, verbosity, and self-enhancement biases.

2. **RAGAS (RAG Assessment)**
   * *Title*: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
   * *Link*: [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
   * *Key Contribution*: Proposed metrics for evaluating RAG systems without ground-truth human labels, introducing mathematical formalizations for Faithfulness and Answer Relevance.

3. **Hallucination and Groundedness**
   * *Title*: "Self-CheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" (Manakul et al., 2023)
   * *Link*: [https://arxiv.org/abs/2303.08896](https://arxiv.org/abs/2303.08896)
   * *Key Contribution*: Discussed methods for checking factual consistency within generative outputs using multiple sampling passes.

4. **Dense Text Retrieval**
   * *Title*: "BGE Embeddings: C-Pack: Packaged Resources for General Chinese Embeddings" (Xiao et al., 2023)
   * *Link*: [https://arxiv.org/abs/2309.07597](https://arxiv.org/abs/2309.07597)
   * *Key Contribution*: Developed highly effective embedding representations for sentence retrieval (BAAI/bge models).

---

## Libraries and Frameworks

1. **Ragas Evaluation Framework**
   * *GitHub*: [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
   * *Documentation*: [https://docs.ragas.io/](https://docs.ragas.io/)

2. **TruLens Evaluation**
   * *GitHub*: [https://github.com/truera/trulens](https://github.com/truera/trulens)
   * *Documentation*: [https://www.trulens.org/](https://www.trulens.org/)

3. **FAISS (Facebook AI Similarity Search)**
   * *GitHub*: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
   * *Paper*: "Billion-scale similarity search with GPUs" (Johnson et al., 2017)

4. **LangChain**
   * *GitHub*: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
   * *Documentation*: [https://python.langchain.com/](https://python.langchain.com/)

---

## Datasets for Evaluation Benchmarks

1. **TruthfulQA**
   * *Description*: A benchmark to measure whether a language model is truthful in generating answers to questions. Designed to mimic human false beliefs and misconceptions.
   * *Link*: [https://arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958)

2. **SQuAD (Stanford Question Answering Dataset)**
   * *Description*: A reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles. Used for context extraction and retrieval benchmarks.
   * *Link*: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
