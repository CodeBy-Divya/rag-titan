<div align="center">

# 🛡️ RAG Titan

**A Modular Swiss Army Knife for Production-Grade Retrieval-Augmented Generation (RAG) & 7-Stage Pipeline Evaluation.**

Maintained by **[Divya (CodeBy-Divya)](https://github.com/CodeBy-Divya)**

[![Author](https://img.shields.io/badge/Author-CodeBy--Divya-orange?style=for-the-badge&logo=github)](https://github.com/CodeBy-Divya)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Tests](https://github.com/CodeBy-Divya/rag-titan/actions/workflows/tests.yml/badge.svg)](https://github.com/CodeBy-Divya/rag-titan/actions/workflows/tests.yml)

</div>

---

## 📖 Overview

**RAG Titan** is an enterprise-ready, modular RAG toolkit and evaluation engine. It lets you build production-ready RAG architectures with swappable components, run completely locally with Ollama (or zero-dependency mocks for testing), and evaluate every single pipeline stage mathematically with 70+ metrics and a deterministic 0–100 scorecard.

---

## 🌟 Why RAG Titan?

- **Many RAG Types, One Unified API**: Switch between `simple` and `adaptive` RAG workflows with a single parameter: `rag_type="adaptive"`.
- **100% Local & Private**: Runs locally with [Ollama](https://ollama.com), ensuring sensitive documents never leave your machine.
- **Grounded Answers & Accurate Citations**: Generates answers strictly from retrieved documents and cites exact source chunks.
- **8 Granular Chunking Strategies**: Word, character, token (via tiktoken or custom tokenizer), recursive separator, sentence-aware, paragraph-aware, heading-aware, and page-aware chunkers.
- **Multi-Format Ingestion**: Supports PDF, Word (.docx), Markdown, and plain text files with document structure preserved.
- **Stage-by-Stage 7-Stage Evaluation**: Scores document preparation, semantic indexing, information search, search ordering, answer context, answer quality, and UX without needing an expensive or nondeterministic LLM judge.
- **Regression Detection**: Catch regressions against saved baseline scorecards directly in CI/CD using `--fail-on-regression`.
- **Zero-Boilerplate CLI**: Ask questions, stream tokens, view step-by-step traces (`--trace`), and run full benchmark evaluations from the terminal.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/CodeBy-Divya/rag-titan.git
cd rag-titan

# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# (Optional) For token-based chunking with tiktoken
pip install -e ".[tokens]"

# Pull models with Ollama
ollama pull gemma3:4b && ollama pull nomic-embed-text
```

---

## 🚀 Quick Start

### Python API

```python
from rag_titan import RAGTitan

# Initialize Simple or Adaptive RAG
rag = RAGTitan(rag_type="simple")   # or "adaptive"

# Ingest and index documents
rag.index(["handbook.pdf", "policy.docx"])

# Ask questions with citations
answer = rag.ask("How many leave days do I get?")
print(answer)
# Output: You get 20 paid leave days every year [chunk_0].

# Inspect every step taken by the pipeline
run_trace = rag.run("How many leave days do I get?")
print(run_trace["log"])

# Evaluate on test questions
rag.evaluate("examples/questions.json")
```

### Command Line Interface (CLI)

```bash
# Ask a direct question over documents
rag-titan ask --files handbook.pdf --question "How many leave days do I get?"

# Inspect execution trace with latencies and stage steps
rag-titan ask --files handbook.pdf --question "How many leave days do I get?" --trace

# Interactive chat session
rag-titan chat --files handbook.pdf

# Evaluate retrieval and generation across chunkers & RAG types
rag-titan evaluate --files examples/sample_docs/*.md --questions examples/questions.json \
    --rag-type simple adaptive --chunker word paragraph
```

---

## 📊 End-to-End & Stage-by-Stage Evaluation

RAG Titan provides a rigorous mathematical evaluation engine that isolates exactly which component failed in your pipeline:

```text
| Pipeline stage       | Score / 100 | Rating    | Measurement coverage |
|----------------------|------------:|-----------|---------------------:|
| Document preparation |        97.6 | Excellent |                  74% |
| Semantic indexing    |        98.4 | Excellent |                 100% |
| Information search   |        79.4 | Good      |                 100% |
| Search ordering      |        89.7 | Very Good |                 100% |
| Answer context       |        92.3 | Excellent |                 100% |
| Answer quality       |       100.0 | Excellent |                 100% |
| User experience      |       100.0 | Excellent |                 100% |
```

### Python Evaluation Example

```python
from rag_titan.evaluation import RAGEvaluator

evaluator = RAGEvaluator(k=5)
report = evaluator.evaluate(
    chunks=chunks,
    queries=queries,
    document_text=text,
)

print(f"Overall Score: {report['scorecard']['overall']['score']}/100")
print(f"First Failing Stage: {report['root_causes']['first_failing_stage']}")
```

---

## ⚖️ Prebuilt RAG Implementations

### [Sakhi — Multilingual Legal Reasoning RAG](./example%20with%20prebuilt%20rag%20model/sakhi)
Located in [`example with prebuilt rag model/sakhi`](./example%20with%20prebuilt%20rag%20model/sakhi):
- **Domain**: Legal advisory, rights education, and statutory citation across 40+ Indian Acts.
- **Workflow Engine**: **LangGraph** StateGraph featuring intent classification, slang normalization, 3-way multi-angle query rewrites, and score-filtered retrieval.
- **Embeddings**: `qwen/qwen3-embedding-8b` with 4,096-dimensional multilingual embeddings.
- **Vector Database**: **ChromaDB** with HNSW indexing.
- **Fast Reasoning**: **Groq Cloud** running `openai/gpt-oss-120b`.
- **Client & Server**: **FastAPI** backend + **React 19 / Vite 6** dark glassmorphic chat interface.

Check out the full walkthrough: [Sakhi Architecture & Quickstart](./example%20with%20prebuilt%20rag%20model/sakhi/README.md).

---

## 🧩 Supported RAG Architectures

| RAG Type | Parameter | Workflow | Best For | LLM Calls |
|---|---|---|---|---|
| **Simple RAG** | `rag_type="simple"` | Retrieve → Rerank → Generate | Fast answers to clear queries | 1 |
| **Adaptive RAG** | `rag_type="adaptive"` | Route → Retrieve → Check Evidence → Rewrite & Retry | Ambiguous queries, complex lookups, hallucination mitigation | 2–5 |

---

## 🧪 Running Tests

Tests run quickly and require no external services or Ollama instances:

```bash
# Run test suite
pytest

# Run linter
ruff check .
```

---

## 🤝 Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [Divya (CodeBy-Divya)](https://github.com/CodeBy-Divya).
