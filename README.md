<div align="center">

# 🛡️ RAG Titan

**Enterprise & Production-Grade Retrieval-Augmented Generation (RAG) Architectures & Reference Implementations.**

Maintained by **[Divya (CodeBy-Divya)](https://github.com/CodeBy-Divya)**

[![Author](https://img.shields.io/badge/Author-CodeBy--Divya-orange?style=for-the-badge&logo=github)](https://github.com/CodeBy-Divya)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

**RAG Titan** is a curated repository containing production-grade RAG patterns, vector pipeline architectures, and end-to-end fullstack AI implementations. Each example provides a full blueprint from data ingestion to UI interaction.

---

## 📂 Prebuilt RAG Implementations

### ⚖️ [Sakhi — Multilingual Legal Reasoning RAG](./example%20with%20prebuilt%20rag%20model/sakhi)
Located in [`example with prebuilt rag model/sakhi`](./example%20with%20prebuilt%20rag%20model/sakhi):
- **Domain**: Legal advisory, rights education, and statutory citation across 40+ Indian Acts.
- **Workflow Engine**: **LangGraph** StateGraph featuring intent understanding, typo/slang normalization, 3-way multi-angle query rewrites, and score-filtered retrieval.
- **Embeddings**: `qwen/qwen3-embedding-8b` (OpenRouter API) with 4,096-dimensional multilingual embeddings.
- **Vector Database**: **ChromaDB** with HNSW indexing.
- **Fast Reasoning**: **Groq Cloud** running `openai/gpt-oss-120b`.
- **Client & Server**: **FastAPI** backend + **React 19 / Vite 6** dark glassmorphic chat interface.

Check out the full walkthrough: [Sakhi Architecture & Quickstart](./example%20with%20prebuilt%20rag%20model/sakhi/README.md).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
