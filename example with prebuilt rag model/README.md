# 📂 Examples with Prebuilt RAG Models

This folder contains end-to-end, production-tested **Retrieval-Augmented Generation (RAG)** reference implementations.

## 🌟 Available Implementations

### 1. [Sakhi — Multilingual Legal RAG](./sakhi)
- **Domain**: Indian Statutory Law & Everyday Citizen Assistance
- **Orchestration**: LangGraph cyclic multi-node state graph
- **Vector Store**: ChromaDB (7,600+ legal chunks indexed)
- **Embedding**: Multilingual Qwen3-Embedding-8B (OpenRouter API)
- **Inference LLM**: Groq Cloud (`openai/gpt-oss-120b`)
- **Frontend**: React 19 + TypeScript + Vite 6 + Tailwind CSS
- **Backend**: FastAPI with async thread-pool invocation

👉 For complete architecture details, setup steps, and pipeline diagrams, see the [Sakhi README](./sakhi/README.md).
