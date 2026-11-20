"""
The same saved-index RAG, built from its parts: you pick every class yourself.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/persistent_rag_parts.py
"""

from pathlib import Path

from rag_titan import SimpleRAG
from rag_titan.chunking import ParagraphChunker
from rag_titan.embeddings import OllamaEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import OllamaLLM
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import DenseRetriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import SQLiteStore

DOCS = Path(__file__).parent / "sample_docs"

# 1. The parts: the store is the only one that keeps anything between runs
llm = OllamaLLM("gemma3:4b")
embedder = OllamaEmbedder("nomic-embed-text")
store = SQLiteStore("hr_index_parts.db")      # or ChromaStore(...) / QdrantStore(...)

# 2. Indexing: load -> chunk -> embed -> store (first run only)
if store.count() == 0:
    pipeline = IndexingPipeline(ParagraphChunker(1000, 0), embedder, store)
    chunk_count = pipeline.run(sorted(DOCS.glob("*.md")))
    print(f"Indexed {chunk_count} chunks")
else:
    print(f"Using the saved index: {store.count()} chunks")

# 3. Answering: retrieve -> rerank -> generate (the embedder must be the one that indexed)
rag = SimpleRAG(DenseRetriever(embedder, store), KeywordReranker(), GroundedGenerator(llm))
print(rag.answer("How many paid leave days do I get?"))
