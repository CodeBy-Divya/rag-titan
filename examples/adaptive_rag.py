"""
Adaptive RAG with every knob set and explained: it chooses how hard to search,
checks its evidence, and rewrites the question and searches wider when the
evidence is not enough.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/adaptive_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "sample_docs").glob("*.md"))

settings = {
    # how the files are cut (CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    "chunker": "paragraph",
    "chunk_size": 300,
    "chunk_overlap": 0,
    # how chunks are found and chosen (RETRIEVER, RERANKER)
    "retriever": "dense",
    "reranker": "keyword",
    # how much it searches (TOP_K, TOP_N): for a "simple" question 4 and 2; a "complex"
    # question searches twice as much, and every retry adds top_k and top_n once more
    "top_k": 4,
    "top_n": 2,
    # searches before it gives up and says it does not know (ADAPTIVE_MAX_ATTEMPTS)
    "adaptive_max_attempts": 2,
}

rag = RAGTitan(rag_type="adaptive", settings=settings)
rag.index(files)

for question in ["Hi there!", "What is the maximum hotel cost per night?"]:
    result = rag.run(question)
    print(f"Question: {question}")
    print(result["log"])
    print(f"Answer: {result['answer']}\n")
