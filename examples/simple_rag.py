"""
Simple RAG with every knob set and explained: search once, keep the best chunks,
answer once.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/simple_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "sample_docs").glob("*.md"))

settings = {
    # how the files are cut (CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    "chunker": "paragraph",  # word, character, token, recursive, sentence, paragraph, ...
    "chunk_size": 300,       # characters for paragraph (words for word, tokens for token)
    "chunk_overlap": 0,      # how much of a chunk is repeated in the next one
    # how chunks are found and chosen (RETRIEVER, RERANKER, TOP_K, TOP_N)
    "retriever": "dense",    # dense (meaning), bm25 (exact words) or hybrid (both, fused)
    "reranker": "keyword",   # keyword, none, mmr (variety), llm (model rates) or cross_encoder
    "top_k": 5,              # chunks the search finds
    "top_n": 3,              # chunks the model reads (default 2: too few for this question)
}

rag = RAGTitan(rag_type="simple", settings=settings)
rag.index(files)

result = rag.run("What do I get for working from home: equipment and money for furniture?")
print(result["log"])                 # every step: what went in, what came out
print(f"\nAnswer: {result['answer']}")
