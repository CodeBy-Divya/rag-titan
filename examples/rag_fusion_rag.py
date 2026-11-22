"""
RAG-Fusion with every knob set and explained: the model writes a few rewordings
of the question, the question and every rewording are searched, and the result
lists are fused (Reciprocal Rank Fusion), so chunks found by several wordings
come first.

The question uses other words than the documents ("gear", "newcomer" for
"equipment", "new employees"), and keyword search (bm25) only finds shared words,
so a plain search finds nothing here.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/rag_fusion_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "retrieval_docs").glob("*.md"))
question = "What gear does a newcomer receive?"

settings = {
    # how the files are cut (CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    "chunker": "paragraph",
    "chunk_size": 300,
    "chunk_overlap": 0,
    # how chunks are found and chosen (RETRIEVER, RERANKER, TOP_K, TOP_N)
    "retriever": "bm25",       # exact words: the rewordings bring other words
    "reranker": "keyword",     # ranks the fused chunks against the original question
    "top_k": 5,                # chunks the fused search returns
    "top_n": 2,                # chunks the model reads
    # RAG-Fusion's own knob (FUSION_QUERIES): rewordings to write, 1 to 10;
    # each one is a search, next to the question's
    "fusion_queries": 3,
}

rag = RAGTitan(rag_type="rag_fusion", settings=settings)
rag.index(files)
result = rag.run(question)

steps = {step["step"]: step for step in result["steps"]}
queries = [question, *steps["write_queries"]["output"]]
print(f"Question: {question}")
for number, query in enumerate(queries, start=1):
    print(f"  query {number}: {query}")
print("Fused, best first (each chunk's rank for every query that found it):")
for chunk in steps.get("fuse_queries", {}).get("output", []):
    ranks = ", ".join(f"{name.replace('_rank', '').replace('_', ' ')} #{rank}"
                      for name, rank in chunk["score_trace"].items() if name.endswith("_rank"))
    print(f"  {chunk['source']:18} {ranks}")
print(f"The model reads: {', '.join(chunk['source'] for chunk in result['sources'])}")
print(f"Answer: {result['answer']}")
