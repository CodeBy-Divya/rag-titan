"""
Parent-Child RAG next to Simple RAG on a question whose answer is a whole section.

"What do I have to do before my last day?" is answered by six steps spread over
one section of examples/parent_child_docs/offboarding.md. Both RAG types cut the
files into 25-word pieces, search the same way and keep two pieces; Parent-Child
RAG then gives the model the whole section each kept piece sits in. (Its pieces
are cut inside each section, so their ids differ from Simple RAG's.)

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/parent_child_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "parent_child_docs").glob("*.md"))
question = "What do I have to do before my last day?"

# the same knobs for both RAG types: the difference is the expand step
shared = {
    "chunker": "word",       # small pieces: word, sentence, paragraph, ... (CHUNKER)
    "chunk_size": 25,        # 25 words per piece (CHUNK_SIZE)
    "chunk_overlap": 0,      # words repeated in the next piece (CHUNK_OVERLAP)
    "retriever": "dense",    # dense (meaning), bm25 (words) or hybrid (both) (RETRIEVER)
    "reranker": "keyword",   # keyword, none, mmr, llm or cross_encoder (RERANKER)
    "top_k": 6,              # pieces the search finds (TOP_K)
    "top_n": 2,              # pieces kept after reranking (TOP_N)
}
parent_child_only = {
    "parent_size": 1500,     # largest section, in characters, cut when indexing (PARENT_SIZE)
    "parent_budget": 3000,   # most characters of sections given to the model (PARENT_BUDGET)
}

print(f"Question: {question}")
for rag_type, settings in [("simple", shared), ("parent_child", {**shared, **parent_child_only})]:
    rag = RAGTitan(rag_type=rag_type, settings=settings)
    rag.index(files)
    result = rag.run(question)

    given = result["sources"]
    size = sum(len(chunk["text"]) for chunk in given)
    print(f"\n{rag_type}: the model reads {len(given)} text(s), {size} characters")
    for chunk in given:
        # a parent section names the small chunks (children) that led to it
        name = chunk.get("parent_id", chunk["id"])
        found = f" (found by {', '.join(chunk['children'])})" if "children" in chunk else ""
        text = " ".join(chunk["text"].split())
        print(f"  {name}{found}: {text[:80]}...")
    print(f"Answer:\n{result['answer']}")
