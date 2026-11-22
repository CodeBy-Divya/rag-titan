"""
HyDE RAG with every knob set and explained: the model writes a made-up answer,
both the made-up answer and the question are searched, and the two lists are
fused. The answer comes from the real chunks only.

The question uses other words than the documents ("gear", "newcomer" for
"equipment", "new employees"), and keyword search (bm25) only finds shared words,
so a plain search finds nothing here.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/hyde_rag.py
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
    "retriever": "bm25",       # exact words: the made-up answer brings the documents' words
    "reranker": "keyword",     # ranks the fused chunks against the original question
    "top_k": 5,                # chunks the fused search returns
    "top_n": 2,                # chunks the model reads
    # HyDE's own knob (HYDE_WITH_QUESTION): also search with the question and fuse
    # both lists; False searches with the made-up answer alone, like the paper
    "hyde_with_question": True,
}

rag = RAGTitan(rag_type="hyde", settings=settings)
rag.index(files)
result = rag.run(question)

steps = {step["step"]: step for step in result["steps"]}
print(f"Question: {question}")
print(f"Made-up answer (only used to search): {steps['hypothesize']['output']}")
searches = [step for step in result["steps"] if step["step"] == "retrieve"]
for number, search in enumerate(searches, start=1):
    files_found = sorted({chunk["source"] for chunk in search["output"]})
    print(f"Search {number} finds: {', '.join(files_found) or 'nothing'}")
print(f"The model reads: {', '.join(chunk['source'] for chunk in result['sources'])}")
print(f"Answer: {result['answer']}")
